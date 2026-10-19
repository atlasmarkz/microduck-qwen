"""Raw held-out action decisions and independent, unshielded actor task rollouts."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from . import ROOT
from .agent import AgentPlanner
from .agent_data import episode, load_splits, scene_profile, tasks
from .sim_data import digest
from .sim_runner import start_scene, stop_process
from .skills import parse_action


def semantic_match(raw, expected):
    """Diagnostic only: strip one code fence to separate formatting from task errors.

    Runtime remains strict. This score NEVER authorizes execution or training labels.
    """
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return parse_action(raw) == expected
    except (ValueError, TypeError, KeyError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--seed", type=int, default=91010)
    parser.add_argument("--port", type=int, default=7872)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--scene-mix", choices=("plain", "hard", "yellow"), default="plain",
                        help="rollout scene difficulty; must match the collected data")
    args = parser.parse_args()
    manifest, splits = load_splits(args.data)
    if args.rollout and args.seed in {row["scene_seed"] for values in splits.values() for row in values}:
        parser.error("rollout seed must be outside all training/development/test scenes")
    if any(i < 0 or i >= len(tasks(args.split)) for i in args.task_ids):
        parser.error("invalid task ID")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    planner = AgentPlanner(args.adapter)
    report = {"mode": "independent_actor_rollout" if args.rollout else "offline_action_exact_match",
              "adapter": planner.adapter, "split": args.split, "dataset_manifest_sha256": digest(args.data / "manifest.json"),
              "results": [], "correct": 0, "total": 0,
              "note": "Development pilot. Rollout teacher is scorer AFTER execution, never a semantic intervention."}
    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    process = log = None
    try:
        if args.rollout:
            occlusion, clutter = scene_profile(args.split, 0, args.scene_mix)
            process, log, info, failures = start_scene(output, args.seed, occlusion, args.port,
                                                       collect_gt=True, clutter=clutter, yellow_course=args.scene_mix == "yellow")
            report.update(scene_seed=args.seed, scene_mix=args.scene_mix,
                          occlusion=occlusion, clutter=clutter, startup_failures=failures)
            for i in args.task_ids:
                instruction, specs = tasks(args.split)[i]
                _, entry = episode(output, output / f"task-{i:02d}", info, args.split, args.seed, instruction, specs, planner, False)
                result = {"task_id": i, "correct": entry["verified"], "error": entry.get("error"), "episode": str(output / f"task-{i:02d}/episode.json")}
                report["results"].append(result)
                report["correct"] += result["correct"]
                report["total"] += 1
                save()
                print(json.dumps(result, ensure_ascii=False), flush=True)
        else:
            for row in splits[args.split]:
                prediction = planner.predict(row["instruction"], args.data / row["image"], row["memory"])
                result = {"id": row["id"], "task_type": row["task_type"], "expected": row["expected"],
                          "prediction": prediction, "correct": prediction.get("action") == row["expected"],
                          "semantic_correct_after_fence_removal": semantic_match(prediction["raw"], row["expected"])}
                report["results"].append(result)
                report["correct"] += result["correct"]
                report["total"] += 1
                report["semantic_correct_after_fence_removal"] = sum(r["semantic_correct_after_fence_removal"] for r in report["results"])
                save()
                print(f"{row['id']}: {result['correct']}", flush=True)
    finally:
        if process:
            stop_process(process)
        if log:
            log.close()
        save()
    print(f"AGENT EVALUATION: {output} ({report['correct']}/{report['total']})", flush=True)


if __name__ == "__main__":
    main()
