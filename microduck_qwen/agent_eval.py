"""Raw held-out action decisions and independent, unshielded actor task rollouts."""

import argparse
from datetime import datetime
import json
import os
import shutil
from pathlib import Path

from . import ROOT
from .agent import AgentPlanner
from .agent_data import AMBIGUOUS_TEACHER_TARGET, episode, explore_entries, load_splits, scene_profile, tasks, world_arguments
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


def wilson_interval(correct, total, z=1.96):
    """95% Wilson score interval for a success rate; (0, 1) when nothing was measured."""
    if total <= 0:
        return 0.0, 1.0
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--seed", type=int, default=91010)
    parser.add_argument("--rollout-seeds", type=int, default=1,
                        help="independent layouts for --rollout (seed, seed+1, ...); each is a separate simulator scene")
    parser.add_argument("--explore-tasks", type=int, default=0,
                        help="world mix: also run explore-and-approach tasks per colour hidden from the first view")
    parser.add_argument("--phrase-variant", type=int, default=0, choices=range(4),
                        help="validation split only: 0 = the original held-out wording, 1..3 = alternative held-out wordings")
    parser.add_argument("--port", type=int, default=7872)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--scene-mix", choices=("plain", "hard", "yellow", "world"), default="plain",
                        help="rollout scene difficulty; must match the collected data")
    parser.add_argument("--world-level", type=int, default=0, help="world mix only: curriculum level 0..3")
    args = parser.parse_args()
    manifest, splits = load_splits(args.data)
    if not 1 <= args.rollout_seeds <= 50:
        parser.error("--rollout-seeds must be 1..50")
    rollout_seeds = [args.seed + k for k in range(args.rollout_seeds)]
    if args.rollout and set(rollout_seeds) & {row["scene_seed"] for values in splits.values() for row in values}:
        parser.error("rollout seed must be outside all training/development/test scenes")
    if args.phrase_variant and args.split != "valid":
        parser.error("--phrase-variant applies to the valid split only")
    if any(i < 0 or i >= len(tasks(args.split)) for i in args.task_ids):
        parser.error("invalid task ID")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    planner = AgentPlanner(args.adapter)
    report = {"mode": "independent_actor_rollout" if args.rollout else "offline_action_exact_match",
              "adapter": planner.adapter, "split": args.split, "phrase_variant": args.phrase_variant, "recovery": os.environ.get("MICRODUCK_RECOVERY", "1") != "0", "legality": os.environ.get("MICRODUCK_LEGALITY", "0") != "0", "map_escape": os.environ.get("MICRODUCK_MAP_ESCAPE", "0") != "0", "dataset_manifest_sha256": digest(args.data / "manifest.json"),
              "results": [], "correct": 0, "total": 0,
              "note": "Development pilot. Rollout teacher is scorer AFTER execution, never a semantic intervention."}
    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    process = log = None
    try:
        if args.rollout:
            report.update(scene_seed=args.seed, scene_seeds=rollout_seeds, scene_mix=args.scene_mix,
                          world_level=args.world_level if args.scene_mix == "world" else None,
                          startup_failures=[], scenes=[],
                          layers={name: 0 for name in ("format_ok", "action_type_ok", "target_ok", "physical_ok", "task_ok")})
            for k, seed in enumerate(rollout_seeds):
                # One seed keeps the historical flat layout; several get one directory each.
                scene_dir = output if args.rollout_seeds == 1 else output / f"seed-{seed}"
                scene_dir.mkdir(exist_ok=True)
                occlusion, clutter = scene_profile(args.split, k, args.scene_mix)
                process, log, info, failures = start_scene(scene_dir, seed, occlusion, args.port,
                                                           collect_gt=True, clutter=clutter, yellow_course=args.scene_mix == "yellow",
                                                           **world_arguments(args.scene_mix, args.world_level))
                report["startup_failures"].extend(failures)
                report["scenes"].append({"seed": seed, "occlusion": occlusion, "clutter": clutter})
                try:
                    jobs = [(i, *tasks(args.split, args.phrase_variant)[i], None) for i in args.task_ids]
                    if args.explore_tasks:
                        jobs += explore_entries(info, args.split, seed, args.explore_tasks)
                    for i, instruction, specs, meta in jobs:
                        directory = scene_dir / f"task-{i:02d}"
                        # Failing to get back to the start between tasks is a harness/locomotion fault, not a wrong
                        # answer: retry it, and report any that persist separately from model errors.
                        for retry in range(3):
                            if directory.exists():
                                shutil.rmtree(directory)  # a retry starts from a clean episode directory
                            _, entry = episode(output, directory, info, args.split, seed, instruction, specs, planner, False, meta)
                            if "cannot restore task start" not in str(entry.get("error")):
                                break
                        infra = "cannot restore task start" in str(entry.get("error"))
                        invalid = AMBIGUOUS_TEACHER_TARGET in str(entry.get("error"))  # harness could not label it: not a model failure
                        result = {"seed": seed, "task_id": i, "instruction": instruction, "correct": entry["verified"], "error": entry.get("error"),
                                  "layers": entry.get("layers"), "episode": str(directory / "episode.json"),
                                  "infra_retries": retry, "infra_failure": infra, "invalid_task": invalid}
                        report["infra_failures"] = report.get("infra_failures", 0) + infra
                        report["results"].append(result)
                        report["invalid_tasks"] = report.get("invalid_tasks", 0) + invalid
                        if not invalid:  # invalid tasks stay in `results` but never count towards the score
                            report["correct"] += result["correct"]
                            report["total"] += 1
                            for name, ok in (result["layers"] or {}).items():
                                report["layers"][name] += bool(ok)
                        low, high = wilson_interval(report["correct"], report["total"])
                        report["wilson95"] = [round(low, 4), round(high, 4)]
                        save()
                        print(json.dumps(result, ensure_ascii=False), flush=True)
                finally:
                    stop_process(process)
                    log.close()
                    process = log = None
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
