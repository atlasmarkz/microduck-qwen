"""Regression canary: per-state action accuracy of an adapter on the held-out split, no simulator needed.

Why: a retrain that fixes one behaviour can silently break another (the corrected-data retrain turned
"move forward" -> `done` into `unsupported`; the next one added invalid JSON), and a full rollout evaluation
needs ~2 hours to show it. The model's own decision is a pure function of (image, instruction, memory), so each
held-out row can be re-decided offline in about a minute and compared with the previous adapter, state by state.

State classes are "last completed skill -> expected skill", e.g. `move->done` or `start->explore`, so a drop in
one situation is visible even when the overall number barely moves.

Two sources of rows:
  * the held-out split of a dataset (21 rows; early single-room scenes, so it is blind to regressions that only show
    in multi-room world images: `move->done` has 2 rows there and every adapter passed it);
  * `--replay RUN_DIR ...`: every step an earlier rollout evaluation actually visited (the numbered image the model saw,
    its instruction and memory, the teacher's action), re-decided by the adapter under test. This is the same
    "evaluate on the states the policy visits" idea as DAgger-style aggregation, used here as a regression test; it
    covers the world distribution, and ~200 decisions take a few minutes instead of a two-hour rollout.
    A replay row counts as correct when the action *type* matches the teacher's (selectors and ids vary legitimately).

Related work this follows (abstract-level reading):
  * Plain LoRA fine-tuning with no anti-forgetting mechanism is the lower bound in continual-learning
    benchmarks, which report retention per task: CL-VISTA (arXiv:2604.00677). Per-class retention is the
    same measurement at the scale of this project.
  * Mixing a little earlier/general data back in is the usual remedy for narrow-fine-tuning regressions
    (replay, arXiv:2603.14493); this canary only detects, `--base-data` aggregation in `agent_data` already replays.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .agent import AgentPlanner
from .agent_data import load_splits
from .sim_data import digest


def state_class(row):
    completed = (row.get("memory") or {}).get("completed") or []
    last = completed[-1]["action"] if completed else "start"
    return f"{last}->{row['expected']['action']}"


def replay_rows(run_directories):
    """Rows for every step recorded by rollout evaluations: the image the model saw, its instruction, memory and the teacher action."""
    rows = []
    for run in map(Path, run_directories):
        for episode_path in sorted(run.glob("seed-*/task-*/episode.json")):
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            for k, step in enumerate(episode.get("steps", [])):
                image = episode_path.parent / f"step-{k:03d}-candidates.png"
                expected = step.get("expected")
                if not image.exists() or not isinstance(expected, dict) or not step.get("memory"):
                    continue
                rows.append({"id": f"{run.name}/{episode_path.parent.relative_to(run).as_posix()}#{k}",
                             "instruction": step.get("prediction", {}).get("instruction") or episode["instruction"],
                             "memory": step["memory"], "expected": expected, "image": str(image)})
    return rows


def decision_correct(prediction, expected, replay):
    action = prediction.get("action")
    if action is None:
        return False
    return action["action"] == expected["action"] if replay else action == expected


def summarize(results):
    """{class: [correct, total]} from rows of {"state": str, "correct": bool}."""
    classes = defaultdict(lambda: [0, 0])
    for result in results:
        classes[result["state"]][0] += bool(result["correct"])
        classes[result["state"]][1] += 1
    return {key: value for key, value in sorted(classes.items())}


def regressions(new, base, min_drop=1):
    """Classes whose correct count fell by at least `min_drop` against the baseline (same totals required)."""
    flagged = []
    for key, (correct, total) in new.items():
        if key in base and base[key][1] == total and base[key][0] - correct >= min_drop:
            flagged.append({"state": key, "before": base[key][0], "after": correct, "total": total})
    return flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="dataset whose held-out split is re-decided (not needed with --replay)")
    parser.add_argument("--replay", type=Path, nargs="+", help="rollout evaluation directories whose recorded steps are re-decided")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--baseline", type=Path, help="canary report of the adapter being replaced")
    parser.add_argument("--min-drop", type=int, default=1, help="flag a state class that loses at least this many rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.data) == bool(args.replay):
        parser.error("give exactly one of --data and --replay")
    if args.replay:
        rows = replay_rows(args.replay)
        source = {"replay": [str(d) for d in args.replay]}
    else:
        _, splits = load_splits(args.data)
        rows = [dict(row, image=str(args.data / row["image"])) for row in splits[args.split]]
        source = {"split": args.split, "dataset_manifest_sha256": digest(args.data / "manifest.json")}
    planner = AgentPlanner(args.adapter)
    results = []
    for row in rows:
        prediction = planner.predict(row["instruction"], row["image"], row["memory"])
        results.append({"id": row["id"], "state": state_class(row), "expected": row["expected"], "raw": prediction["raw"],
                        "correct": decision_correct(prediction, row["expected"], bool(args.replay))})
    classes = summarize(results)
    report = {"adapter": planner.adapter, **source, "correct": sum(r["correct"] for r in results), "total": len(results),
              "classes": classes, "results": results}
    if args.baseline:
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        report["baseline"] = str(args.baseline)
        report["regressions"] = regressions(classes, base["classes"], args.min_drop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CANARY {args.adapter}: {report['correct']}/{report['total']}")
    for key, (correct, total) in classes.items():
        print(f"  {key:28s} {correct}/{total}")
    for item in report.get("regressions", []):
        print(f"  REGRESSION {item['state']}: {item['before']} -> {item['after']} of {item['total']}")
    raise SystemExit(1 if report.get("regressions") else 0)


if __name__ == "__main__":
    main()
