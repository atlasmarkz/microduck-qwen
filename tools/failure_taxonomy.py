"""Failure taxonomy of rollout evaluation runs: what kind of failure, and for explore tasks which skill ended it.

    python tools/failure_taxonomy.py runs/lang/eff-98000 runs/lang/eff-98100
"""

import collections
import json
from pathlib import Path
import re
import sys


def kind(result):
    error = str(result.get("error"))
    layers = result.get("layers") or {}
    if "invalid actor JSON" in error:
        return "model: invalid JSON"
    if layers and not layers.get("action_type_ok", True):
        return "model: wrong action type"
    if "no confidently visible" in error:
        return "perception: target not confidently visible"
    if "cannot restore" in error:
        return "infra: reset"
    if "净空" in error:
        return "execution: clearance guard stop"
    if "视觉保护停止" in error:
        return "execution: visual guard stop (target lost)"
    if "failed physical" in error:
        return "physical outcome wrong"
    return "other: " + error[:60]


def last_stage(result):
    steps = json.loads(Path(result["episode"]).read_text(encoding="utf-8"))["steps"]
    last = steps[-1]
    execution = last.get("execution", {})
    action = (last["prediction"].get("action") or {}).get("action", "?")
    reason = re.sub(r"[0-9.]+", "N", str(execution.get("error") or ""))[:70] or f"success={execution.get('success')}, outcome check failed"
    return action, execution.get("stopped") or reason


def main(directories):
    total = failed = 0
    kinds, stages, families, seeds = (collections.Counter() for _ in range(4))
    for directory in directories:
        for result in json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))["results"]:
            total += 1
            if result["correct"]:
                continue
            failed += 1
            kinds[kind(result)] += 1
            families["explore" if result["task_id"] >= 100 else f"task{result['task_id']}"] += 1
            seeds[result["seed"]] += 1
            if result["task_id"] >= 100:
                stages[last_stage(result)] += 1
    print(f"{total - failed}/{total} correct; failures by task family {dict(families)}")
    print("by category:")
    for key, count in kinds.most_common():
        print(f"  {count:3d}  {key}")
    print("explore tasks, by last skill and reason:")
    for (action, reason), count in stages.most_common():
        print(f"  {count:3d}  {action}: {reason}")
    print("layouts with the most failures:", dict(seeds.most_common(5)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
