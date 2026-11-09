"""Failure taxonomy of rollout evaluation runs: what kind of failure, and for explore tasks which skill ended it.

    python tools/failure_taxonomy.py runs/lang/eff-98000 runs/lang/eff-98100
"""

import collections
import json
from pathlib import Path
import re
import sys

from microduck_qwen.compare_runs import is_invalid


def done_for_stop(result):
    """True when the only miss is `done` where the teacher expected `stop`, after every motion skill succeeded.

    Both end the episode with the robot already standing still, so the physical outcome is the same; the strict
    action-type score still counts it as a miss.
    """
    try:
        steps = json.loads(Path(result["episode"]).read_text(encoding="utf-8"))["steps"]
    except (OSError, KeyError, ValueError):
        return False
    if not steps:
        return False
    last = steps[-1]
    predicted = (last.get("prediction") or {}).get("action") or {}
    return ((last.get("expected") or {}).get("action") == "stop" and predicted.get("action") == "done"
            and all(s.get("physical_ok", False) for s in steps[:-1]) and all(s.get("target_ok", True) for s in steps))


def kind(result):
    error = str(result.get("error"))
    layers = result.get("layers") or {}
    if done_for_stop(result):
        return "terminal: `done` where the teacher expects `stop` (motion ok)"
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
    if not steps:
        return "?", "episode recorded no steps"
    last = steps[-1]
    execution = last.get("execution", {})
    action = (last["prediction"].get("action") or {}).get("action", "?")
    reason = re.sub(r"[0-9.]+", "N", str(execution.get("error") or ""))[:70] or f"success={execution.get('success')}, outcome check failed"
    return action, execution.get("stopped") or reason


def main(directories):
    total = failed = invalid = 0
    kinds, stages, families, seeds = (collections.Counter() for _ in range(4))
    for directory in directories:
        for result in json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))["results"]:
            if is_invalid(result):  # the harness could not label this task: not a model outcome
                invalid += 1
                continue
            total += 1
            if result["correct"]:
                continue
            failed += 1
            kinds[kind(result)] += 1
            families["explore" if result["task_id"] >= 100 else f"task{result['task_id']}"] += 1
            seeds[result["seed"]] += 1
            if result["task_id"] >= 100:
                stages[last_stage(result)] += 1
    print(f"{total - failed}/{total} correct ({invalid} invalid tasks excluded); failures by task family {dict(families)}")
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
