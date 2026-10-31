"""Level-by-level curriculum over multi-room worlds. Development tool; never deploys weights.

For each world level the current adapter is measured on held-out layouts. A level that
already meets --promote-rate is passed without training. Otherwise one expert-corrected
agent_cycle runs at that level, and the result is accepted only if the held-out rate at
the level does not drop and no earlier level forgets more than --forgetting-tolerance.
The shared MuJoCo/skill/safety stack is unchanged; only the world grows. Held-out seeds
sit outside every training seed range, and the teacher only scores after execution.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from . import ROOT
from .agent_data import load_splits
from .agent_eval import wilson_interval
from .vision_world import WORLD_LEVELS

HELD_OUT_BASE = 95000  # training scenes use 61000.., valid 71000.., test 81000..; rollout default is 91010


def held_out_seed(level):
    return HELD_OUT_BASE + 100 * level


def free_seed_offset(splits, level, scenes):
    """First correction seed offset (agent_data: 61000 + offset) clear of every scene seed in the parent data."""
    used = {row["scene_seed"] for rows in splits.values() for row in rows}
    offset = 100 + 200 * level
    while any(61000 + offset + i in used for i in range(scenes)):
        offset += 100
    if offset > 5000:
        raise ValueError("no free correction seed range")
    return offset


def decide(before, after, earlier_before, earlier_after, tolerance):
    """Accept a trained adapter only if it helps (or holds) its level and forgets nothing badly.

    `before`/`after` are (correct, total) at the trained level; `earlier_*` map level -> (correct, total).
    Returns (accepted, reasons).
    """
    reasons = []
    rate = lambda pair: pair[0] / pair[1] if pair[1] else 0.0
    if rate(after) < rate(before):
        reasons.append(f"held-out rate fell {rate(before):.2f} -> {rate(after):.2f}")
    for level, old in earlier_before.items():
        new = earlier_after.get(level)
        if new is None or not new[1]:
            reasons.append(f"level {level} was not re-measured")
        elif rate(old) - rate(new) > tolerance:
            reasons.append(f"level {level} forgot {rate(old):.2f} -> {rate(new):.2f}")
    return not reasons, reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-data", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--name", default="curriculum-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-seeds", type=int, default=3, help="held-out layouts per level measurement")
    parser.add_argument("--forget-seeds", type=int, default=2, help="held-out layouts per earlier-level re-check")
    parser.add_argument("--promote-rate", type=float, default=0.75)
    parser.add_argument("--forgetting-tolerance", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--correction-scenes", type=int, default=2)
    parser.add_argument("--port", type=int, default=7880)
    args = parser.parse_args()
    if args.levels != sorted(set(args.levels)) or not all(0 <= v < len(WORLD_LEVELS) for v in args.levels):
        parser.error("levels must be ascending, unique and valid world levels")
    if not 1 <= args.forget_seeds <= args.eval_seeds <= 20 or not 0 <= args.promote_rate <= 1:
        parser.error("invalid seeds or promote rate")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be one path component")
    load_splits(args.initial_data)
    output = ROOT / "runs/agent-curriculum" / args.name
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "levels": [], "commands": [], "initial_adapter": str(args.initial_adapter),
              "final_adapter": None, "promote_rate": args.promote_rate,
              "note": "Development-set curriculum. Held-out world layouts, teacher scores only after execution. No automatic deployment."}

    def save():
        (output / "curriculum.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def run(module, *flags):
        command = [sys.executable, "-X", "utf8", "-u", "-m", "microduck_qwen." + module, *map(str, flags)]
        entry = {"command": command, "log": str(output / f"stage-{len(report['commands']):02d}.log")}
        report["commands"].append(entry)
        save()
        print("RUN: " + " ".join(command), flush=True)
        with Path(entry["log"]).open("w") as log:
            entry["returncode"] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).wait()
        save()
        if entry["returncode"]:
            raise RuntimeError(f"stage failed: {entry['log']}")

    def measure(adapter, data, level, seeds, tag):
        target = output / f"eval-L{level}-{tag}"
        run("agent_eval", "--data", data, "--adapter", adapter, "--output", target, "--rollout",
            "--scene-mix", "world", "--world-level", level, "--seed", held_out_seed(level),
            "--rollout-seeds", seeds, "--port", args.port)
        result = json.loads((target / "report.json").read_text())
        low, high = wilson_interval(result["correct"], result["total"])
        return {"correct": result["correct"], "total": result["total"], "wilson95": [round(low, 4), round(high, 4)],
                "layers": result.get("layers"), "report": str(target / "report.json")}

    adapter, data = args.initial_adapter.resolve(), args.initial_data.resolve()
    measured = {}  # level -> last accepted (correct, total) at forget_seeds-comparable seeds
    try:
        for level in args.levels:
            entry = {"level": level, "world": WORLD_LEVELS[level], "adapter_in": str(adapter)}
            report["levels"].append(entry)
            before = measure(adapter, data, level, args.eval_seeds, "before")
            entry["before"] = before
            if before["total"] and before["correct"] / before["total"] >= args.promote_rate:
                entry["action"] = "promoted_without_training"
                measured[level] = (before["correct"], before["total"])
                save()
                continue
            cycle_name = f"{args.name}-L{level}"
            offset = free_seed_offset(load_splits(data)[1], level, args.correction_scenes)
            run("agent_cycle", "--initial-data", data, "--initial-adapter", adapter, "--name", cycle_name,
                "--rounds", args.rounds, "--steps", args.steps, "--no-reward-guided", "--collect-first",
                "--scene-mix", "world", "--world-level", level, "--correction-scenes", args.correction_scenes,
                "--seed-offset", offset)
            cycle = json.loads((ROOT / "runs/agent-cycle" / cycle_name / "cycle.json").read_text())
            entry["cycle"] = str(ROOT / "runs/agent-cycle" / cycle_name / "cycle.json")
            if not cycle.get("selected_adapter"):
                entry["action"] = "training_not_qualified"
                measured[level] = (before["correct"], before["total"])
                save()
                continue
            candidate = Path(cycle["selected_adapter"])
            candidate_data = Path(next(r["data"] for r in cycle["rounds"] if r["adapter"] == cycle["selected_adapter"]))
            after = measure(candidate, candidate_data, level, args.eval_seeds, "after")
            entry["after"] = after
            earlier_before = {k: v for k, v in measured.items() if k < level}
            earlier_after = {}
            for earlier in earlier_before:
                recheck = measure(candidate, candidate_data, earlier, args.forget_seeds, "recheck")
                entry.setdefault("forgetting", {})[str(earlier)] = recheck
                earlier_after[earlier] = (recheck["correct"], recheck["total"])
            # Earlier levels are compared on the same number of seeds they were last measured with.
            accepted, reasons = decide((before["correct"], before["total"]), (after["correct"], after["total"]),
                                       earlier_before, earlier_after, args.forgetting_tolerance)
            entry.update(action="accepted" if accepted else "rejected", reasons=reasons)
            if accepted:
                adapter, data = candidate.resolve(), candidate_data.resolve()
                measured[level] = (after["correct"], after["total"])
                for earlier, pair in earlier_after.items():
                    measured[earlier] = pair
            else:
                measured[level] = (before["correct"], before["total"])
            entry["adapter_out"] = str(adapter)
            save()
        report.update(status="completed", final_adapter=str(adapter))
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(f"CURRICULUM REPORT: {output / 'curriculum.json'}", flush=True)


if __name__ == "__main__":
    main()
