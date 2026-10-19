"""Bounded local collect/SFT/correct/SFT/evaluate orchestration. No automatic deployment."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from . import ROOT
from .agent_data import load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-data", type=Path, required=True)
    parser.add_argument("--name", default="agent-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--correction-scenes", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--scene-mix", choices=("plain", "hard"), default="plain",
                        help="scene difficulty for correction collection and rollouts")
    parser.add_argument("--rollout", action="store_true", help="also test 4 independent physical tasks each round")
    args = parser.parse_args()
    if not 1 <= args.rounds <= 3 or not 1 <= args.steps <= 300 or not 1 <= args.correction_scenes <= 10 or not 100 <= args.seed_offset <= 5000:
        parser.error("bounded small-Mac iteration budget exceeded")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be one path component")
    manifest, splits = load_splits(args.initial_data)
    output = ROOT / "runs/agent-cycle" / args.name
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "commands": [], "rounds": [], "selected_adapter": None,
              "note": "Development-set selection, not final test. Learned reward and online RL not implemented."}
    def save():
        (output / "cycle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    def run(module, *flags):
        command = [sys.executable, "-u", "-m", "microduck_qwen." + module, *map(str, flags)]
        entry = {"command": command, "log": str(output / f"stage-{len(report['commands']):02d}.log")}
        report["commands"].append(entry)
        save()
        print("RUN: " + " ".join(command), flush=True)
        with Path(entry["log"]).open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                entry["returncode"] = process.wait()
            except BaseException:
                process.send_signal(__import__("signal").SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=15)
                raise
        save()
        if entry["returncode"]:
            raise RuntimeError(f"stage failed: {entry['log']}")
    data, parent, best = args.initial_data.resolve(), None, -1
    try:
        baseline = output / "base-valid"
        run("agent_eval", "--data", data, "--output", baseline)
        report["baseline"] = json.loads((baseline / "report.json").read_text())
        for index in range(args.rounds):
            if index:
                corrected = ROOT / "data" / f"{args.name}-r{index}-corrections"
                run("agent_data", "--output", corrected, "--base-data", data, "--actor-adapter", parent,
                    "--train-scenes", args.correction_scenes, "--seed-offset", args.seed_offset + index * 100,
                    "--scene-mix", args.scene_mix)
                data = corrected
            adapter = ROOT / "adapters" / f"{args.name}-r{index}"
            if os.name == "nt":
                flags = ["train", "--task", "agent", "--data", data, "--output", adapter, "--steps", args.steps]
                if parent:
                    flags.extend(["--resume", parent])
                run("vision_torch", *flags)
            else:
                flags = ["train", "--data", data, "--output", adapter, "--steps", args.steps,
                         "--accumulation", 1, "--max-seq-length", 1024]
                if parent:
                    flags.extend(["--resume-adapter", parent])
                run("vision_sft", *flags)
            evaluation = output / f"r{index}-valid"
            run("agent_eval", "--data", data, "--adapter", adapter, "--output", evaluation)
            metrics = json.loads((evaluation / "report.json").read_text())
            entry = {"round": index, "data": str(data), "adapter": str(adapter),
                     "valid_correct": metrics["correct"], "valid_total": metrics["total"]}
            if args.rollout:
                rollout = output / f"r{index}-rollout"
                run("agent_eval", "--data", data, "--adapter", adapter, "--output", rollout, "--rollout",
                    "--scene-mix", args.scene_mix)
                physical = json.loads((rollout / "report.json").read_text())
                entry.update(rollout_correct=physical["correct"], rollout_total=physical["total"])
            score = entry["valid_correct"] / entry["valid_total"]
            qualifies = score >= report["baseline"]["correct"] / report["baseline"]["total"]
            if args.rollout:
                qualifies &= entry["rollout_correct"] == entry["rollout_total"]
            if qualifies and score > best:
                best, report["selected_adapter"] = score, str(adapter)
            report["rounds"].append(entry)
            save()
            parent = adapter  # Continue latest trained round, even if not promoted.
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(f"CYCLE REPORT: {output / 'cycle.json'}", flush=True)


if __name__ == "__main__":
    main()
