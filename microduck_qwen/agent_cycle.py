"""Bounded actor/reward SFT, reward-ranked expert correction and evaluation. No automatic deployment."""

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
    parser.add_argument("--initial-adapter", type=Path, help="continue an existing actor instead of starting from the base model")
    parser.add_argument("--name", default="agent-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--reward-guided", action=argparse.BooleanOptionalAction, default=os.name == "nt")
    parser.add_argument("--reward-steps", type=int, default=50)
    parser.add_argument("--reward-samples", type=int, default=3)
    parser.add_argument("--reward-threshold", type=float, default=0.5)
    parser.add_argument("--correction-scenes", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--scene-mix", choices=("plain", "hard", "yellow"), default="plain",
                        help="scene difficulty for correction collection and rollouts")
    parser.add_argument("--rollout", action="store_true", help="also test 4 independent physical tasks each round")
    args = parser.parse_args()
    if args.reward_guided and os.name != "nt":
        parser.error("reward-guided iteration currently requires the Windows CUDA backend")
    if not 1 <= args.reward_steps <= 300 or not 1 <= args.reward_samples <= 5 or not 0 <= args.reward_threshold <= 1:
        parser.error("invalid bounded reward budget/threshold")
    if not 1 <= args.rounds <= 3 or not 1 <= args.steps <= 300 or not 1 <= args.correction_scenes <= 10 or not 100 <= args.seed_offset <= 5000:
        parser.error("bounded small-Mac iteration budget exceeded")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be one path component")
    manifest, splits = load_splits(args.initial_data)
    output = ROOT / "runs/agent-cycle" / args.name
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "commands": [], "rounds": [], "selected_adapter": None,
              "reward_guided": args.reward_guided, "reward_samples": args.reward_samples,
              "reward_threshold": args.reward_threshold,
              "initial_adapter": str(args.initial_adapter) if args.initial_adapter else None,
              "note": "Development-set selection, not final test. Reward ranks proposals; independent expert verifies labels. No online RL or automatic deployment."}
    def save():
        (output / "cycle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    def run(module, *flags):
        command = [sys.executable, "-X", "utf8", "-u", "-m", "microduck_qwen." + module, *map(str, flags)]
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
    data, parent, reward_parent, best = args.initial_data.resolve(), args.initial_adapter, None, -1
    try:
        baseline = output / "base-valid"
        run("agent_eval", "--data", data, "--output", baseline)
        report["baseline"] = json.loads((baseline / "report.json").read_text())
        for index in range(args.rounds):
            if index:
                corrected = ROOT / "data" / f"{args.name}-r{index}-corrections"
                flags = ["--output", corrected, "--base-data", data, "--actor-adapter", parent,
                    "--train-scenes", args.correction_scenes, "--seed-offset", args.seed_offset + index * 100,
                    "--scene-mix", args.scene_mix]
                if args.reward_guided:
                    flags.extend(["--reward-adapter", reward_parent, "--reward-samples", args.reward_samples,
                                  "--reward-threshold", args.reward_threshold])
                run("agent_data", *flags)
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
            if args.reward_guided:
                reward_adapter = ROOT / "adapters" / f"{args.name}-reward-r{index}"
                flags = ["train", "--task", "reward", "--data", data, "--output", reward_adapter,
                         "--steps", args.reward_steps]
                if reward_parent:
                    flags.extend(["--resume", reward_parent])
                run("vision_torch", *flags)
                reward_evaluation = output / f"reward-r{index}-valid.json"
                run("vision_torch", "evaluate-reward", "--data", data, "--adapter", reward_adapter,
                    "--output", reward_evaluation, "--split", "valid")
                entry.update(reward_adapter=str(reward_adapter),
                             reward_valid=json.loads(reward_evaluation.read_text())["summary"])
                reward_parent = reward_adapter
            if args.rollout:
                rollout = output / f"r{index}-rollout"
                run("agent_eval", "--data", data, "--adapter", adapter, "--output", rollout, "--rollout",
                    "--scene-mix", args.scene_mix, *(["--task-ids", 9] if args.scene_mix == "yellow" else []))
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
