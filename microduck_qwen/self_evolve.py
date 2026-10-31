"""Self-evolution loop: free exploration -> self-posed tasks -> own attempts -> environment check -> retrain.

No teacher and no scene labels are involved in producing a training row:
  1. The robot explores a world for --explore-minutes on its own and remembers every ball it
     perceived (colour and depth-estimated position).
  2. It poses itself tasks about what it found (find a colour that was not in the first view,
     approach a visible one), in natural language.
  3. The current actor attempts each task with its own actions (one greedy attempt, then one
     sampled attempt if that failed).
  4. A verifier built from the robot's own perception, its odometry and a contact bumper decides
     success. Only verified attempts become training rows, labelled with the actor's own actions.
  5. The actor is retrained on base data + those rows and kept only if it does not get worse on
     held-out worlds, scored by the same independent rollout used everywhere else.

Pose and camera extrinsics remain simulator oracle; the contact bumper reads the simulator.
It is rejection-sampling self-training, not online RL. Nothing is deployed automatically.
"""

import argparse
from datetime import datetime
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

from . import ROOT
from .agent_curriculum import decide
from .agent_data import load_splits, write_splits
from .agent_eval import wilson_interval
from .agent_language import colour_tasks, explore_tasks
from .explore import free_explore
from .sim_data import digest, request
from .sim_runner import start_scene, stop_process
from .skills import SkillExecutor, memory_state
from .vision import capture, propose_targets, save_frame
from .vision_world import WORLD_LEVELS

SELF_SEED_BASE = 97000      # disjoint from train 61000.., valid 71000.., test 81000.., held-out 95000..
# approach stops 0.30 m from the ball; arrival tolerance and gait coast add a little (see agent_data).
APPROACH_OK_M = 0.30 + 0.025 + 0.05
HOME_OK_M = 0.125
HOME_YAW_OK_DEG = 12.0
LABEL = "self-play: actor's own action, verified by perception + odometry + contact bumper; no teacher"


def self_tasks(inventory, visible_colors, rng, per_color):
    """Natural-language tasks about what the robot has actually perceived.

    Colours it saw but that were NOT in the first view become find-and-approach tasks; colours
    that were in view become plain approach tasks.
    """
    found = {i["color"] for i in inventory}
    hidden = found - set(visible_colors)
    tasks = explore_tasks("train", hidden, rng, per_color)
    tasks += colour_tasks("train", sorted(found & set(visible_colors)), rng, per_color)
    return [(text, meta) for text, _, meta in tasks]


def self_verify(meta, steps, contact_delta, inventory):
    """(ok, reasons) from the robot's own evidence. `steps`: executed {action, result} dicts in order."""
    reasons = []
    if contact_delta:
        reasons.append("scene contact")
    actions = [s["action"]["action"] for s in steps]
    tail = meta["tail"]
    if not steps or actions[-1] != ("stop" if tail == "stop" else "done"):
        reasons.append("did not end with " + ("stop" if tail == "stop" else "done"))
    if any(not s["result"].get("success") for s in steps if s["action"]["action"] not in ("done", "stop")):
        reasons.append("a skill failed")
    approaches = [s for s in steps if s["action"]["action"] == "approach"]
    if actions.count("approach") != 1:
        reasons.append("expected exactly one approach")
    else:
        bound = approaches[0]["result"].get("bound_target")
        if not bound or bound["color"] != meta["color"]:
            reasons.append("approached the wrong colour")
        elif approaches[0].get("final_distance_m") is None or approaches[0]["final_distance_m"] > APPROACH_OK_M:
            reasons.append("did not end near the bound target")
        elif not any(i["color"] == meta["color"] and math.hypot(i["x"] - bound["world_position"][0],
                                                                i["y"] - bound["world_position"][1]) < 0.5
                     for i in inventory):
            reasons.append("target does not match anything it perceived while exploring")
    if meta["kind"] == "explore_find":
        if "explore" not in actions or ("approach" in actions and actions.index("explore") > actions.index("approach")):
            reasons.append("did not explore before approaching")
    if tail == "home":
        homes = [s for s in steps if s["action"]["action"] == "return_home"]
        if not homes or homes[-1]["result"].get("distance_home_m", 9) > HOME_OK_M or \
                homes[-1]["result"].get("home_yaw_error_deg", 999) >= HOME_YAW_OK_DEG:
            reasons.append("not back home")
    if tail == "none" and "return_home" in actions:
        reasons.append("unrequested return_home")
    return not reasons, reasons


def attempt(planner, executor, info, directory, output, seed, instruction, meta, max_steps, temperature, inventory):
    """One actor-only attempt. Returns (rows, report); rows are produced only if self-verified."""
    directory.mkdir(parents=True)
    report = {"instruction": instruction, "temperature": temperature, "steps": [], "verified": False}
    steps, completed, rows = [], [], []
    port = info["port"]
    try:
        reset = executor.run({"action": "return_home"})
        if not reset["success"]:
            raise RuntimeError("cannot restore start")
        start = executor.healthy()
        executor.task_start_pose = {"x": start["trunk"][0], "y": start["trunk"][1], "yaw": start["yaw"]}
        contact0 = request(port, "collection_stats")["steps_with_scene_contact"]
        for index in range(max_steps):
            executor.robot.stop()
            frame = capture(port)
            candidates = propose_targets(frame)
            image = save_frame(directory, frame, candidates, f"step-{index:03d}")
            memory = memory_state(info, executor.healthy(), completed)
            prediction = planner.predict(instruction, image, memory, temperature=temperature)
            action = prediction.get("action")
            entry = {"memory": memory, "raw": prediction.get("raw"), "action": action}
            report["steps"].append(entry)
            if action is None:
                report["error"] = "invalid actor JSON"
                break
            result = executor.run(action, candidates)
            entry["result_success"] = result["success"]
            step = {"action": action, "result": result}
            if action["action"] == "approach" and result.get("bound_target"):
                position = executor.healthy()["trunk"][:2]
                target = result["bound_target"]["world_position"]
                step["final_distance_m"] = math.dist(position, target[:2])
            steps.append(step)
            rows.append({"image": str(image.relative_to(output)), "image_sha256": digest(image),
                         "instruction": instruction, "memory": memory, "expected": action})
            if action["action"] in ("done", "stop", "clarify", "unsupported") or not result["success"]:
                break
            completed.append(action)
        delta = request(port, "collection_stats")["steps_with_scene_contact"] - contact0
        ok, reasons = self_verify(meta, steps, delta, inventory)
        report.update(verified=ok, reasons=reasons, contact_steps=delta)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        rows = []
    finally:
        (directory / "attempt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return (rows if report["verified"] else []), report


def collect_world(planner, output, seed, port, level, minutes, per_color, max_steps, attempts=3):
    directory = output / "worlds" / f"world-{seed}"
    directory.mkdir(parents=True)
    process, log, info, _ = start_scene(directory, seed, "none", port, collect_gt=True,
                                        world=WORLD_LEVELS[level]["rooms"], density=WORLD_LEVELS[level]["density"])
    summary = {"seed": seed, "level": level, "tasks": [], "rows": 0}
    rows = []
    executor = None
    try:
        executor = SkillExecutor(info, directory / "run")
        executor.run({"action": "return_home"})
        first = capture(info["port"])
        visible = {c["color"] for c in propose_targets(first) if c.get("kind", "ball") == "ball"}
        exploration = free_explore(executor, minutes, directory / "free-explore")
        summary["exploration"] = {k: exploration[k] for k in ("minutes", "elapsed_s", "explored_cells", "stopped") if k in exploration}
        summary["exploration"]["balls_perceived"] = len(exploration["inventory"])
        summary["exploration"]["frontier_steps"] = len(exploration["iterations"])
        inventory = exploration["inventory"]
        rng = random.Random(seed)
        for k, (text, meta) in enumerate(self_tasks(inventory, visible, rng, per_color)):
            outcome = {"instruction": text, "meta": meta, "attempts": []}
            for n, temperature in enumerate([0.0] + [0.7] * (attempts - 1)):
                new_rows, report = attempt(planner, executor, info, directory / f"task-{k:02d}-try{n}", output, seed,
                                           text, meta, max_steps, temperature, inventory)
                outcome["attempts"].append({"temperature": temperature, "verified": report["verified"],
                                            "reasons": report.get("reasons"), "error": report.get("error")})
                if new_rows:
                    for j, row in enumerate(new_rows):
                        rows.append(dict(row, id=f"self-{seed}-{k}-{n}-{j}", scene_seed=seed, split="train",
                                         scene_occlusion="none", scene_clutter=False, scene_yellow_course=False,
                                         verified=True, task_type=row["expected"]["action"], task_meta=meta,
                                         label_source=LABEL))
                    break
            summary["tasks"].append(outcome)
        summary["rows"] = len(rows)
        (directory / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    finally:
        if executor:
            executor.close()
        stop_process(process)
        log.close()
    return rows, summary


def merge(base, output, rows, name, source):
    """Base dataset (valid/test untouched) plus the self-verified rows as extra train rows.

    `source` is the directory the rows' image paths are relative to; images are copied in.
    """
    base_manifest, splits = load_splits(base)
    shutil.copytree(base, output / "base")
    splits = {s: [dict(r, image="base/" + r["image"]) for r in values] for s, values in splits.items()}
    copied = []
    for row in rows:
        target = output / "self" / row["image"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / row["image"], target)
        copied.append(dict(row, image=str(Path("self") / row["image"])))
    splits["train"].extend(copied)
    manifest = {"format": "simulation_agent_v1", "status": "completed", "splits": {}, "episodes": [],
                "method": "self-play, environment-verified rejection sampling; valid/test fixed from parent",
                "parent_manifest_sha256": digest(base / "manifest.json"), "name": name}
    write_splits(output, manifest, splits)
    load_splits(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-data", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--name", default="self-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--worlds", type=int, default=2, help="worlds explored per round")
    parser.add_argument("--level", type=int, default=2, choices=range(1, len(WORLD_LEVELS)))
    parser.add_argument("--explore-minutes", type=float, default=5.0, help="free exploration per world")
    parser.add_argument("--tasks-per-color", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3, help="per task: one greedy attempt, then sampled ones until one verifies")
    parser.add_argument("--steps", type=int, default=150, help="training steps per round")
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=98000)
    parser.add_argument("--eval-task-ids", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--forgetting-tolerance", type=float, default=0.25)
    parser.add_argument("--port", type=int, default=7930)
    parser.add_argument("--resume-collection", type=Path,
                        help="round directory whose worlds/*/rows.jsonl are reused instead of collecting again (round 0 only)")
    args = parser.parse_args()
    if not 0.2 <= args.explore_minutes <= 60 or not 1 <= args.rounds <= 5 or not 1 <= args.worlds <= 10:
        parser.error("explore-minutes 0.2..60, rounds 1..5, worlds 1..10")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be one path component")
    load_splits(args.initial_data)
    output = ROOT / "runs/self-evolve" / args.name
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "rounds": [], "commands": [], "initial_adapter": str(args.initial_adapter),
              "final_adapter": None, "note": __doc__.strip().splitlines()[0]}

    def save():
        (output / "self_evolve.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

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

    def measure(adapter, data, tag):
        target = output / f"eval-{tag}"
        run("agent_eval", "--data", data, "--adapter", adapter, "--output", target, "--rollout",
            "--scene-mix", "world", "--world-level", args.level, "--seed", args.eval_seed,
            "--rollout-seeds", args.eval_seeds, "--task-ids", *args.eval_task_ids, "--explore-tasks", 1,
            "--port", args.port + 1)
        result = json.loads((target / "report.json").read_text())
        low, high = wilson_interval(result["correct"], result["total"])
        return {"correct": result["correct"], "total": result["total"], "wilson95": [round(low, 4), round(high, 4)],
                "layers": result.get("layers"), "report": str(target / "report.json")}

    adapter, data = args.initial_adapter.resolve(), args.initial_data.resolve()
    try:
        from .vision_torch import TorchAgentPlanner

        current = measure(adapter, data, "r0-start") if not args.resume_collection else json.loads(
            (args.resume_collection.parent.parent / args.resume_collection.parent.name / "self_evolve.json").read_text())["start"]
        report["start"] = current
        for index in range(args.rounds):
            entry = {"round": index, "adapter_in": str(adapter), "worlds": []}
            report["rounds"].append(entry)
            rows, round_dir = [], output / f"round-{index}"
            round_dir.mkdir()
            planner = None
            if args.resume_collection and index == 0:
                round_dir = args.resume_collection.resolve()
                for path in sorted(round_dir.glob("worlds/*/rows.jsonl")):
                    rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
                entry["resumed_from"] = str(round_dir)
            else:
                planner = TorchAgentPlanner(adapter)
            try:
                for w in range(0 if planner is None else args.worlds):
                    seed = SELF_SEED_BASE + index * 100 + w
                    new_rows, summary = collect_world(planner, round_dir, seed, args.port, args.level,
                                                      args.explore_minutes, args.tasks_per_color, args.max_steps, args.attempts)
                    rows.extend(new_rows)
                    entry["worlds"].append(summary)
                    save()
            finally:
                planner = None
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
            entry["self_rows"] = len(rows)
            tasks = [t for w in entry["worlds"] for t in w["tasks"]]
            if tasks:
                entry["tasks"] = {"posed": len(tasks), "verified": sum(any(a["verified"] for a in t["attempts"]) for t in tasks)}
            if not rows:
                entry["action"] = "no_verified_rows"
                save()
                continue
            merged = ROOT / "data" / f"{args.name}-r{index}"
            merge(data, merged, rows, f"{args.name}-r{index}", round_dir)
            candidate = ROOT / "adapters" / f"{args.name}-r{index}"
            run("vision_torch", "train", "--task", "agent", "--data", merged, "--output", candidate,
                "--steps", args.steps, "--resume", adapter)
            after = measure(candidate, merged, f"r{index}-after")
            accepted, reasons = decide((current["correct"], current["total"]), (after["correct"], after["total"]),
                                       {}, {}, args.forgetting_tolerance)
            entry.update(after=after, action="accepted" if accepted else "rejected", reasons=reasons,
                         adapter_out=str(candidate if accepted else adapter))
            if accepted:
                adapter, data, current = candidate.resolve(), merged.resolve(), after
            save()
        report.update(status="completed", final_adapter=str(adapter))
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(f"SELF-EVOLVE REPORT: {output / 'self_evolve.json'}", flush=True)


if __name__ == "__main__":
    main()
