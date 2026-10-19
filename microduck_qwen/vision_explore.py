"""Bounded reward-ranked visual exploration; simulator truth scores only after selection."""

import argparse
import json
import math
from pathlib import Path

from .navigation import Observer, Robot, execute
from .vision import VisualGuard, VisualOutputError, capture, propose_balls, save_frame, visibility_check
from .vision_data import collection_frame, digest, request, verify_objects, visual_examples
from .sim_runner import start_scene, stop_process
from .vision_scene import make_scene
from .vision_torch import TorchVisionPlanner, checked_adapter, outcome_prompt, reward_prompt, reward_score


def teacher_choice(scene, frame, truth, candidates, instruction):
    verified = verify_objects(scene, frame, truth, candidates)
    matches = [(choice, kind) for text, choice, kind in visual_examples(verified) if text == instruction]
    return matches[0] if matches else (None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--reward", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--protocol", type=Path, default=Path("experiments/windows-vision-v2/protocol.json"))
    args = parser.parse_args()
    import torch

    torch.manual_seed(42)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if digest(args.data / "manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Dataset does not match frozen experiment protocol")
    if args.explore and (args.adapter is None or args.reward is None):
        parser.error("Exploration requires both actor and reward adapters")
    args.output.mkdir(parents=True, exist_ok=False)
    planner = TorchVisionPlanner(args.adapter)
    if args.reward:
        checked_adapter(args.reward, "reward")
        planner.model.load_adapter(str(args.reward), adapter_name="reward")
    cases = protocol["physical_cases"]
    if args.explore:
        cases = []
        for index in range(protocol["exploration"]["scenes"]):
            seed = protocol["exploration"]["train_seed_base"] + index
            kind = ("none", "partial", "full", "absent", "none", "none", "none", "partial")[index % 8]
            color = make_scene(seed, "none")["objects"]["ball_0"]["color"]
            cn = {"red": "红", "blue": "蓝", "green": "绿"}[color]
            instruction = (f"走到{cn}色球旁边" if kind in ("full", "absent") else
                ("走到画面右边的球旁边" if index == 4 else "走到画面左边的球旁边"))
            cases.append([seed, kind, instruction])
    report = {"status": "running", "exploration": args.explore, "cases": [],
        "actor_sha256": digest(args.adapter / "adapter_model.safetensors") if args.adapter else None,
        "reward_sha256": digest(args.reward / "adapter_model.safetensors") if args.reward else None,
        "note": "Three sampled proposals, one retained action; single-action adaptation, not full UI-Genie beam search."}
    actor_rows, reward_rows = [], []

    def save():
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name, rows in (("actor-replay", actor_rows), ("reward-replay", reward_rows)):
            (args.output / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    try:
        for index, (seed, kind, instruction) in enumerate(cases):
            episode = args.data / "replay" / args.output.name / f"case-{index:02d}"
            episode.mkdir(parents=True, exist_ok=False)
            entry = {"seed": seed, "scenario": kind, "instruction": instruction, "proposals": [], "verified_success": False}
            process = log = robot = observer = None
            try:
                process, log, info, failures = start_scene(episode, seed, kind, 7844, collect_gt=True)
                entry["startup_failures"] = failures
                frame, truth = collection_frame(info["port"])
                candidates = propose_balls(frame)
                image = save_frame(episode, frame, candidates)
                samples = protocol["exploration"]["sampled_candidates"] if args.explore else 1
                unique = set()
                for sample in range(samples):
                    if args.reward:
                        planner.model.set_adapter("default")
                    try:
                        prediction = planner.predict(instruction, image, candidates, temperature=0.7 if sample else 0.0)
                        action = prediction["visual_choice"]
                        key = json.dumps(action, sort_keys=True)
                        if key in unique:
                            continue
                        unique.add(key)
                        item = {"prediction": prediction, "action": action}
                        if args.reward:
                            planner.model.set_adapter("reward")
                            item["reward"] = reward_score(planner.model, planner.processor, image, reward_prompt(instruction, action))
                        entry["proposals"].append(item)
                    except VisualOutputError as error:
                        entry["proposals"].append({"error": str(error), "raw": error.prediction["raw"]})
                valid = [p for p in entry["proposals"] if "action" in p]
                selected = max(valid, key=lambda p: p.get("reward", 0.0)) if valid else None
                entry["selected"] = selected
                # No scene names, segmentation or true coordinates were passed to actor/RM above.
                scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
                expected, task_type = teacher_choice(scene, frame, truth, candidates, instruction)
                entry["expected"] = expected
                if expected is None:
                    entry["discarded"] = "Independent image/segmentation association was ambiguous"
                    continue
                entry["grounding_correct"] = bool(selected and selected["action"] == expected)
                row = {"id": f"replay-{seed}", "scene_seed": seed, "split": "train" if args.explore else "test",
                    "scenario": kind, "frame": 0, "image": str(image.relative_to(args.data)), "image_sha256": digest(image),
                    "instruction": instruction, "expected": expected, "task_type": task_type}
                if args.explore:
                    for number, proposal in enumerate(valid):
                        correct = proposal["action"] == expected
                        reward_rows.append(dict(row, id=f"{row['id']}-proposal-{number}",
                            prompt=reward_prompt(instruction, proposal["action"]), completion="yes" if correct else "no",
                            reward_label=int(correct), proposed_action=proposal["action"], reward_scope="action",
                            hard_negative=bool(not correct and proposal.get("reward", 0) >= 0.5)))
                robot, observer = Robot(info["socket"]), Observer(info["port"])
                robot.stop()
                if selected and selected["action"]["status"] == "visible":
                    candidate = next(c for c in candidates if c["id"] == selected["action"]["id"])
                    check = visibility_check(capture(info["port"]), candidate["world_position"], candidate["color"])
                    entry["preflight_visibility"] = check
                    if candidate["visibility"] == "visible" and check["safe"]:
                        x, y, _ = candidate["world_position"]
                        plan = {"status": "ok", "goals": [{"target": "point", "x": x, "y": y,
                            "speed": "slow", "stop_distance_m": 0.3}]}
                        entry["execution"] = execute(plan, robot, observer, episode / "trajectory.jsonl", timeout_s=35,
                            scene=scene, guard=VisualGuard(info["port"], candidate["world_position"], candidate["color"], episode))
                        state = observer.read()
                        stats = request(info["port"], "collection_stats")
                        entry["contacts"] = stats
                        if expected["status"] == "visible":
                            target_candidate = next(c for c in candidates if c["id"] == expected["id"])
                            target = min((obj for obj in scene["objects"].values() if obj["color"] == expected["color"]),
                                key=lambda obj: math.dist([obj["x"], obj["y"]], target_candidate["world_position"][:2]))
                            entry["true_target_distance_m"] = math.dist(state["trunk"][:2], [target["x"], target["y"]])
                            entry["verified_success"] = bool(entry["grounding_correct"] and entry["execution"]["success"]
                                and entry["true_target_distance_m"] <= 0.35 and stats["steps_with_scene_contact"] == 0)
                else:
                    entry["verified_success"] = entry["grounding_correct"]
                if args.explore:
                    acceptance_reward = selected.get("reward", 0.0) if selected else 0.0
                    if selected and selected["action"]["status"] == "visible":
                        final_frame = capture(info["port"])
                        final_image = save_frame(episode, final_frame, propose_balls(final_frame), "final")
                        reward_rows.append(dict(row, id=f"{row['id']}-outcome", image=str(final_image.relative_to(args.data)),
                            image_sha256=digest(final_image), history_image=row["image"], history_sha256=row["image_sha256"],
                            prompt=outcome_prompt(instruction), completion="yes" if entry["verified_success"] else "no",
                            reward_label=int(entry["verified_success"]), reward_scope="task"))
                        planner.model.set_adapter("reward")
                        entry["outcome_reward"] = reward_score(planner.model, planner.processor,
                            [image, final_image], outcome_prompt(instruction))
                        acceptance_reward = entry["outcome_reward"]
                    entry["accepted_for_replay"] = bool(entry["verified_success"] and acceptance_reward >= 0.5)
                    if entry["accepted_for_replay"]:
                        actor_rows.append(row)
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
            finally:
                if robot:
                    try:
                        robot.stop()
                    except Exception as error:
                        entry["stop_error"] = str(error)
                    try:
                        robot.close()
                    except Exception as error:
                        entry["robot_close_error"] = str(error)
                if observer:
                    try:
                        observer.close()
                    except Exception as error:
                        entry["observer_close_error"] = str(error)
                if process:
                    stop_process(process)
                if log:
                    log.close()
                report["cases"].append(entry)
                save()
                print(json.dumps({"seed": seed, "success": entry["verified_success"], "error": entry.get("error")}), flush=True)
        report.update(status="completed", actor_replay_rows=len(actor_rows), reward_replay_rows=len(reward_rows),
            success_count=sum(c["verified_success"] for c in report["cases"]), total=len(report["cases"]))
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
