"""Expert-intervened simulator data aggregation; NOT unassisted DAgger or online RL.

Teacher task DSL and segmentation are used only for labels/scoring. Model input:
numbered RGB, instruction, own oracle odometry and completed bounded actions.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

import random

from . import ROOT
from .agent_language import colour_tasks, explore_paraphrases, explore_tasks, paraphrases
from .agent import AgentPlanner, agent_prompt, checked_memory
from .skills import NeedsUserInput, SkillExecutor, bind_target, memory_state, validate_action
from .vision import propose_targets, save_frame
from .sim_data import collection_frame, digest, request, verify_objects
from .sim_runner import start_scene, stop_process
from .vision_world import WORLD_LEVELS


# Hard scene profiles as (occlusion, clutter). Split offsets below keep the splits on
# different difficulty and, with disjoint seed ranges, keep them leakage-free.
#   train 0..2 -> none/full/absent with the duplicated-colour row
#   valid      -> partial occlusion with the duplicated-colour row
#   test       -> full occlusion with the duplicated-colour row (the watched scene)
SCENE_PROFILES = [
    ("none", True),
    ("full", True),
    ("absent", True),
    ("partial", True),
    ("full", True),
    ("none", False),
]
PROFILE_BASE = {"train": 0, "valid": 3, "test": 4}

# The approach plan stops at 0.30m and execute() breaks its loop at 0.30 + 0.025m
# of arrival tolerance, but the airframe coasts a few centimetres before it is
# actually at rest. The teacher's independent check measures the resting pose, so
# it needs its own allowance: without one it discards sound demonstrations for a
# few millimetres of drift (measured on seeded hard-valid-71700: in-loop 0.3211m,
# at rest 0.3303m, against a bare 0.325m bound).
APPROACH_STOP_M = 0.30
ARRIVAL_TOLERANCE_M = 0.025
COAST_ALLOWANCE_M = 0.05
APPROACH_TRUE_DISTANCE_M = APPROACH_STOP_M + ARRIVAL_TOLERANCE_M + COAST_ALLOWANCE_M


def world_arguments(mix, level):
    """start_scene keywords for the world mix; empty for every other mix."""
    if mix != "world":
        return {}
    return {"world": WORLD_LEVELS[level]["rooms"], "density": WORLD_LEVELS[level]["density"]}


def scene_profile(split, index, mix):
    if mix in ("plain", "yellow", "varied", "world"):
        return "none", False
    return SCENE_PROFILES[(PROFILE_BASE[split] + index) % len(SCENE_PROFILES)]


def model_rows(rows, directory):
    return [{"image": str((directory / row["image"]).resolve()), "messages": [
        {"role": "user", "content": agent_prompt(row["instruction"], row["memory"])},
        {"role": "assistant", "content": json.dumps(row["expected"], separators=(",", ":"))}
    ]} for row in rows]


def load_splits(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "simulation_agent_v1" or manifest.get("status") != "completed":
        raise ValueError("requires completed simulation_agent_v1 dataset")
    splits, seeds, images, ids = {}, set(), set(), set()
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.jsonl"
        if digest(path) != manifest["splits"][split]["sha256"]:
            raise ValueError("split hash mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            raise ValueError("empty split")
        groups = {row["scene_seed"] for row in rows}
        hashes = {row["image_sha256"] for row in rows}
        if seeds & groups or images & hashes:
            raise ValueError("scene/image leakage")
        seeds |= groups
        images |= hashes
        for row in rows:
            image = (directory / row["image"]).resolve()
            if not image.is_relative_to(directory) or digest(image) != row["image_sha256"]:
                raise ValueError("image path/hash invalid")
            if row["split"] != split or row["id"] in ids or not row["verified"]:
                raise ValueError("split/id/verification invalid")
            ids.add(row["id"])
            checked_memory(row["memory"])
            validate_action(row["expected"])
        splits[split] = rows
    return manifest, splits


def tasks(split, variant=0):
    """Finite teacher tasks, not a runtime natural-language parser. Held-out phrasing."""
    heldout = split != "train"
    move = lambda direction: {"action": "move", "direction": direction, "distance_m": 0.2}
    turn = lambda direction: {"action": "turn", "direction": direction, "degrees": 45}
    home = {"action": "return_home"}
    result = [
        ("先往前走20厘米，然后回到起点" if heldout else "前进0.2米，再回到初始位置", [move("forward"), home]),
        ("先向后退20厘米，随后回到出发点" if heldout else "后退0.2米，再回到初始位置", [move("backward"), home]),
        ("先左转45度，再右转45度" if heldout else "向左转45度，然后向右转45度", [turn("left"), turn("right")]),
        ("靠近照片里最左侧的球，之后返回起点" if heldout else "走到画面左边的球旁边，然后回到初始位置", [{"teacher_target": "left"}, home]),
        ("往前移动20厘米" if heldout else "前进0.2米", [move("forward")]),
        ("向后退20厘米" if heldout else "后退0.2米", [move("backward")]),
        ("立即停止移动" if heldout else "停下来", [{"action": "stop"}]),
        ("帮我把球踢走" if heldout else "踢一下蓝色球", [{"action": "unsupported"}]),
        ("到那个地方去" if heldout else "去那里", [{"action": "clarify"}]),
    ]
    if split == "train" and variant:
        variants = [
            ["往前挪0.2米，接着返回出发位置", "倒退0.2米，然后返回出发位置", "先往左旋转45度，再往右旋转45度",
             "接近图片左侧的球，再返回出发位置", "往前挪0.2米", "倒退0.2米", "原地停住", "踢走绿色的球", "去那边"],
            ["向前行进二十厘米后回原点", "向后移动二十厘米后回原点", "左旋四十五度后右旋四十五度",
             "先接近图中最左边的球，然后回原点", "向前行进二十厘米", "向后移动二十厘米", "不要再移动了", "跳一支舞", "到那儿"],
        ]
        result = [(text, specs) for text, (_, specs) in zip(variants[(variant - 1) % 2], result)]
    if split == "test":
        test_text = ["前行20厘米后返回最初的位置", "先后退20厘米再返回最初的位置", "左转45度后请右转45度",
                     "走近画面最左球后回到最初位置", "前行20厘米", "后退20厘米", "停止行进", "把红球踢开", "去那个位置"]
        result = [(text, specs) for text, (_, specs) in zip(test_text, result)]
    phrases = {
        "train": [
            ["依次绕过黄色障碍物并回到原位", "依次从左侧绕过黄色障碍物，再回到原位", "依次从右侧绕过黄色障碍物，再回到原位"],
            ["按从近到远的顺序绕开黄色柱子，最后返回出发位置", "从左边逐个绕过黄色柱子后回出发点", "从右边逐个绕过黄色柱子后回出发点"],
            ["逐个经过黄色障碍后方，然后回到本次任务起点", "逐个从左边经过黄色障碍后方并返回起点", "逐个从右边经过黄色障碍后方并返回起点"]],
        "valid": [["绕行所有黄色柱体，按由近到远依次通过后返回开始的位置", "按近远顺序从左边绕过各黄色柱体，最后返回任务开始处", "按近远顺序从右边绕过各黄色柱体，最后返回任务开始处"]],
        "test": [["请逐一绕开黄色障碍，完成后回到刚才出发的地方", "请逐一从黄色障碍左边通过，随后回到刚才出发的地方", "请逐一从黄色障碍右边通过，随后回到刚才出发的地方"]]}
    selected = phrases[split][variant % 3 if split == "train" else 0]
    result.extend((text, [{"action": "bypass_return", "side": side}])
                  for text, side in zip(selected, ("auto", "left", "right")))
    return result


def explore_entries(info, split, seed, per_color=1):
    """Explore-and-approach tasks for the colours that exist in the world but are not in the first view."""
    frame, _ = collection_frame(info["port"])
    seen = {c["color"] for c in propose_targets(frame) if c.get("kind", "ball") == "ball"}
    scene = json.loads(Path(info["scene_file"]).read_text())
    hidden = {o["color"] for o in scene["objects"].values()} - seen
    return [(200 + k, text, specs, meta) for k, (text, specs, meta) in
            enumerate(explore_tasks(split, hidden, random.Random(seed), per_color))]


def teacher_action(spec, verified):
    if "teacher_nearest" in spec:
        color = spec["teacher_nearest"]
        matches = [o for o in verified if o["color"] == color and o["status"] == "visible"
                   and o["candidate"]["visibility"] == "visible"]
        if not matches:
            raise ValueError(f"no confidently visible {color} ball to approach")
        action = {"action": "approach", "color": color, "selector": "nearest"}
        if color == "yellow":
            action["kind"] = "ball"
        return validate_action(action)
    if "teacher_color" in spec:
        color = spec["teacher_color"]
        matches = [o for o in verified if o["color"] == color]
        # Exactly one confidently visible object of that colour; otherwise refuse the label.
        if len(matches) != 1 or matches[0]["status"] != "visible" or matches[0]["candidate"]["visibility"] != "visible":
            raise ValueError(f"teacher colour target {color} not uniquely and confidently visible")
        candidate = matches[0]["candidate"]
        if color == "yellow":
            return validate_action({"action": "approach", "color": "yellow", "kind": "ball", "selector": "only"})
        return validate_action({"action": "approach", "color": color, "id": candidate["id"]})
    if "teacher_target" not in spec:
        return validate_action(spec.copy())
    visible = [obj for obj in verified if obj["status"] != "not_visible"]
    if not visible:
        raise ValueError("teacher target absent")
    obj = min(visible, key=lambda item: item["pixel"][0])
    if obj["status"] != "visible" or obj["candidate"]["visibility"] != "visible":
        raise ValueError("teacher leftmost target not confidently visible")
    if obj["color"] == "yellow":
        # Yellow targets are bound by selector, never a guessed id; refuse when it is ambiguous.
        if sum(1 for o in visible if o["color"] == "yellow" and o["candidate"]) != 1:
            raise ValueError("several yellow balls in view; teacher leftmost target is ambiguous")
        return validate_action({"action": "approach", "color": "yellow", "kind": "ball", "selector": "only"})
    return {"action": "approach", "color": obj["color"], "id": obj["candidate"]["id"]}


def same_action(action, expected, candidates, position):
    """Whether an actor action is the teacher's action, judged by what it does.

    Approach targets are compared after binding: `selector=leftmost` and the
    teacher's `id=4` are the same action when both bind the same current-frame
    candidate. Anything that fails to bind, or binds a different object, is a
    different action. Every other action must match exactly. Used only to SCORE
    independent rollouts; labels and execution still use the strict schema.
    """
    if action == expected:
        return True
    if not action or action.get("action") != "approach" or expected.get("action") != "approach":
        return False
    try:
        chosen = bind_target(action, candidates, position)
        wanted = bind_target(expected, candidates, position)
    except NeedsUserInput:
        return False
    return chosen["id"] == wanted["id"] and chosen["color"] == wanted["color"]


def score_layers(report):
    """Separate why an independent episode failed: format, action type, target, motion, task."""
    steps = report["steps"]
    return {
        "format_ok": report.get("error") != "invalid actor JSON",
        "action_type_ok": all(s.get("action_type_ok", True) for s in steps),
        "target_ok": all(s.get("target_ok", True) for s in steps),
        "physical_ok": all(s.get("physical_ok", False) for s in steps if "execution" in s),
        "task_ok": bool(report["verified"]),
    }


def episode(output, directory, info, split, seed, instruction, specs, planner=None, intervene=True, meta=None):
    directory.mkdir(parents=True)
    report = {"instruction": instruction, "teacher_specs": specs, "steps": [], "verified": False,
              "intervene": intervene, "mode": "expert" if planner is None else "shielded_actor" if intervene else "independent_actor"}
    executor = SkillExecutor(info, directory)
    rows, completed = [], []
    contact_start = request(info["port"], "collection_stats")["steps_with_scene_contact"]
    try:
        # Restore home between tasks. Reset is NOT included in model history or supervised actions.
        reset = executor.run({"action": "return_home"})
        report["reset"] = reset
        if not reset["success"]:
            raise RuntimeError("cannot restore task start")
        initial = executor.healthy()
        executor.task_start_pose = {"x": initial["trunk"][0], "y": initial["trunk"][1], "yaw": initial["yaw"]}
        for index, spec in enumerate(specs + ([{"action": "done"}] if specs[-1].get("action") not in ("stop", "clarify", "unsupported") else [])):
            executor.robot.stop()
            frame, gt = collection_frame(info["port"])
            candidates = propose_targets(frame)
            verified = verify_objects(executor.scene, frame, gt, candidates)
            # Persist the observation BEFORE the teacher can refuse it. A refused
            # task is exactly the case whose frame and verdicts have to stay
            # inspectable, and teacher_action raises on purpose.
            image = save_frame(directory, frame, candidates, f"step-{index:03d}")
            (directory / f"step-{index:03d}-truth.json").write_text(json.dumps({"ground_truth": gt, "verified_objects": verified}, indent=2) + "\n")
            expected = teacher_action(spec, verified)
            memory = memory_state(info, executor.healthy(), completed)
            step = {"expected": expected, "memory": memory}
            action = expected
            if planner is not None:
                prediction = planner.predict(instruction, image, memory)
                matches = prediction.get("action") == expected
                step.update(prediction=prediction, semantic_match=matches, teacher_intervened=intervene and not matches)
                if not intervene:
                    action = prediction.get("action")
                    if action is None:
                        report["error"] = "invalid actor JSON"
                        report["steps"].append(step)
                        break
            equivalent = True
            if not intervene:
                # Judge by effect, before the body moves: binding uses this frame and pose.
                equivalent = same_action(action, expected, candidates, executor.healthy()["trunk"][:2])
                both_approach = action["action"] == expected["action"] == "approach"
                step.update(equivalent_to_teacher=equivalent, exact_match=action == expected,
                            action_type_ok=action["action"] == expected["action"],
                            # Target correctness only exists for approach steps; any other
                            # mismatch is an action-type or parameter error, not a wrong target.
                            target_ok=equivalent or not both_approach)
            result = executor.run(action, candidates)
            step["execution"] = result
            report["steps"].append(step)
            terminal = action["action"] in ("done", "stop", "clarify", "unsupported")
            physical_ok = result["success"] or terminal
            step["physical_ok"] = bool(physical_ok)
            if not physical_ok or not equivalent:
                report["error"] = "failed physical skill or wrong task action (scored AFTER execution)"
                break
            if action["action"] == "approach":
                bound = result.get("bound_target")
                ident = bound["id"] if bound else action.get("id")
                obj = next(o for o in verified if o["candidate"] and o["color"] == action["color"]
                           and o["candidate"]["id"] == ident)
                target = executor.scene["objects"][obj["name"]]
                import math
                distance = math.dist(executor.healthy()["trunk"][:2], [target["x"], target["y"]])
                step["true_goal_distance_m"] = distance
                if distance > APPROACH_TRUE_DISTANCE_M:
                    raise RuntimeError("approach failed independent true-distance check")
            rows.append({"id": f"{split}-{seed}-{directory.name}-{index}", "scene_seed": seed, "split": split,
                         "scene_occlusion": info.get("occlusion"), "scene_clutter": bool(info.get("clutter")),
                         "scene_yellow_course": bool(info.get("yellow_course")),
                         "image": str(image.relative_to(output)), "image_sha256": digest(image),
                         "instruction": instruction, "memory": memory, "expected": expected,
                         "verified": True, "task_type": expected["action"], "task_meta": meta,
                         "label_source": "task DSL + physically verified expert + MuJoCo segmentation"})
            if planner is not None and "proposals" in prediction:
                rows[-1]["reward_proposals"] = [p["action"] for p in prediction["proposals"] if p.get("action") is not None]
                rows[-1]["reward_verified_accepted"] = bool(prediction.get("reward_accepted") and matches)
            if terminal:
                report["verified"] = True
                break
            completed.append(action)
        stats = request(info["port"], "collection_stats")
        report["scene_contact_steps_delta"] = stats["steps_with_scene_contact"] - contact_start
        report["verified"] &= report["scene_contact_steps_delta"] == 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            executor.close()
        finally:
            if not intervene:
                report["layers"] = score_layers(report)
            (directory / "episode.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Failed episodes never become successful demos; raw failures retained in sidecar.
    return rows if report["verified"] else [], report


def write_splits(output, manifest, splits):
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][split] = {"sha256": digest(path), "examples": len(rows),
                                     "scene_seeds": sorted({r["scene_seed"] for r in rows}),
                                     "actions": dict(Counter(r["task_type"] for r in rows))}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=3)
    parser.add_argument("--valid-scenes", type=int, default=1)
    parser.add_argument("--test-scenes", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=10)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--scene-mix", choices=("plain", "hard", "yellow", "varied", "world"), default="plain",
                        help="plain reproduces the open-row scenes; hard adds occluded/absent/duplicated-row scenes; "
                             "world uses multi-room worlds of --world-level")
    parser.add_argument("--world-level", type=int, default=0, choices=range(len(WORLD_LEVELS)),
                        help="world mix only: curriculum level (rooms and clutter grow with it)")
    parser.add_argument("--actor-adapter", type=Path)
    parser.add_argument("--holdout-color", choices=("red", "blue", "green", "yellow"),
                        help="varied mix: never target this colour in train; valid/test target only it (composition test)")
    parser.add_argument("--explore-tasks", type=int, default=0,
                        help="world mix: explore-and-approach tasks per colour hidden from the first view")
    parser.add_argument("--language-per-color", type=int, default=2, help="varied mix: colour-approach tasks per colour per scene")
    parser.add_argument("--task-ids", type=int, nargs="+", help="subset of finite teacher tasks; yellow defaults to task 9")
    parser.add_argument("--reward-adapter", type=Path)
    parser.add_argument("--reward-samples", type=int, default=3)
    parser.add_argument("--reward-threshold", type=float, default=0.5)
    parser.add_argument("--base-data", type=Path, help="keep old valid/test fixed; collect TRAIN corrections only")
    args = parser.parse_args()
    if not 1 <= args.train_scenes <= 100 or not 1 <= args.valid_scenes <= 100 or not 1 <= args.test_scenes <= 100 or not 0 <= args.seed_offset < 9000:
        parser.error("invalid scene counts/seed offset")
    if bool(args.base_data) != bool(args.actor_adapter):
        parser.error("correction collection requires BOTH base-data and actor-adapter")
    if args.reward_adapter and (not args.actor_adapter or os.name != "nt"):
        parser.error("reward-guided collection requires Windows CUDA and an actor adapter")
    task_ids = args.task_ids if args.task_ids is not None else ([9] if args.scene_mix == "yellow" else list(range(9)))
    if len(set(task_ids)) != len(task_ids) or any(i < 0 or i >= len(tasks("train")) for i in task_ids):
        parser.error("invalid/duplicate task IDs")
    output = args.output.resolve()
    if args.base_data and output.is_relative_to(args.base_data.resolve()):
        parser.error("output must not be inside base-data")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "simulation_agent_v1", "status": "collecting", "episodes": [], "splits": {},
                "inputs": ["numbered RGB", "instruction", "own oracle odometry", "completed actions"],
                "method": "expert-intervened data aggregation; not unrestricted DAgger or learned reward model",
                "scene_mix": args.scene_mix, "seed_offset": args.seed_offset,
                "world_level": args.world_level if args.scene_mix == "world" else None}
    manifest["source_sha256"] = {name: digest(ROOT / "microduck_qwen" / name) for name in
        ("agent_data.py", "agent_language.py", "skills.py", "bypass.py", "navigation.py", "vision.py", "sim_data.py", "sim_runner.py", "vision_scene.py", "vision_world.py", "sim.py", "sim_body.py")}
    splits = {split: [] for split in ("train", "valid", "test")}
    planner = None
    if args.base_data:
        base_manifest, base_splits = load_splits(args.base_data)
        shutil.copytree(args.base_data, output / "base")
        for split, rows in base_splits.items():
            splits[split] = [dict(row, image="base/" + row["image"]) for row in rows]
        manifest["parent_manifest_sha256"] = digest(args.base_data / "manifest.json")
        if args.reward_adapter:
            from .vision_torch import RewardGuidedPlanner
            planner = RewardGuidedPlanner(args.actor_adapter, args.reward_adapter, args.reward_samples, args.reward_threshold)
            manifest["reward"] = planner.reward_adapter
            manifest["reward_sampling"] = {"samples": args.reward_samples, "threshold": args.reward_threshold,
                                           "temperature": 0.7, "seed": 42, "first_proposal_greedy": True}
            manifest["method"] = "reward-ranked proposals with independent expert intervention and verified feedback"
        else:
            planner = AgentPlanner(args.actor_adapter)
        manifest["actor"] = planner.adapter
        old_seeds = {row["scene_seed"] for rows in splits.values() for row in rows}
        if old_seeds & set(range(61000 + args.seed_offset, 61000 + args.seed_offset + args.train_scenes)):
            raise ValueError("correction seeds overlap parent data")
    try:
        configs = [("train", args.train_scenes, 61000)]
        if not args.base_data:
            configs.extend([("valid", args.valid_scenes, 71000), ("test", args.test_scenes, 81000)])
        for split, count, base in configs:
            for i in range(count):
                seed = base + args.seed_offset + i
                scene_dir = output / "episodes" / f"{split}-{seed}"
                scene_dir.mkdir(parents=True)
                occlusion, clutter = scene_profile(split, i, args.scene_mix)
                process = log = None
                try:
                    process, log, info, failures = start_scene(scene_dir, seed, occlusion, args.port,
                                                              collect_gt=True, clutter=clutter, yellow_course=args.scene_mix == "yellow",
                                                              varied=args.scene_mix == "varied",
                                                              **world_arguments(args.scene_mix, args.world_level))
                    shutil.copyfile(info["scene_file"], scene_dir / "scene.json")
                    manifest.setdefault("startup_failures", []).extend(failures)
                    variant = 1 + i % 2 if args.base_data else i % 3
                    entries = [(index, instruction, specs, None) for index, (instruction, specs) in enumerate(tasks(split, variant))
                               if index in task_ids]
                    language_rng = random.Random(seed)
                    if args.scene_mix == "world" and args.explore_tasks:
                        entries += explore_entries(info, split, seed, args.explore_tasks)
                    if args.scene_mix == "varied":
                        scene_colors = [o["color"] for o in json.loads(Path(info["scene_file"]).read_text())["objects"].values()]
                        language_rng = random.Random(seed)
                        entries += [(100 + k, text, language_specs, meta) for k, (text, language_specs, meta) in enumerate(
                            colour_tasks(split, scene_colors, language_rng, args.language_per_color, args.holdout_color))]
                    for index, instruction, specs, meta in entries:
                        directory = scene_dir / f"task-{index:02d}"
                        rows, report = episode(output, directory, info, split, seed, instruction, specs, planner, True, meta)
                        if args.reward_adapter:
                            stats = manifest.setdefault("reward_collection", {"sampled_proposals": 0, "distinct_proposals": 0,
                                "accepted_steps": 0, "rejected_steps": 0, "false_accept_steps": 0, "teacher_interventions": 0})
                            for step in report["steps"]:
                                prediction = step.get("prediction", {})
                                stats["sampled_proposals"] += len(prediction.get("proposals", []))
                                stats["distinct_proposals"] += prediction.get("distinct_proposals", 0)
                                accepted = prediction.get("reward_accepted", False)
                                stats["accepted_steps" if accepted else "rejected_steps"] += 1
                                stats["false_accept_steps"] += bool(accepted and not step.get("semantic_match", False))
                                stats["teacher_interventions"] += bool(step.get("teacher_intervened"))
                        if split == "train":
                            # Same physically verified intent/history, explicit template paraphrases.
                            # These are NOT additional physical trajectories or independent samples.
                            augmented = []
                            alternatives = ([tasks("train", alternative)[index][0] for alternative in range(3)] if meta is None
                                             else explore_paraphrases(meta["color"], meta["tail"], instruction, rng=language_rng)
                                             if meta["kind"] == "explore_find"
                                             else paraphrases(meta["color"], meta["tail"], instruction, rng=language_rng))
                            for alternative, text in enumerate(alternatives):
                                if text != instruction:
                                    # Paraphrases were not sampled/scored by the reward model.
                                    augmented.extend(dict(
                                        {k: v for k, v in row.items() if k not in ("reward_proposals", "reward_verified_accepted")},
                                        id=row["id"] + f"-paraphrase-{alternative}", instruction=text,
                                        language_source="finite teacher template paraphrase") for row in rows)
                            rows.extend(augmented)
                        splits[split].extend(rows)
                        manifest["episodes"].append({"seed": seed, "split": split, "task": index, "verified": report["verified"],
                            "occlusion": occlusion, "clutter": clutter,
                            "rows": len(rows), "error": report.get("error"),
                            "interventions": sum(s.get("teacher_intervened", False) for s in report["steps"])})
                        write_splits(output, manifest, splits)
                        print(json.dumps(manifest["episodes"][-1], ensure_ascii=False), flush=True)
                finally:
                    if process:
                        stop_process(process)
                    if log:
                        log.close()
        if any(not rows for rows in splits.values()):
            raise ValueError("empty split; inspect retained failures")
        if args.holdout_color:
            seen = {split: sum(1 for r in rows if (r.get("task_meta") or {}).get("color") == args.holdout_color) for split, rows in splits.items()}
            if seen["train"] or not seen["valid"] or not seen["test"]:
                raise ValueError(f"holdout colour {args.holdout_color} must be absent from train and present in valid/test: {seen}")
        manifest["status"] = "completed"
        write_splits(output, manifest, splits)
        load_splits(output)
    except BaseException as error:
        manifest.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
        write_splits(output, manifest, splits)
        raise
    print(f"AGENT DATASET: {output}", flush=True)


if __name__ == "__main__":
    main()
