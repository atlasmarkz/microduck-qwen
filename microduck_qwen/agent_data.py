"""Expert-intervened simulator data aggregation; NOT unassisted DAgger or online RL.

Teacher task DSL and segmentation are used only for labels/scoring. Model input:
numbered RGB, instruction, own oracle odometry and completed bounded actions.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil

from . import ROOT
from .agent import AgentPlanner, agent_prompt, checked_memory
from .skills import SkillExecutor, memory_state, validate_action
from .vision import propose_balls, save_frame
from .vision_data import collection_frame, digest, request, verify_objects
from .sim_runner import start_scene, stop_process


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
    return result


def teacher_action(spec, verified):
    if "teacher_target" not in spec:
        return validate_action(spec.copy())
    visible = [obj for obj in verified if obj["status"] != "not_visible"]
    if not visible:
        raise ValueError("teacher target absent")
    obj = min(visible, key=lambda item: item["pixel"][0])
    if obj["status"] != "visible" or obj["candidate"]["visibility"] != "visible":
        raise ValueError("teacher leftmost target not confidently visible")
    return {"action": "approach", "color": obj["color"], "id": obj["candidate"]["id"]}


def episode(output, directory, info, split, seed, instruction, specs, planner=None, intervene=True):
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
        for index, spec in enumerate(specs + ([{"action": "done"}] if specs[-1].get("action") not in ("stop", "clarify", "unsupported") else [])):
            executor.robot.stop()
            frame, gt = collection_frame(info["port"])
            candidates = propose_balls(frame)
            verified = verify_objects(executor.scene, frame, gt, candidates)
            expected = teacher_action(spec, verified)
            image = save_frame(directory, frame, candidates, f"step-{index:03d}")
            (directory / f"step-{index:03d}-truth.json").write_text(json.dumps({"ground_truth": gt, "verified_objects": verified}, indent=2) + "\n")
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
            result = executor.run(action, candidates)
            step["execution"] = result
            report["steps"].append(step)
            terminal = action["action"] in ("done", "stop", "clarify", "unsupported")
            physical_ok = result["success"] or terminal
            if not physical_ok or (not intervene and action != expected):
                report["error"] = "failed physical skill or wrong task action (scored AFTER execution)"
                break
            if action["action"] == "approach":
                obj = next(o for o in verified if o["candidate"] and o["candidate"]["id"] == action["id"] and o["color"] == action["color"])
                target = executor.scene["objects"][obj["name"]]
                import math
                distance = math.dist(executor.healthy()["trunk"][:2], [target["x"], target["y"]])
                step["true_goal_distance_m"] = distance
                if distance > 0.325:
                    raise RuntimeError("approach failed independent true-distance check")
            rows.append({"id": f"{split}-{seed}-{directory.name}-{index}", "scene_seed": seed, "split": split,
                         "image": str(image.relative_to(output)), "image_sha256": digest(image),
                         "instruction": instruction, "memory": memory, "expected": expected,
                         "verified": True, "task_type": expected["action"],
                         "label_source": "task DSL + physically verified expert + MuJoCo segmentation"})
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
    parser.add_argument("--actor-adapter", type=Path)
    parser.add_argument("--base-data", type=Path, help="keep old valid/test fixed; collect TRAIN corrections only")
    args = parser.parse_args()
    if not 1 <= args.train_scenes <= 100 or not 1 <= args.valid_scenes <= 100 or not 1 <= args.test_scenes <= 100 or not 0 <= args.seed_offset < 9000:
        parser.error("invalid scene counts/seed offset")
    if bool(args.base_data) != bool(args.actor_adapter):
        parser.error("correction collection requires BOTH base-data and actor-adapter")
    output = args.output.resolve()
    if args.base_data and output.is_relative_to(args.base_data.resolve()):
        parser.error("output must not be inside base-data")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "simulation_agent_v1", "status": "collecting", "episodes": [], "splits": {},
                "inputs": ["numbered RGB", "instruction", "own oracle odometry", "completed actions"],
                "method": "expert-intervened data aggregation; not unrestricted DAgger or learned reward model",
                "seed_offset": args.seed_offset}
    manifest["source_sha256"] = {name: digest(ROOT / "microduck_qwen" / name) for name in
        ("agent_data.py", "skills.py", "navigation.py", "vision.py", "vision_scene.py", "sim.py", "sim_body.py")}
    splits = {split: [] for split in ("train", "valid", "test")}
    planner = None
    if args.base_data:
        base_manifest, base_splits = load_splits(args.base_data)
        shutil.copytree(args.base_data, output / "base")
        for split, rows in base_splits.items():
            splits[split] = [dict(row, image="base/" + row["image"]) for row in rows]
        manifest["parent_manifest_sha256"] = digest(args.base_data / "manifest.json")
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
                process = log = None
                try:
                    process, log, info, failures = start_scene(scene_dir, seed, "none", args.port, collect_gt=True)
                    shutil.copyfile(info["scene_file"], scene_dir / "scene.json")
                    manifest.setdefault("startup_failures", []).extend(failures)
                    variant = 1 + i % 2 if args.base_data else i % 3
                    for index, (instruction, specs) in enumerate(tasks(split, variant)):
                        directory = scene_dir / f"task-{index:02d}"
                        rows, report = episode(output, directory, info, split, seed, instruction, specs, planner, True)
                        if split == "train":
                            # Same physically verified intent/history, explicit template paraphrases.
                            # These are NOT additional physical trajectories or independent samples.
                            augmented = []
                            for alternative in range(3):
                                text = tasks("train", alternative)[index][0]
                                if text != instruction:
                                    augmented.extend(dict(row, id=row["id"] + f"-paraphrase-{alternative}",
                                                          instruction=text, language_source="finite teacher template paraphrase") for row in rows)
                            rows.extend(augmented)
                        splits[split].extend(rows)
                        manifest["episodes"].append({"seed": seed, "split": split, "task": index, "verified": report["verified"],
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
