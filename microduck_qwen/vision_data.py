"""Collect real MuJoCo RGB-D observations with simulator-verified visual labels.

The initial SFT objective is visual referent selection, not joint/trajectory imitation.
Privileged state is kept in episode sidecars, never in model messages.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from . import ROOT
from .navigation import JsonConnection, Observer, Robot, execute
from .vision import Frame, decode_array, project, propose_balls, save_frame
from .vision_eval import start_scene, stop_process


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request(port, op):
    connection = JsonConnection(("127.0.0.1", port))
    try:
        connection.send({"op": op})
        return connection.receive()
    finally:
        connection.close()


def collection_frame(port):
    packet = request(port, "collection_frame")
    metadata = packet["sensor"]
    if not 0 <= (time.time_ns() - metadata["wall_ns"]) / 1e9 < 2:
        raise RuntimeError("stale collection frame")
    shape = metadata["height"], metadata["width"]
    frame = Frame(decode_array(metadata.pop("rgb_zlib"), np.uint8, (*shape, 3)),
                  decode_array(metadata.pop("depth_zlib"), "<f4", shape), metadata)
    if frame.metadata["sim_time"] != packet["ground_truth"]["sim_time"]:
        raise ValueError("RGB/segmentation must come from the same rendered scene")
    return frame, packet["ground_truth"]


def bbox_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(1, area_a + area_b - overlap)


def verify_objects(scene, frame, gt, candidates):
    """Associate RGB candidate labels with independent simulator segmentation.

    No segmentation ID or true coordinate is put in the numbered RGB image.
    Ambiguous detector/visibility cases are discarded, not silently relabeled.
    """
    verified = []
    for name, obj in scene["objects"].items():
        visible = gt["objects"][name]
        row = {"name": name, "color": obj["color"], "status": "not_visible", "candidate": None}
        if visible["visible_pixels"] >= 20:
            pixel = project([obj["x"], obj["y"], obj["radius"]], frame.metadata)
            if pixel is None:
                row["status"] = "discard"
                row["pixel"] = visible["pixel"]
            else:
                radius = frame.metadata["fy"] * obj["radius"] / pixel[2]
                fraction = visible["visible_pixels"] / max(1, math.pi * radius**2)
                row["visible_fraction_gt"] = fraction
                row["pixel"] = visible["pixel"]
                matches = [c for c in candidates if c["color"] == obj["color"]
                           and bbox_iou(c["bbox"], visible["bbox"]) > 0.45]
                if fraction >= 0.85 and len(matches) == 1:
                    row.update(status="visible", candidate=matches[0])
                elif fraction <= 0.55:
                    row["status"] = "partial"
                else:
                    row["status"] = "discard"
        verified.append(row)
    return verified


def visual_examples(verified):
    def answer(obj):
        status = obj["status"] if obj else "not_visible"
        if status == "discard":
            return None
        return {"status": status, "color": obj["color"] if status == "visible" else None,
                "id": obj["candidate"]["id"] if status == "visible" else None}

    rows = []
    for color, cn in (("red", "红"), ("blue", "蓝"), ("green", "绿")):
        matches = [obj for obj in verified if obj["color"] == color]
        if len(matches) > 1:
            expected = {"status": "ambiguous", "color": None, "id": None}
        else:
            expected = answer(matches[0] if matches else None)
        if expected is not None:
            rows.append((f"走到{cn}色球旁边", expected, "color"))
    visible = [obj for obj in verified if obj["status"] != "not_visible"]
    if visible:
        for side, choose in (("左", min), ("右", max)):
            # A discarded/partial extreme is never replaced by an easier visible object.
            target = choose(visible, key=lambda obj: obj["pixel"][0])
            expected = answer(target)
            if expected is not None:
                rows.append((f"走到画面{side}边的球旁边", expected, "side"))
    rows.extend([("停下来", {"status": "stop", "color": None, "id": None}, "stop"),
                 ("请画一幅画", {"status": "unsupported", "color": None, "id": None}, "unsupported")])
    return rows


def load_splits(directory):
    """Integrity and group-leak checks shared by training and evaluation."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "simulation_visual_v1" or manifest.get("status") != "completed":
        raise ValueError("dataset must be a completed simulation_visual_v1 collection")
    result, seen, seen_images = {}, set(), set()
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.jsonl"
        if digest(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} manifest hash mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows:
            raise ValueError(f"{split} is empty")
        groups = {row["scene_seed"] for row in rows}
        if seen & groups:
            raise ValueError("scene seed leakage across splits")
        seen |= groups
        image_hashes = {row["image_sha256"] for row in rows}
        if seen_images & image_hashes:
            raise ValueError("image leakage across splits")
        seen_images |= image_hashes
        for row in rows:
            if row["split"] != split:
                raise ValueError("row split mismatch")
            image = (directory / row["image"]).resolve()
            if not image.is_relative_to(directory) or digest(image) != row["image_sha256"]:
                raise ValueError("image path or hash invalid")
            if set(row["expected"]) != {"status", "color", "id"}:
                raise ValueError("unexpected label schema")
            answer = row["expected"]
            if answer["status"] == "visible":
                if answer["color"] not in ("red", "blue", "green") or type(answer["id"]) is not int or answer["id"] < 1:
                    raise ValueError("invalid visible label")
            elif answer["status"] not in ("partial", "not_visible", "ambiguous", "stop", "unsupported") or answer["color"] is not None or answer["id"] is not None:
                raise ValueError("invalid non-visible label")
        result[split] = rows
    return manifest, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-scenes", type=int, default=12)
    parser.add_argument("--valid-scenes", type=int, default=6)
    parser.add_argument("--test-scenes", type=int, default=6)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--port", type=int, default=7842)
    parser.add_argument("--navigate", action="store_true", help="record oracle expert trajectories for clear training scenes")
    args = parser.parse_args()
    if min(args.train_scenes, args.valid_scenes, args.test_scenes) < 1 or args.seed_offset < 0:
        parser.error("positive scene counts and nonnegative seed offset required")
    if max(args.train_scenes, args.valid_scenes, args.test_scenes) >= 10000:
        parser.error("scene counts must stay within disjoint seed ranges")
    output = (args.output or ROOT / "data" / ("simulation-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "simulation_visual_v1", "status": "collecting", "episodes": [], "splits": {},
                "model_inputs": ["annotated_rgb", "instruction"],
                "supervision": "MuJoCo segmentation and scene truth; per-frame grounding, not joint imitation",
                "navigate": args.navigate, "seed_offset": args.seed_offset}
    all_rows = {split: [] for split in ("train", "valid", "test")}

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        for split, count, base in (("train", args.train_scenes, 10000), ("valid", args.valid_scenes, 20000), ("test", args.test_scenes, 30000)):
            for index in range(count):
                seed = base + args.seed_offset * 100000 + index
                kind = ("none", "partial", "full", "absent", "none", "none")[index % 6]
                episode = output / "episodes" / f"{split}-{seed}-{kind}"
                episode.mkdir(parents=True)
                entry = {"split": split, "seed": seed, "scenario": kind, "frames": [], "status": "started"}
                process = log = robot = observer = None
                print(f"Collecting {split} seed={seed} scenario={kind}", flush=True)
                try:
                    process, log, info, failures = start_scene(episode, seed, kind, args.port, collect_gt=True)
                    scene = json.loads(Path(info["scene_file"]).read_text())
                    (episode / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
                    entry["startup_failures"] = failures
                    robot, observer = Robot(info["socket"]), Observer(args.port)
                    robot.stop()

                    def record():
                        frame, gt = collection_frame(args.port)
                        candidates = propose_balls(frame)
                        verified = verify_objects(scene, frame, gt, candidates)
                        number = len(entry["frames"])
                        prefix = f"frame-{number:03d}"
                        marked = save_frame(episode, frame, candidates, prefix)
                        (episode / f"{prefix}-truth.json").write_text(json.dumps({"ground_truth": gt, "verified": verified}, indent=2) + "\n")
                        rows = []
                        for instruction, expected, task_type in visual_examples(verified):
                            rows.append({"id": f"{split}-{seed}-{number}-{len(rows)}", "scene_seed": seed,
                                         "split": split, "scenario": kind, "frame": number,
                                         "image": str(marked.relative_to(output)), "image_sha256": digest(marked),
                                         "instruction": instruction, "expected": expected, "task_type": task_type,
                                         "label_source": "independent MuJoCo segmentation + task specification"})
                        all_rows[split].extend(rows)
                        entry["frames"].append({"frame": number, "sim_time": frame.metadata["sim_time"], "examples": len(rows)})
                        return verified

                    verified = record()
                    if args.navigate and split == "train" and kind == "none":
                        targets = [obj for obj in verified if obj["status"] == "visible"]
                        if targets:
                            selected = min(targets, key=lambda obj: obj["pixel"][0])
                            target = scene["objects"][selected["name"]]
                            last_record = [time.monotonic()]

                            def guard(state, goal):
                                if time.monotonic() - last_record[0] > 2 and len(entry["frames"]) < 4:
                                    record()
                                    last_record[0] = time.monotonic()
                                return {"safe": True, "status": "oracle_expert_collection"}

                            plan = {"status": "ok", "goals": [{"target": "point", "x": target["x"], "y": target["y"],
                                    "speed": "slow", "stop_distance_m": 0.30}]}
                            expert = execute(plan, robot, observer, episode / "trajectory.jsonl", timeout_s=35,
                                             scene=scene, guard=guard, observation_label="oracle expert demonstration")
                            state = observer.read()
                            stats = request(args.port, "collection_stats")
                            distance = math.dist(state["trunk"][:2], [target["x"], target["y"]])
                            entry["expert"] = {"target": selected["name"], "plan": plan, "execution": expert,
                                               "distance_m": distance, "contacts": stats,
                                               "verified_success": bool(expert["success"] and distance <= 0.325
                                                                        and stats["steps_with_scene_contact"] == 0)}
                            record()
                    entry["status"] = "completed"
                except Exception as error:
                    entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                    # Correct per-frame grounding labels remain useful; no failed motion is a successful demonstration.
                finally:
                    if robot is not None:
                        try:
                            robot.stop()
                        except Exception as error:
                            entry.setdefault("cleanup_errors", []).append(str(error))
                        try:
                            robot.close()
                        except Exception as error:
                            entry.setdefault("cleanup_errors", []).append(str(error))
                    if observer is not None:
                        try:
                            observer.close()
                        except Exception as error:
                            entry.setdefault("cleanup_errors", []).append(str(error))
                    if process is not None:
                        stop_process(process)
                    if log is not None:
                        log.close()
                    (episode / "episode.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
                    manifest["episodes"].append(entry)
                    save_manifest()
                print(json.dumps({"seed": seed, "status": entry["status"], "frames": len(entry["frames"]), "error": entry.get("error")}), flush=True)
        for split, rows in all_rows.items():
            path = output / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            manifest["splits"][split] = {"examples": len(rows), "scene_seeds": sorted({r["scene_seed"] for r in rows}),
                                         "sha256": digest(path), "statuses": dict(Counter(r["expected"]["status"] for r in rows))}
        if any(not rows for rows in all_rows.values()):
            raise ValueError("one or more splits are empty; inspect episode failures")
        manifest["status"] = "completed"
        save_manifest()
        load_splits(output)
        print(f"DATASET: {output}", flush=True)
    except BaseException:
        manifest["status"] = "failed_or_interrupted"
        save_manifest()
        raise


if __name__ == "__main__":
    main()
