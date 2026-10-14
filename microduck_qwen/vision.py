"""RGB-D proposals, real VLM referent selection, and conservative visual tracking.

This is a modular colored-ball prototype, NOT general object detection, SLAM or
an already fine-tuned VLA. Ground-truth scene labels never enter the VLM prompt.
"""

import argparse
import base64
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import random
import re
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import ROOT
from .navigation import JsonConnection, Observer, Robot, execute

MODEL = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
REVISION = "9c4f5209e57b31f4b9dfba735de3fb983739c9cc"
BALL_RADIUS = 0.07
MIN_VISIBILITY = 0.65


class VisualOutputError(ValueError):
    def __init__(self, message, prediction):
        super().__init__(message)
        self.prediction = prediction


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    metadata: dict


def decode_array(encoded, dtype, shape):
    count = math.prod(shape) * np.dtype(dtype).itemsize
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(encoded, validate=True), count + 1)
    if len(raw) != count or not decoder.eof:
        raise ValueError("camera array size/compression mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def capture(port, max_age=1.0):
    connection = JsonConnection(("127.0.0.1", port))
    try:
        connection.send({"op": "vision"})
        packet = connection.receive()
    finally:
        connection.close()
    age = (time.time_ns() - packet["wall_ns"]) / 1e9
    if not 0 <= age <= max_age:
        raise RuntimeError(f"camera stale ({age:.2f}s)")
    width, height = packet["width"], packet["height"]
    if not (16 <= width <= 1920 and 16 <= height <= 1080):
        raise ValueError("unsupported camera size")
    rgb = decode_array(packet.pop("rgb_zlib"), np.uint8, (height, width, 3))
    depth = decode_array(packet.pop("depth_zlib"), "<f4", (height, width))
    return Frame(rgb, depth, packet)


def unproject(u, v, z, metadata):
    camera = np.array(
        [
            ((u + 0.5) - metadata["cx"]) / metadata["fx"] * z,
            -((v + 0.5) - metadata["cy"]) / metadata["fy"] * z,
            -z,
        ]
    )
    rotation = np.asarray(metadata["camera_rotation"])
    return np.asarray(metadata["camera_position"]) + rotation @ camera


def project(point, metadata):
    local = np.asarray(metadata["camera_rotation"]).T @ (
        np.asarray(point) - metadata["camera_position"]
    )
    z = -local[2]
    if z <= 0:
        return None
    u = metadata["fx"] * local[0] / z + metadata["cx"] - 0.5
    v = -metadata["fy"] * local[1] / z + metadata["cy"] - 0.5
    return float(u), float(v), float(z)


def propose_balls(frame):
    """Known red/blue/green ball proposals from RGB, never MuJoCo segmentation IDs.

    IDs are shuffled to avoid encoding left-to-right order in label numbers.
    Visibility estimates assume the calibrated 7 cm spheres in this demo.
    """
    import cv2

    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue = {
        "red": (h < 12) | (h > 168),
        "blue": (h > 95) & (h < 135),
        "green": (h > 35) & (h < 90),
    }
    candidates = []
    for color, mask in hue.items():
        mask = (mask & (s > 180) & (v > 50)).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for index in range(1, count):
            x, y, w, height, area = [int(a) for a in stats[index]]
            if (
                area < 35
                or area > frame.rgb.shape[0] * frame.rgb.shape[1] * 0.25
                or not 0.15 < w / height < 4
            ):
                continue
            pixels = labels == index
            depths = frame.depth[pixels]
            valid = depths[np.isfinite(depths) & (depths > 0.06) & (depths < 3)]
            if len(valid) < area * 0.8:
                continue
            z = float(np.median(valid))
            u, vv = centers[index]
            surface = unproject(float(u), float(vv), z, frame.metadata)
            ray = surface - np.asarray(frame.metadata["camera_position"])
            position = surface + BALL_RADIUS * ray / np.linalg.norm(ray)
            # The checker floor is bluish and reflects the colored spheres. A
            # reflection has floor depth, not a sphere center above the ground.
            # This RGB-D geometric gate uses calibration, not simulator geom IDs.
            if not 0.03 < position[2] < 0.16:
                continue
            expected_area = (
                math.pi * (frame.metadata["fy"] * BALL_RADIUS / (z + BALL_RADIUS)) ** 2
            )
            fraction = min(1.0, area / max(1.0, expected_area))
            candidates.append(
                {
                    "color": color,
                    "bbox": [x, y, x + w, y + height],
                    "pixel": [float(u), float(vv)],
                    "world_position": position.tolist(),
                    "surface_depth_m": z,
                    "visible_fraction_estimate": fraction,
                    "visibility": "visible"
                    if fraction >= MIN_VISIBILITY
                    else "partial",
                }
            )
    ids = list(range(1, len(candidates) + 1))
    random.Random(zlib.crc32(frame.rgb.tobytes())).shuffle(ids)
    for candidate, ident in zip(candidates, ids):
        candidate["id"] = ident
    return candidates


def annotate(frame, candidates):
    image = Image.fromarray(frame.rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    for candidate in candidates:
        x0, y0, x1, y1 = candidate["bbox"]
        draw.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), outline="white", width=2)
        label = str(candidate["id"])
        text_y = max(0, y0 - 33)
        draw.rectangle((x0, text_y, x0 + 32, text_y + 30), fill="white")
        draw.text((x0 + 8, text_y + 2), label, fill="black", font=font)
    return image


def save_frame(directory, frame, candidates, prefix="initial"):
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / f"{prefix}-rgb.png"
    marked = directory / f"{prefix}-candidates.png"
    Image.fromarray(frame.rgb).save(raw)
    annotate(frame, candidates).save(marked)
    np.savez_compressed(directory / f"{prefix}-depth.npz", depth=frame.depth)
    (directory / f"{prefix}-observation.json").write_text(
        json.dumps(
            {
                "sensor": frame.metadata,
                "candidates": candidates,
                "note": "Only the annotated RGB image and instruction enter the VLM; this file includes controller calibration.",
            },
            indent=2,
        )
        + "\n"
    )
    return marked


def validate_answer(raw, candidates):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate VLM field")
            out[key] = value
        return out

    answer = json.loads(text, object_pairs_hook=unique)
    if not isinstance(answer, dict) or set(answer) != {
        "status",
        "target_id",
        "visibility",
        "reason",
    }:
        raise ValueError("invalid VLM schema")
    if answer["status"] not in ("ok", "observe", "clarify", "unsupported", "stop"):
        raise ValueError("invalid VLM status")
    if answer["visibility"] not in ("visible", "partial", "not_visible", "uncertain"):
        raise ValueError("invalid visibility")
    if not isinstance(answer["reason"], str) or len(answer["reason"]) > 200:
        raise ValueError("invalid reason")
    ident = answer["target_id"]
    if answer["status"] == "ok":
        if type(ident) is not int or ident not in {c["id"] for c in candidates}:
            raise ValueError("VLM selected nonexistent candidate")
        if answer["visibility"] != "visible":
            raise ValueError("non-visible targets cannot be executed")
    elif ident is not None:
        raise ValueError("non-executable answer must use null target")
    return answer


class VisionPlanner:
    def __init__(self):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.path = (
            ROOT
            / ".cache/vlm-hub"
            / ("models--" + MODEL.replace("/", "--"))
            / "snapshots"
            / REVISION
        )
        if not self.path.exists():
            raise FileNotFoundError("先执行 vision download 获取固定版本视觉模型")
        self.model, self.processor = load(str(self.path), trust_remote_code=False)
        self.config = load_config(str(self.path))

    def predict(self, instruction, image, candidates):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # A short visual question preserves spatial reading better than the long
        # action/schema prompt used in the initial failed experiment. The VLM only
        # selects a referent; controller actions are deliberately outside its job.
        prompt = (
            "Which ball does the user mean? Read its color and the number directly ABOVE it in the image. "
            "Left/right means image left/right, never number order. User: "
            + instruction
            + ". Return ONLY JSON with keys status (visible/partial/not_visible/ambiguous/stop/unsupported), "
            "color (red/blue/green or null), id (integer or null). "
            "If occluded, absent or ambiguous, use null color and id."
        )
        formatted = apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )
        started = time.perf_counter()
        output = generate(
            self.model,
            self.processor,
            formatted,
            image=[str(image)],
            max_tokens=160,
            temperature=0.0,
            verbose=False,
        )
        raw = output.text
        prediction = {
            "instruction": instruction,
            "raw": raw,
            "prompt": prompt,
            "latency_s": time.perf_counter() - started,
            "model": MODEL,
            "revision": REVISION,
            "fine_tuned": False,
            "input_image": str(image),
            "peak_memory_gb": getattr(output, "peak_memory", None),
            "prompt_tokens": output.prompt_tokens,
            "generation_tokens": output.generation_tokens,
        }
        try:
            answer, choice = parse_visual_choice(raw, candidates)
        except (ValueError, TypeError, KeyError) as error:
            raise VisualOutputError(str(error), prediction) from error
        prediction.update(answer=answer, visual_choice=choice)
        return prediction


def parse_visual_choice(raw, candidates):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate visual field")
            obj[key] = value
        return obj

    choice = json.loads(text, object_pairs_hook=unique)
    if not isinstance(choice, dict) or set(choice) != {"status", "color", "id"}:
        raise ValueError("invalid visual choice schema: " + raw)
    status = choice["status"]
    statuses = {
        "visible": ("ok", "visible"),
        "partial": ("observe", "partial"),
        "not_visible": ("observe", "not_visible"),
        "ambiguous": ("clarify", "uncertain"),
        "stop": ("stop", "uncertain"),
        "unsupported": ("unsupported", "uncertain"),
    }
    if status not in statuses:
        raise ValueError("invalid visual choice status")
    if status == "visible":
        selected = next(
            (
                c
                for c in candidates
                if type(choice["id"]) is int and c["id"] == choice["id"]
            ),
            None,
        )
        if selected is None or selected["color"] != choice["color"]:
            raise ValueError("VLM color/number binding disagrees with RGB proposals")
    action, visibility = statuses[status]
    answer = {
        "status": action,
        "target_id": choice["id"] if action == "ok" else None,
        "visibility": visibility,
        "reason": "VLM visual choice: " + str(choice["color"] or status),
    }
    return validate_answer(json.dumps(answer), candidates), choice


def semantic_guard(instruction, selected, candidates):
    """Reject (never replace) a VLM choice that violates supported spatial constraints.

    This small symbolic verifier is explicit; it is not counted as VLM accuracy.
    Only simple left/right ball references are covered, not arbitrary language.
    """
    text = instruction.lower()
    if re.search(
        r"(?:紫|黄|橙|白|黑|粉)(?:色)?球|(?:purple|yellow|orange|white|black|pink) ball",
        text,
    ):
        return {"safe": False, "status": "unsupported_ball_color"}
    if re.search(r"挡住|遮挡|后面的球|occluded|behind", text):
        return {"safe": False, "status": "occlusion_reference_requires_observation"}
    side = None
    if re.search(r"左边(?:的)?(?:红|蓝|绿)?球|最左(?:边|侧)?|leftmost|left side", text):
        side = "left"
    if re.search(
        r"右边(?:的)?(?:红|蓝|绿)?球|最右(?:边|侧)?|rightmost|right side", text
    ):
        if side is not None:
            return {"safe": False, "status": "ambiguous_side"}
        side = "right"
    color = None
    for name, aliases in {
        "red": ["红球", "红色", "red ball"],
        "blue": ["蓝球", "蓝色", "blue ball"],
        "green": ["绿球", "绿色", "green ball"],
    }.items():
        if any(a in text for a in aliases):
            if color is not None:
                return {"safe": False, "status": "ambiguous_color"}
            color = name
    pool = [c for c in candidates if color is None or c["color"] == color]
    if color is not None and selected["color"] != color:
        return {"safe": False, "status": "color_constraint_mismatch"}
    if side and pool:
        expected = (min if side == "left" else max)(pool, key=lambda c: c["pixel"][0])
        return {
            "safe": selected["id"] == expected["id"],
            "status": "spatial_constraint_checked",
            "side": side,
            "expected_id": expected["id"],
        }
    return {"safe": True, "status": "no_supported_spatial_constraint"}


def visibility_check(frame, target, color):
    candidates = propose_balls(frame)
    matching = [
        c
        for c in candidates
        if c["color"] == color
        and np.linalg.norm(np.asarray(c["world_position"]) - target) < 0.20
    ]
    if matching:
        candidate = min(
            matching,
            key=lambda c: np.linalg.norm(np.asarray(c["world_position"]) - target),
        )
        return {
            "safe": candidate["visibility"] == "visible",
            "status": candidate["visibility"],
            "visible_fraction_estimate": candidate["visible_fraction_estimate"],
        }
    pixel = project(target, frame.metadata)
    if pixel is None:
        return {"safe": False, "status": "out_of_view"}
    u, v, z = pixel
    if not (
        2 <= u < frame.metadata["width"] - 2 and 2 <= v < frame.metadata["height"] - 2
    ):
        return {"safe": False, "status": "out_of_view"}
    crop = frame.depth[int(v) - 2 : int(v) + 3, int(u) - 2 : int(u) + 3]
    actual = float(np.median(crop))
    blocked = math.isfinite(actual) and actual < z - BALL_RADIUS - 0.04
    return {
        "safe": False,
        "status": "occluded" if blocked else "missing",
        "expected_depth_m": z,
        "observed_depth_m": actual,
    }


class VisualGuard:
    def __init__(self, port, target, color, directory):
        self.port, self.target, self.color, self.directory = (
            port,
            np.asarray(target),
            color,
            directory,
        )
        self.last_time = 0.0
        self.last = {"safe": False, "status": "unchecked"}
        self.index = 0

    def __call__(self, state, goal):
        if time.monotonic() - self.last_time < 0.6:
            return self.last
        frame = capture(self.port)
        self.last = visibility_check(frame, self.target, self.color)
        self.last_time = time.monotonic()
        save_frame(
            self.directory, frame, propose_balls(frame), f"track-{self.index:04d}"
        )
        self.index += 1
        return self.last


def run_instruction(planner, instruction, info, preview=False, timeout=60):
    directory = ROOT / "runs/vision" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    result = {
        "instruction": instruction,
        "executed": False,
        "success": False,
        "directory": str(directory),
        "observation": "RGB-D target; oracle robot pose and safety map; RGB color proposals + VLM selection",
    }
    with ExitStack() as stack:
        lock = stack.enter_context((ROOT / "runs/navigation.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        robot = Robot(info["socket"])
        stack.callback(robot.close)
        robot.stop()  # VLM runs only while stationary; no stale velocity during inference.
        try:
            frame = capture(info["port"])
            candidates = propose_balls(frame)
            marked = save_frame(directory, frame, candidates)
            result.update(planner.predict(instruction, marked, candidates))
            print(json.dumps(result["answer"], ensure_ascii=False), flush=True)
            answer = result["answer"]
            if answer["status"] == "stop":
                result.update(success=True, stopped=True)
            elif answer["status"] != "ok":
                result["stopped_reason"] = answer["status"] + ": " + answer["reason"]
            else:
                selected = next(c for c in candidates if c["id"] == answer["target_id"])
                result["selected"] = selected
                result["semantic_check"] = semantic_guard(
                    instruction, selected, candidates
                )
                if not result["semantic_check"]["safe"]:
                    result["stopped_reason"] = (
                        "VLM selection violates image-space constraint"
                    )
                elif selected["visibility"] != "visible":
                    result["stopped_reason"] = (
                        "partial visibility: depth/RGB guard refuses movement"
                    )
                elif not preview:
                    # Refresh after slow inference; never execute a stale scene selection.
                    check = visibility_check(
                        capture(info["port"]),
                        np.asarray(selected["world_position"]),
                        selected["color"],
                    )
                    result["preflight_visibility"] = check
                    if not check["safe"]:
                        result["stopped_reason"] = check["status"]
                    else:
                        x, y, _ = selected["world_position"]
                        plan = {
                            "status": "ok",
                            "goals": [
                                {
                                    "target": "point",
                                    "x": x,
                                    "y": y,
                                    "speed": "slow",
                                    "stop_distance_m": 0.3,
                                }
                            ],
                        }
                        observer = Observer(info["port"])
                        stack.callback(observer.close)
                        scene = json.loads(Path(info["scene_file"]).read_text())
                        result["executed"] = True
                        result["execution"] = execute(
                            plan,
                            robot,
                            observer,
                            directory / "trajectory.jsonl",
                            timeout_s=timeout,
                            scene=scene,
                            cancel_file=ROOT / "runs/cancel_navigation",
                            guard=VisualGuard(
                                info["port"],
                                selected["world_position"],
                                selected["color"],
                                directory,
                            ),
                            observation_label=result["observation"],
                        )
                        result["success"] = result["execution"]["success"]
        except BaseException as error:
            if isinstance(error, VisualOutputError):
                result.update(error.prediction)
            result["error"] = str(error) or type(error).__name__
            raise
        finally:
            try:
                robot.stop()
            except (OSError, RuntimeError, ConnectionError) as error:
                result["stop_error"] = str(error)
            finally:
                (directory / "result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
                )
            print("记录：" + str(directory), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["download", "observe", "plan", "run", "chat", "stop"]
    )
    parser.add_argument("instruction", nargs="?")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            MODEL,
            revision=REVISION,
            cache_dir=ROOT / ".cache/vlm-hub",
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.jinja",
                "*.txt",
                "*.model",
                "README.md",
            ],
        )
        print(path)
        return
    info = json.loads((ROOT / "runs/vision-simulator.json").read_text())
    if not info.get("vision") or not Path(info["socket"]).exists():
        raise RuntimeError(
            "视觉实例未运行，请先执行 python -m microduck_qwen.sim --vision"
        )
    if args.command == "stop":
        (ROOT / "runs/cancel_navigation").write_text(str(time.time_ns()))
        robot = Robot(info["socket"])
        try:
            robot.stop()
        finally:
            robot.close()
        return
    if args.command == "observe":
        frame = capture(info["port"])
        candidates = propose_balls(frame)
        directory = ROOT / "runs/vision" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        print(save_frame(directory, frame, candidates))
        print(json.dumps(candidates, ensure_ascii=False))
        return
    if args.command in ("plan", "run") and not args.instruction:
        parser.error("需要提供自然语言指令")
    planner = VisionPlanner()
    if args.command == "chat":
        print("第一视角视觉导航：输入指令，exit退出；Ctrl+C停止。")
        while True:
            try:
                instruction = input("你 > ").strip()
                if instruction in ("exit", "quit", "退出"):
                    break
                if instruction:
                    print(
                        json.dumps(
                            run_instruction(
                                planner, instruction, info, timeout=args.timeout
                            ),
                            ensure_ascii=False,
                        )
                    )
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as error:
                print("未执行：" + str(error), flush=True)
    else:
        print(
            json.dumps(
                run_instruction(
                    planner,
                    args.instruction,
                    info,
                    preview=args.command == "plan",
                    timeout=args.timeout,
                ),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
