"""RGB-D capture, calibrated ball/pillar proposals and visibility guarding.

Shared by every VLM link. This is a calibrated known-shape prototype, NOT general
object detection, SLAM or a fine-tuned VLA. Ground-truth scene labels never enter
the VLM prompt.
"""

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import ROOT
from .navigation import JsonConnection, Robot

MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
REVISION = "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b"
BALL_RADIUS = 0.07
MIN_VISIBILITY = 0.65
# A partially occluded target is still approachable above this fraction: the
# blob is big enough that its world position and its numbered box are sound.
APPROACH_VISIBILITY_FLOOR = 0.35
# Consecutive out-of-frame reads tolerated while the body re-aims at a bound
# target. The target's world position is static and was confirmed at bind time,
# so a brief loss of sight is not evidence that it moved. Only applied to
# "out_of_view": "occluded" and "missing" still abort immediately.
OUT_OF_VIEW_GRACE = 3


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


def propose_targets(frame):
    """Number calibrated balls and yellow 10 cm-radius, 30 cm-high pillars together."""
    from .bypass import yellow_detections

    candidates = propose_balls(frame)
    for detection in yellow_detections(frame):
        surface = np.asarray(detection["surface_position"])
        if not 0.04 < surface[2] < 0.35:
            continue  # Reflections on the floor are not pillars.
        ray = surface[:2] - np.asarray(frame.metadata["camera_position"][:2])
        length = float(np.linalg.norm(ray))
        if length < 0.06:
            continue
        center = surface.copy()
        center[:2] += 0.10 * ray / length
        center[2] = 0.15
        expected = frame.metadata["fx"] * frame.metadata["fy"] * 0.2 * 0.3 / detection["depth"] ** 2
        fraction = min(1.0, detection["pixels"] / max(1.0, expected))
        candidates.append({"color": "yellow", "kind": "pillar", "bbox": detection["bbox"],
            "pixel": detection["pixel"], "world_position": center.tolist(),
            "surface_depth_m": detection["depth"], "visible_fraction_estimate": fraction,
            "visibility": "visible" if fraction >= MIN_VISIBILITY and detection["valid_depth_fraction"] >= 0.8 else "partial"})
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


def approachable(candidate):
    """A bound target stays approachable while enough of its blob is segmentable.

    Fully visible blobs always qualify. A partially occluded blob still gives a
    trustworthy world position as long as it is above the floor, which is what
    the duplicate-colour row needs: the numbered box is visible, so the binding
    the model chose is still the binding it can be held to. Below the floor the
    target is treated as not localisable and the caller must refuse or search.
    """
    if candidate.get("color") == "yellow":
        return candidate["visibility"] == "visible"
    return candidate["visibility"] == "visible" or (
        candidate["visibility"] == "partial"
        and candidate["visible_fraction_estimate"] >= APPROACH_VISIBILITY_FLOOR
    )


def visibility_check(frame, target, color):
    candidates = propose_targets(frame) if color == "yellow" else propose_balls(frame)
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
            "safe": approachable(candidate),
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
    radius = 0.10 if color == "yellow" else BALL_RADIUS
    blocked = math.isfinite(actual) and actual < z - radius - 0.04
    return {
        "safe": False,
        "status": "occluded" if blocked else "missing",
        "expected_depth_m": z,
        "observed_depth_m": actual,
    }


def apply_grace(check, lost):
    """Tolerate a bounded run of out-of-frame reads while the body re-aims.

    Returns the (possibly tolerated) check and the updated consecutive count.
    Any other verdict — including "occluded" and "missing" — resets the run.
    """
    if check.get("status") != "out_of_view":
        return check, 0
    lost += 1
    if lost > OUT_OF_VIEW_GRACE:
        return check, lost
    return dict(check, safe=True, status="out_of_view_tolerated", tolerated_checks=lost), lost


class VisualGuard:
    """Bounds the last approach by the live visibility of the bound object."""

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
        self.lost = 0

    def suspend(self):
        """A detour leg is driven by the planned route, so the sight line cannot gate it.

        Dropping the cached verdict makes the first aimed leg re-capture instead
        of inheriting a verdict taken while the object was legitimately hidden.
        """
        self.last = {"safe": False, "status": "suspended"}
        self.last_time = 0.0
        self.lost = 0

    def __call__(self, state, goal):
        if time.monotonic() - self.last_time < 0.6:
            return self.last
        frame = capture(self.port)
        self.last, self.lost = apply_grace(
            visibility_check(frame, self.target, self.color), self.lost
        )
        self.last_time = time.monotonic()
        save_frame(
            self.directory, frame, propose_targets(frame), f"track-{self.index:04d}"
        )
        self.index += 1
        return self.last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "observe", "stop"])
    parser.add_argument("--manifest", type=Path, help="explicit isolated simulator metadata")
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download
        if os.name == "nt":
            from .vision_torch import MODEL as torch_model, REVISION as torch_revision

            print(snapshot_download(torch_model, revision=torch_revision, cache_dir=ROOT / ".cache/vlm-torch-hub",
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"]))
            return

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
    info = json.loads((args.manifest or ROOT / "runs/vision-simulator.json").read_text())
    if not info.get("vision") or (not isinstance(info["socket"], list) and not Path(info["socket"]).exists()):
        raise RuntimeError(
            "视觉实例未运行，请先执行 python -m microduck_qwen.sim_native --vision"
        )
    if args.command == "stop":
        (ROOT / "runs/cancel_navigation").write_text(str(time.time_ns()))
        robot = Robot(info["socket"])
        try:
            robot.stop()
        finally:
            robot.close()
        return
    frame = capture(info["port"])
    candidates = propose_targets(frame)
    directory = ROOT / "runs/vision" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    print(save_frame(directory, frame, candidates))
    print(json.dumps(candidates, ensure_ascii=False))


if __name__ == "__main__":
    main()
