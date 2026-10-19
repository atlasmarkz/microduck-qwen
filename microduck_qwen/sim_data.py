"""Shared MuJoCo RGB-D observation helpers with simulator-verified labels.

Digest, JSON IPC, collection-frame decoding and object verification used by
every collector. Privileged state stays in episode sidecars, never in model
messages.
"""

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .navigation import JsonConnection
from .vision import Frame, decode_array, project


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
