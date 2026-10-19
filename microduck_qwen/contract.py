"""Strict navigation-plan contract shared by the controller and the executor."""

import json
import math
from pathlib import Path

ARENA_FILE = Path(__file__).with_name("arena.json")


def arena():
    return json.loads(ARENA_FILE.read_text(encoding="utf-8"))


def _number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} 必须在 {low}..{high} 范围内")
    return float(value)


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"重复的 JSON 字段: {key}")
        out[key] = value
    return out


def validate(raw, scene=None):
    if isinstance(raw, str):
        raw = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(raw, dict) or set(raw) != {"status", "goals"}:
        raise ValueError("输出必须只有 status 和 goals")
    status, goals = raw["status"], raw["goals"]
    if status not in ("ok", "stop", "unsupported") or not isinstance(goals, list):
        raise ValueError("status/goals 格式不正确")
    if status != "ok":
        if goals:
            raise ValueError("停止/不支持的指令不能包含目标")
        return raw
    if not 1 <= len(goals) <= 3:
        raise ValueError("每条指令必须包含 1..3 个目标")
    scene = scene or arena()
    xmin, xmax, ymin, ymax = scene["bounds"]
    for goal in goals:
        if not isinstance(goal, dict):
            raise ValueError("目标必须是对象")
        target = goal.get("target")
        keys = {"target", "speed", "stop_distance_m"}
        if target == "point":
            keys |= {"x", "y"}
        elif target not in scene["objects"]:
            raise ValueError(f"场景不存在目标 {target!r}")
        if set(goal) != keys or goal["speed"] not in ("slow", "normal", "fast"):
            raise ValueError("目标字段或速度不符合约定")
        _number(goal["stop_distance_m"], 0.1 if target == "point" else 0.25, 0.6, "停靠距离")
        if target == "point":
            _number(goal["x"], xmin, xmax, "x")
            _number(goal["y"], ymin, ymax, "y")
    return raw
