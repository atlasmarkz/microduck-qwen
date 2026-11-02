"""Frontier exploration on a depth-built map: the `explore` skill and long free exploration.

Where to go comes from a depth-built occupancy map (mapping.py), and navigation plans on that
perceived map only, so unseen walls are never assumed. Pose and extrinsics stay simulator
oracle, and the executor's own fall / clearance / bounds stops (which read the known scene)
remain a safety layer; they are not used to choose frontiers or routes.
"""

import math
import time

import numpy as np

from .mapping import INFLATE_M, OccupancyMap
from .navigation import execute
from .vision import approachable, capture, propose_targets

MAX_ITERATIONS = 6
BUDGET_S = 240.0
CHUNK = 3  # waypoints per navigation call (~1.5 m), then re-look
SAME_BALL_M = 0.35
TIGHT_INFLATE_M = 0.25  # last resort when no frontier is reachable at the normal clearance (doorways are 1.1 m)


def visible_targets(frame, color):
    # Fully visible only: the follow-up approach must find the same ball confidently in a fresh frame.
    return [c for c in propose_targets(frame) if c["color"] == color and c.get("kind", "ball") == "ball"
            and c["visibility"] == "visible" and approachable(c)]


def remember(inventory, candidates):
    """Add newly perceived fully visible balls; one entry per physical ball (colour + position)."""
    for c in candidates:
        if c.get("kind", "ball") != "ball" or c["visibility"] != "visible":
            continue
        x, y = c["world_position"][:2]
        if not any(i["color"] == c["color"] and math.hypot(i["x"] - x, i["y"] - y) < SAME_BALL_M for i in inventory):
            inventory.append({"color": c["color"], "x": float(x), "y": float(y)})


class Explorer:
    def __init__(self, executor, output):
        self.executor, self.output = executor, output
        self.mapper = OccupancyMap()
        self.inventory = []
        self.avoid = []
        self.iterations = []
        self.turn_trace = output.with_suffix(".turn.jsonl")
        self.port = executor.info["port"]

    def look(self, color=None):
        time.sleep(0.3)  # let the 5 Hz camera deliver a frame taken after the body settled
        frame = capture(self.port)
        self.mapper.update_from_frame(frame)
        candidates = propose_targets(frame)
        remember(self.inventory, candidates)
        return visible_targets(frame, color) if color else []

    def scan(self, color=None):
        """Four headings 90 degrees apart (the camera is ~97 degrees wide).

        With a colour, stop as soon as it is in view; without one, always complete the turn.
        """
        for k in range(4):
            seen = self.look(color)
            if seen:
                return seen
            if k < 3:
                with self.turn_trace.open("a") as trace:
                    turned = self.executor._bounded({"action": "turn", "direction": "left", "degrees": 90}, trace)
                if not turned["success"]:
                    raise RuntimeError("explore scan turn failed")
        return []

    def step(self, index):
        """Walk to the nearest reachable frontier. Returns the record, or None if none is left."""
        position = self.executor.healthy()["trunk"][:2]
        # Normal clearance first; then forget past failures; then accept tighter clearance.
        for inflate, avoid in ((INFLATE_M, self.avoid), (INFLATE_M, ()), (TIGHT_INFLATE_M, ())):
            path = self.mapper.plan_to_frontier(position, avoid, inflate_m=inflate)
            if path is not None:
                break
        else:
            return None
        goal = path[-1]
        record = {"from": [round(float(v), 2) for v in position],
                  "frontier": [round(float(goal[0]), 2), round(float(goal[1]), 2)], "waypoints": len(path)}
        self.iterations.append(record)
        try:
            for start in range(0, len(path), CHUNK):
                chunk = path[start:start + CHUNK]
                plan = {"status": "ok", "goals": [{"target": "point", "x": float(x), "y": float(y), "speed": "slow",
                                                  "stop_distance_m": 0.10} for x, y in chunk]}
                execute(plan, self.executor.robot, self.executor.observer, self.output.with_suffix(f".nav{index}-{start}.jsonl"),
                        timeout_s=30, scene=self.mapper.as_scene(inflate), cancel_file=self.executor.cancel_file,
                        cancel_started_ns=self.executor.started_ns,
                        observation_label="depth-built occupancy map; oracle pose")
                self.look()  # new sight lines open as the body moves
            record["reached"] = True
        except (ValueError, TimeoutError) as error:
            record.update(reached=False, error=f"{type(error).__name__}: {error}", failed_chunk=start,
                          chunk=[[round(float(x), 2), round(float(y), 2)] for x, y in chunk])
        self.avoid.append((float(goal[0]), float(goal[1])))
        self.executor.robot.stop()
        return record

    def finish(self, started):
        self.executor.robot.stop()
        np.save(self.output.with_suffix(".map.npy"), self.mapper.grid)
        return {"explored_cells": self.mapper.explored_cells(), "elapsed_s": time.monotonic() - started,
                "iterations": self.iterations, "inventory": self.inventory}


def run_explore(executor, action, output):
    """The model-facing skill: explore until a fully visible ball of the colour is in view."""
    color = action["color"]
    explorer = Explorer(executor, output)
    started = time.monotonic()
    result = {"found": False, "color": color, "map": "depth-built occupancy; oracle pose"}
    try:
        seen = explorer.scan(color)
        for index in range(MAX_ITERATIONS):
            if seen or time.monotonic() - started > BUDGET_S:
                break
            if explorer.step(index) is None:
                result["stopped"] = "no reachable frontier left"
                break
            seen = explorer.scan(color)
        result["found"] = bool(seen)
        if seen:
            result["candidate_colors"] = sorted({c["color"] for c in seen})
        elif "stopped" not in result:
            result["stopped"] = "iteration/time budget used"
    finally:
        result.update(explorer.finish(started))
    result["success"] = result["found"]
    if not result["found"]:
        result["error"] = "target colour not found while exploring"
    return result


def free_explore(executor, minutes, output):
    """Unprompted exploration for a fixed time: cover as much of the world as possible.

    Returns the perceived inventory (colour and position of every ball seen) and map coverage.
    No target and no model: this is the robot's own curiosity phase.
    """
    explorer = Explorer(executor, output)
    started = time.monotonic()
    result = {"minutes": minutes, "map": "depth-built occupancy; oracle pose"}
    try:
        explorer.scan()
        index = 0
        while time.monotonic() - started < minutes * 60:
            if explorer.step(index) is None:
                result["stopped"] = "no reachable frontier left"
                break
            explorer.scan()
            index += 1
        else:
            result["stopped"] = "time budget used"
    finally:
        result.update(explorer.finish(started))
    return result
