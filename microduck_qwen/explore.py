"""Frontier exploration on a depth-built map: the `explore` skill and long free exploration.

Where to go comes from a depth-built occupancy map (mapping.py), and navigation plans on that
perceived map only, so unseen walls are never assumed. Pose and extrinsics stay simulator
oracle, and the executor's own fall / clearance / bounds stops (which read the known scene)
remain a safety layer; they are not used to choose frontiers or routes.

Ideas taken from the literature (abstract-level reading, adapted to this simulator, not reimplemented):
  * Frontier-based exploration of an occupancy map with a value per frontier: VLFM (Yokoyama et al., 2023,
    arXiv:2312.03275); ESC (Zhou et al., 2023, arXiv:2301.13166).
  * "Suspected targets": partly seen balls of the target colour are revisited (ApexNav, Zhang et al., 2025).
  * Recovery when no frontier is reachable, by re-observing from the closest reachable vantage point instead of
    giving up: SEER (Safe Efficient Exploration for Aerial Robots, arXiv:2209.11034) keeps a detected door "to be
    confirmed" until the robot has been there; mode-switching frontier search (KAIST SGVR, RiTA 2025) and
    AERR-Nav switch to a recovery mode when the trajectory history shows a deadlock.
"""

import math
import os
import time

import numpy as np

from .mapping import INFLATE_M, MIN_PAD_M, OccupancyMap
from .navigation import execute
from .vision import approachable, capture, propose_targets

MAX_ITERATIONS = 12
RECOVERY = os.environ.get("MICRODUCK_RECOVERY", "1") != "0"  # MICRODUCK_RECOVERY=0 switches the recoveries off (ablation)
MAX_RECOVERIES = 2 if RECOVERY else 0  # "nothing reachable" re-observations per explore call
BUDGET_S = 300.0
CHUNK = 3  # waypoints per navigation call (~1.5 m), then re-look
SAME_BALL_M = 0.35
NAV_SLACK_M = 0.05  # the navigator's 5 cm lattice sits half a cell off the map's, which costs it up to this much width
TIGHT_INFLATE_M = 0.26
TIGHTEST_INFLATE_M = 0.22  # only when nothing else is reachable; the navigator then keeps just 3 cm of pad  # last resort when no frontier is reachable at the normal clearance (balls beside a doorway need it)


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
        self.suspects = []   # target-coloured balls seen only partly: where to look again (ApexNav's "suspected targets")
        self.checked = []
        self.iterations = []
        self.turn_trace = output.with_suffix(".turn.jsonl")
        self.port = executor.info["port"]

    def look(self, color=None):
        time.sleep(0.3)  # let the 5 Hz camera deliver a frame taken after the body settled
        frame = capture(self.port)
        self.mapper.update_from_frame(frame)
        candidates = propose_targets(frame)
        remember(self.inventory, candidates)
        self.mapper.mark_balls([c["world_position"][:2] for c in candidates if c.get("kind", "ball") == "ball"])
        for c in candidates:
            if color and c["color"] == color and c.get("kind", "ball") == "ball" and c["visibility"] != "visible":
                x, y = c["world_position"][:2]
                if not any(math.hypot(x - sx, y - sy) < SAME_BALL_M for sx, sy in self.suspects + self.checked):
                    self.suspects.append((float(x), float(y)))
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

    def check_suspect(self, color):
        """Walk toward the nearest partly-seen ball of the target colour and look again. [] if none is left."""
        if not self.suspects:
            return []
        position = self.executor.healthy()["trunk"][:2]
        x, y = min(self.suspects, key=lambda p: math.hypot(p[0] - position[0], p[1] - position[1]))
        self.suspects.remove((x, y))
        self.checked.append((x, y))
        record = {"suspect": [round(x, 2), round(y, 2)]}
        self.iterations.append(record)
        try:
            plan = {"status": "ok", "goals": [{"target": "point", "x": x, "y": y, "speed": "slow", "stop_distance_m": 0.9}]}
            execute(plan, self.executor.robot, self.executor.observer, self.output.with_suffix(f".suspect{len(self.checked)}.jsonl"),
                    timeout_s=40, scene=self.mapper.as_scene(), cancel_file=self.executor.cancel_file,
                    cancel_started_ns=self.executor.started_ns, observation_label="depth-built occupancy map; oracle pose")
            record["reached"] = True
        except (ValueError, TimeoutError) as error:
            record.update(reached=False, error=f"{type(error).__name__}: {error}")
        self.executor.robot.stop()
        return self.scan(color)

    def _walk(self, path, inflate, index, record, tag=""):
        """Walk a planned path in chunks, re-looking after each; records the outcome in `record`."""
        try:
            for start in range(0, len(path), CHUNK):
                chunk = path[start:start + CHUNK]
                plan = {"status": "ok", "goals": [{"target": "point", "x": float(x), "y": float(y), "speed": "slow",
                                                  "stop_distance_m": 0.10} for x, y in chunk]}
                execute(plan, self.executor.robot, self.executor.observer, self.output.with_suffix(f".nav{tag}{index}-{start}.jsonl"),
                        timeout_s=30, scene=self.mapper.as_scene(inflate - NAV_SLACK_M, 0.03 if inflate <= TIGHTEST_INFLATE_M else MIN_PAD_M), cancel_file=self.executor.cancel_file,
                        cancel_started_ns=self.executor.started_ns,
                        observation_label="depth-built occupancy map; oracle pose")
                self.look()  # new sight lines open as the body moves
            record["reached"] = True
        except (ValueError, TimeoutError, RuntimeError) as error:
            # The depth map can under-draw a wall, so the true-scene safety stop may fire mid-route: that
            # frontier failed, not the whole search. Cancel, a fall and a stalled simulator still end it.
            if isinstance(error, RuntimeError) and any(w in str(error) for w in ("cancelled", "fallen", "stale")):
                raise
            record.update(reached=False, error=f"{type(error).__name__}: {error}", failed_chunk=start,
                          chunk=[[round(float(x), 2), round(float(y), 2)] for x, y in chunk])
        self.executor.robot.stop()
        if not record.get("reached", True):
            self.executor.escape_cramped()

    def step(self, index):
        """Walk to the nearest reachable frontier. Returns the record, or None if none is left."""
        self.executor.escape_cramped()
        position = self.executor.healthy()["trunk"][:2]
        # Normal clearance first; then forget past failures; then accept tighter clearance.
        for inflate, avoid in ((INFLATE_M, self.avoid), (INFLATE_M, ()), (TIGHT_INFLATE_M, ()), (TIGHTEST_INFLATE_M, ())):
            path = self.mapper.plan_to_frontier(position, avoid, inflate_m=inflate)
            if path is not None:
                break
        else:
            return None
        goal = path[-1]
        record = {"from": [round(float(v), 2) for v in position],
                  "frontier": [round(float(goal[0]), 2), round(float(goal[1]), 2)], "waypoints": len(path)}
        self.iterations.append(record)
        self._walk(path, inflate, index, record)
        self.avoid.append((float(goal[0]), float(goal[1])))
        return record

    def recover(self, index):
        """No frontier is reachable: stand as close to the nearest one as the map allows, then the caller looks again.

        Returns False when there is nowhere better to stand (the failure is then final).
        """
        self.executor.escape_cramped()
        position = self.executor.healthy()["trunk"][:2]
        for inflate in (INFLATE_M, TIGHT_INFLATE_M):
            targets = self.mapper.vantage_targets(position, inflate)
            if targets is None:
                continue
            path = self.mapper.plan_to_frontier(position, (), min_distance=0.3, inflate_m=inflate, targets_override=targets)
            if path:
                break
        else:
            return False
        goal = path[-1]
        record = {"recover": True, "from": [round(float(v), 2) for v in position],
                  "vantage": [round(float(goal[0]), 2), round(float(goal[1]), 2)], "waypoints": len(path)}
        self.iterations.append(record)
        self._walk(path, inflate, index, record, tag="v")
        return True

    def finish(self, started):
        self.executor.robot.stop()
        np.save(self.output.with_suffix(".map.npy"), self.mapper.grid)
        np.save(self.output.with_suffix(".soft.npy"), self.mapper.soft)
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
        recoveries = 0
        for index in range(MAX_ITERATIONS):
            if seen or time.monotonic() - started > BUDGET_S:
                break
            seen = explorer.check_suspect(color)
            if seen:
                break
            if explorer.step(index) is None:
                if recoveries < MAX_RECOVERIES and explorer.recover(index):
                    recoveries += 1
                    seen = explorer.scan(color)
                    continue
                result["stopped"] = "no reachable frontier left"
                break
            seen = explorer.scan(color)
        while not seen and explorer.suspects and time.monotonic() - started <= BUDGET_S:
            seen = explorer.check_suspect(color)
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
