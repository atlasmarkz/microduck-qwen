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
  * Settling on the target before the last leg: face the detected ball and confirm it in a fresh frame, in the spirit of
    detection-driven goal refinement with visual servoing for quadrupeds (CVPR Workshops 2026) and of MG-Nav's
    re-localisation before the final approach (arXiv:2511.22609).
"""

import collections
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
SUSPECT_STOP_M = 0.6  # the contract caps a point goal's stop distance at 0.6 m; 0.9 here was rejected, so no suspect was ever walked to
NAV_SLACK_M = 0.05  # the navigator's 5 cm lattice sits half a cell off the map's, which costs it up to this much width
TIGHT_INFLATE_M = 0.26
TIGHTEST_INFLATE_M = 0.22  # only when nothing else is reachable; the navigator then keeps just 3 cm of pad  # last resort when no frontier is reachable at the normal clearance (balls beside a doorway need it)
# Narrowest tier, tried only when every tier above found no reachable frontier. The robot's collision geoms reach about
# 0.10 m from the trunk at rest while the navigator assumes 0.14 m plus a 5 cm map cell, so 0.20 m still leaves a margin
# over the geometry; the navigator keeps only 1 cm of pad here. A clearance stop on such a route is a failed frontier
# (caught in _walk), not a failed search. MICRODUCK_NARROW=0 removes the tier (ablation).
NARROW = os.environ.get("MICRODUCK_NARROW", "1") != "0"
NARROW_INFLATE_M = 0.20
NARROW_PAD_M = 0.01


# The navigator's guard measures clearance on the perceived map, the default escape on the true scene. When a map cell
# sits within the guard of the robot while the true clearance is fine, every walk stops on its first in-place turn and
# the escape never moves: one evaluated layout spent 8 of its 12 iterations so (99104). MICRODUCK_MAP_ESCAPE=1 also
# backs away from the perceived map before each plan and after each failed walk.
MAP_ESCAPE = os.environ.get("MICRODUCK_MAP_ESCAPE", "0") != "0"

FRONTIER_TRIES = 2  # a frontier that failed this many walks is never chosen again, however far the tiers are relaxed


def clearance_tiers(avoid, exhausted=()):
    """(inflation, frontiers to avoid) pairs in the order they are tried: normal, forget failures, tighter, tightest, narrowest.

    "Forget failures" forgets the soft avoid list, not the `exhausted` frontiers: one evaluated layout retried the same
    frontier twelve times (the clearance guard stopped every walk at its first step) and used up the whole budget on it.
    """
    exhausted = tuple(exhausted)
    tiers = [(INFLATE_M, tuple(avoid) + exhausted), (INFLATE_M, exhausted), (TIGHT_INFLATE_M, exhausted),
             (TIGHTEST_INFLATE_M, exhausted)]
    return tiers + [(NARROW_INFLATE_M, exhausted)] if NARROW else tiers


# A ball found at the edge of the view is only partly in the frame, and the next frame may not show it as fully
# visible (two evaluated tasks ended exactly so: explore reported the colour found, then no fully visible ball
# was left to approach). Before returning success, explore turns to face the ball and requires a fresh frame in
# which it is still fully visible and clear of the image border. MICRODUCK_SETTLE=0 removes the step (ablation).
SETTLE = os.environ.get("MICRODUCK_SETTLE", "1") != "0"
SETTLE_ATTEMPTS = 2
SETTLE_MIN_TURN_DEG = 8.0
EDGE_MARGIN_PX = 50


def navigator_pad(inflate):
    """Least path margin the navigator keeps around perceived obstacles for a route planned at this clearance."""
    if inflate <= NARROW_INFLATE_M:
        return NARROW_PAD_M
    return 0.03 if inflate <= TIGHTEST_INFLATE_M else MIN_PAD_M


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
        self.failed_walks = collections.Counter()   # rounded frontier -> walks to it that failed
        self.suspects = []   # target-coloured balls seen only partly: where to look again (ApexNav's "suspected targets")
        self.checked = []
        self.iterations = []
        self.turn_trace = output.with_suffix(".turn.jsonl")
        self.port = executor.info["port"]

    def look(self, color=None):
        time.sleep(0.3)  # let the 5 Hz camera deliver a frame taken after the body settled
        frame = capture(self.port)
        self.frame_width = int(frame.rgb.shape[1])
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
            if seen and SETTLE:
                seen = self.settle(color, seen)  # [] when the ball cannot be confirmed: keep turning and exploring
            if seen:
                return seen
            if k < 3:
                with self.turn_trace.open("a") as trace:
                    turned = self.executor._bounded({"action": "turn", "direction": "left", "degrees": 90}, trace)
                if not turned["success"]:
                    raise RuntimeError("explore scan turn failed")
        return []

    def settle(self, color, seen):
        """Face the nearest fully visible ball of `color` and confirm it in a fresh frame, clear of the image edge.

        Returns the fresh targets, or [] when the ball could not be confirmed; its position is then kept as a
        suspect so the search walks closer later instead of giving up on it. A safety stop of the body (clearance
        too small next to a wall, say) is not a reason to lose a ball already found: the unsettled sighting is returned.
        Cancel, a fall and a stalled simulator still end the search.
        """
        try:
            return self._settle(color, seen)
        except RuntimeError as error:
            if any(w in str(error) for w in ("cancelled", "fallen", "stale")):
                raise
            return seen

    def _settle(self, color, seen):
        for _ in range(SETTLE_ATTEMPTS):
            state = self.executor.healthy()
            x, y = state["trunk"][:2]
            target = self._nearest(seen, x, y)
            if self._clear_of_edge(target):
                return seen
            bearing = math.degrees(math.atan2(target["world_position"][1] - y, target["world_position"][0] - x) - state["yaw"])
            bearing = (bearing + 180.0) % 360.0 - 180.0
            if abs(bearing) >= SETTLE_MIN_TURN_DEG:
                turn = {"action": "turn", "direction": "left" if bearing > 0 else "right",
                        "degrees": float(min(120.0, max(5.0, round(abs(bearing)))))}
                try:
                    with self.turn_trace.open("a") as trace:
                        self.executor._bounded(turn, trace)
                except (TimeoutError, RuntimeError):
                    return seen  # could not turn: keep what was found rather than lose it
            previous, seen = seen, self.look(color)
            if not seen:
                return self._unconfirmed(color, previous)  # it left the view: remember where it was
        x, y = self.executor.healthy()["trunk"][:2]
        return seen if self._clear_of_edge(self._nearest(seen, x, y)) else self._unconfirmed(color, seen)

    @staticmethod
    def _nearest(seen, x, y):
        return min(seen, key=lambda c: math.hypot(c["world_position"][0] - x, c["world_position"][1] - y))

    def _clear_of_edge(self, candidate):
        x0, _, x1, _ = candidate["bbox"]
        return x0 >= EDGE_MARGIN_PX and x1 <= getattr(self, "frame_width", 640) - EDGE_MARGIN_PX

    def _unconfirmed(self, color, seen):
        for c in seen:
            x, y = c["world_position"][:2]
            if not any(math.hypot(x - sx, y - sy) < SAME_BALL_M for sx, sy in self.suspects + self.checked):
                self.suspects.append((float(x), float(y)))
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
            plan = {"status": "ok", "goals": [{"target": "point", "x": x, "y": y, "speed": "slow", "stop_distance_m": SUSPECT_STOP_M}]}
            execute(plan, self.executor.robot, self.executor.observer, self.output.with_suffix(f".suspect{len(self.checked)}.jsonl"),
                    timeout_s=40, scene=self.mapper.as_scene(), cancel_file=self.executor.cancel_file,
                    cancel_started_ns=self.executor.started_ns, observation_label="depth-built occupancy map; oracle pose")
            record["reached"] = True
        except (ValueError, TimeoutError) as error:
            record.update(reached=False, error=f"{type(error).__name__}: {error}")
        self.executor.robot.stop()
        return self.scan(color)

    def unpin(self, record=None):
        """Back away from the true scene and, with MAP_ESCAPE, from the perceived map the navigator guards against."""
        self.executor.escape_cramped()
        if MAP_ESCAPE:
            moved = self.executor.escape_cramped(scene=self.mapper.as_scene())
            if moved and record is not None:
                record["map_escape"] = True
            return moved
        return False

    def _walk(self, path, inflate, index, record, tag=""):
        """Walk a planned path in chunks, re-looking after each; records the outcome in `record`."""
        try:
            for start in range(0, len(path), CHUNK):
                chunk = path[start:start + CHUNK]
                plan = {"status": "ok", "goals": [{"target": "point", "x": float(x), "y": float(y), "speed": "slow",
                                                  "stop_distance_m": 0.10} for x, y in chunk]}
                execute(plan, self.executor.robot, self.executor.observer, self.output.with_suffix(f".nav{tag}{index}-{start}.jsonl"),
                        timeout_s=30, scene=self.mapper.as_scene(inflate - NAV_SLACK_M, navigator_pad(inflate)), cancel_file=self.executor.cancel_file,
                        cancel_started_ns=self.executor.started_ns,
                        observation_label="depth-built occupancy map; oracle pose")
                self.look()  # new sight lines open as the body moves
            record["reached"] = True
        except (ValueError, TimeoutError, RuntimeError) as error:
            # The depth map can under-draw a wall, so the true-scene safety stop may fire mid-route: that
            # frontier failed, not the whole search. Cancel, a fall and a stalled simulator still end it.
            if isinstance(error, RuntimeError) and any(w in str(error) for w in ("cancelled", "fallen", "stale")):
                raise
            if isinstance(error, RuntimeError) and "净空" in str(error):
                # The route went closer to something than the guard allows: stop giving balls the relief that made such
                # routes look possible, so the next plan goes around them.
                self.mapper.harden_balls()
            record.update(reached=False, error=f"{type(error).__name__}: {error}", failed_chunk=start,
                          chunk=[[round(float(x), 2), round(float(y), 2)] for x, y in chunk])
        self.executor.robot.stop()
        if not record.get("reached", True):
            self.unpin(record)

    def step(self, index):
        """Walk to the nearest reachable frontier. Returns the record, or None if none is left."""
        self.unpin()
        position = self.executor.healthy()["trunk"][:2]
        # Normal clearance first; then forget past failures; then accept tighter clearance.
        exhausted = [key for key, count in self.failed_walks.items() if count >= FRONTIER_TRIES]
        for inflate, avoid in clearance_tiers(self.avoid, exhausted):
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
        if not record.get("reached", True):
            self.failed_walks[(round(float(goal[0]), 1), round(float(goal[1]), 1))] += 1
        return record

    def recover(self, index):
        """No frontier is reachable: stand as close to the nearest one as the map allows, then the caller looks again.

        Returns False when there is nowhere better to stand (the failure is then final).
        """
        self.unpin()
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
