"""`explore`: look around, move to the nearest reachable frontier, repeat until a colour is seen.

Where to go comes from a depth-built occupancy map (mapping.py), and navigation plans on that
perceived map only, so unseen walls are never assumed. Pose and extrinsics stay simulator
oracle, and the executor's own fall / clearance / bounds stops (which read the known scene)
remain a safety layer; they are not used to choose frontiers or routes.
"""

import math
import time

import numpy as np

from .mapping import OccupancyMap
from .navigation import execute
from .vision import approachable, capture, propose_targets

MAX_ITERATIONS = 6
BUDGET_S = 240.0
CHUNK = 3  # waypoints per navigation call (~1.5 m), then re-look


def visible_targets(frame, color):
    # Fully visible only: the follow-up approach must find the same ball confidently in a fresh frame.
    return [c for c in propose_targets(frame) if c["color"] == color and c.get("kind", "ball") == "ball"
            and c["visibility"] == "visible" and approachable(c)]


def run_explore(executor, action, output):
    color = action["color"]
    mapper = OccupancyMap()
    started = time.monotonic()
    result = {"found": False, "iterations": [], "color": color, "map": "depth-built occupancy; oracle pose"}
    port = executor.info["port"]
    turn_trace = output.with_suffix(".turn.jsonl")

    def look():
        time.sleep(0.3)  # let the 5 Hz camera deliver a frame taken after the body settled
        frame = capture(port)
        mapper.update_from_frame(frame)
        return visible_targets(frame, color)

    def scan():
        """Four headings 90 degrees apart (the camera is ~97 degrees wide); stop as soon as seen."""
        for k in range(4):
            seen = look()
            if seen:
                return seen
            if k < 3:
                with turn_trace.open("a") as trace:
                    turned = executor._bounded({"action": "turn", "direction": "left", "degrees": 90}, trace)
                if not turned["success"]:
                    raise RuntimeError("explore scan turn failed")
        return []

    avoid = []
    try:
        seen = scan()
        for index in range(MAX_ITERATIONS):
            if seen or time.monotonic() - started > BUDGET_S:
                break
            position = executor.healthy()["trunk"][:2]
            path = mapper.plan_to_frontier(position, avoid)
            if path is None:
                result["stopped"] = "no reachable frontier left"
                break
            goal = path[-1]
            record = {"from": [round(float(v), 2) for v in position], "frontier": [round(float(goal[0]), 2), round(float(goal[1]), 2)],
                      "waypoints": len(path)}
            result["iterations"].append(record)
            try:
                for start in range(0, len(path), CHUNK):
                    chunk = path[start:start + CHUNK]
                    plan = {"status": "ok", "goals": [{"target": "point", "x": float(x), "y": float(y), "speed": "slow",
                                                      "stop_distance_m": 0.10} for x, y in chunk]}
                    execute(plan, executor.robot, executor.observer, output.with_suffix(f".nav{index}-{start}.jsonl"),
                            timeout_s=30, scene=mapper.as_scene(), cancel_file=executor.cancel_file,
                            cancel_started_ns=executor.started_ns,
                            observation_label="depth-built occupancy map; oracle pose")
                    # New sight lines open as the body moves; refresh the map between chunks.
                    look()
                record["reached"] = True
            except (ValueError, TimeoutError) as error:
                record.update(reached=False, error=f"{type(error).__name__}: {error}")
            avoid.append((float(goal[0]), float(goal[1])))
            executor.robot.stop()
            seen = scan()
        result["found"] = bool(seen)
        if seen:
            result["candidate_colors"] = sorted({c["color"] for c in seen})
        elif "stopped" not in result:
            result["stopped"] = "iteration/time budget used"
    finally:
        executor.robot.stop()
        result["explored_cells"] = mapper.explored_cells()
        result["elapsed_s"] = time.monotonic() - started
        np.save(output.with_suffix(".map.npy"), mapper.grid)
    result["success"] = result["found"]
    if not result["found"]:
        result["error"] = "target colour not found while exploring"
    return result
