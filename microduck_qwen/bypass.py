"""Bounded yellow-obstacle passing task with measured order and return checks.

RGB-D confirms yellow objects; the existing privileged static map supplies their
centers and safety geometry. This is not general visual mapping or online RL.
"""

import json
import math
import time

from .navigation import ROBOT_RADIUS, clearance, execute, inside, pathfind


def yellow_obstacles(scene, start):
    objects = [o for o in scene["obstacles"] if o.get("rgba", [0, 0, 0])[0] > 0.6
               and o["rgba"][1] > 0.4 and o["rgba"][2] < 0.25]
    if not 1 <= len(objects) <= 3:
        raise ValueError("yellow bypass supports one to three known yellow obstacles")
    return sorted(objects, key=lambda o: (math.hypot(o["x"] - start["x"], o["y"] - start["y"]), o["name"]))


def yellow_detections(frame):
    import cv2
    import numpy as np
    from .vision import unproject

    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 40)
            & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 80)).astype("uint8")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 25:
            continue
        ys, xs = np.where(labels == index)
        depths = frame.depth[ys, xs]
        good = np.isfinite(depths) & (depths > 0.06) & (depths < 5)
        if good.sum() < len(depths) * 0.6:
            continue
        point = unproject(float(np.median(xs[good])), float(np.median(ys[good])), float(np.median(depths[good])), frame.metadata)
        x, y, width, height, area = [int(value) for value in stats[index]]
        found.append({"position": point[:2].tolist(), "pixels": int(good.sum()),
                      "surface_position": point.tolist(), "pixel": [float(np.median(xs[good])), float(np.median(ys[good]))],
                      "bbox": [x, y, x + width, y + height], "depth": float(np.median(depths[good])),
                      "valid_depth_fraction": float(good.sum() / area)})
    return found


def plan_bypass(scene, start, side):
    direction = (math.cos(start["yaw"]), math.sin(start["yaw"]))
    normal = (-direction[1], direction[0])
    current = (start["x"], start["y"])
    legs = []
    for obstacle in yellow_obstacles(scene, start):
        radius = obstacle["radius"] + ROBOT_RADIUS + 0.28
        choices = []
        for label, sign in (("left", 1), ("right", -1)):
            if side != "auto" and side != label:
                continue
            points = [[obstacle["x"] + along * radius * direction[0] + sign * radius * normal[0],
                       obstacle["y"] + along * radius * direction[1] + sign * radius * normal[1]] for along in (-1, 1)]
            if any(not inside(p, scene) or clearance(p, scene) < 0.04 for p in points):
                continue
            cost, previous = 0.0, current
            try:
                for point in points:
                    route = pathfind(previous, point, 0.10, scene)
                    cost += sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                    previous = point
            except ValueError:
                continue
            choices.append((cost, label, points))
        if not choices:
            raise ValueError(f"no safe {side} passing route for {obstacle['name']}")
        _, label, points = min(choices)
        legs.append({"obstacle": obstacle, "side": label, "points": points})
        current = points[-1]
    pathfind(current, (start["x"], start["y"]), 0.10, scene)
    return legs


def verify_passes(positions, legs, start):
    """Score the actual trace, not merely whether waypoint controllers returned OK."""
    if any(math.dist(a, b) > 0.25 for a, b in zip(positions, positions[1:])):
        return {"success": False, "reason": "discontinuous position trace"}
    direction = (math.cos(start["yaw"]), math.sin(start["yaw"]))
    normal = (-direction[1], direction[0])
    completions = []
    for leg in legs:
        obstacle, phase = leg["obstacle"], 0
        envelope = obstacle["radius"] + ROBOT_RADIUS + 0.03
        sign = 1 if leg["side"] == "left" else -1
        for index, position in enumerate(positions):
            dx, dy = position[0] - obstacle["x"], position[1] - obstacle["y"]
            along, across = dx * direction[0] + dy * direction[1], dx * normal[0] + dy * normal[1]
            if math.hypot(dx, dy) > 0.95:
                continue
            if phase == 0 and along < -envelope:
                phase = 1
            elif phase == 1 and abs(along) < 0.15 and sign * across >= envelope:
                phase = 2
            elif phase == 2 and along > envelope:
                completions.append(index)
                break
        else:
            return {"success": False, "reason": f"trace did not pass the requested side of {obstacle['name']}"}
    ordered = all(a < b for a, b in zip(completions, completions[1:]))
    return {"success": ordered, "completion_indices": completions,
            "obstacle_order": [leg["obstacle"]["name"] for leg in legs]}


def run_bypass(executor, action, output):
    from .skills import NeedsUserInput
    from .vision import capture

    state = executor.healthy()
    start = getattr(executor, "task_start_pose", None) or {"x": state["trunk"][0], "y": state["trunk"][1], "yaw": state["yaw"]}
    try:
        legs = plan_bypass(executor.scene, start, action["side"])
    except ValueError as error:
        raise NeedsUserInput(str(error)) from error
    detections = yellow_detections(capture(executor.info["port"]))
    bound = []
    for leg in legs:
        center = (leg["obstacle"]["x"], leg["obstacle"]["y"])
        matches = [d for d in detections if math.dist(d["position"], center) < 0.27]
        if not matches:
            raise NeedsUserInput(f"yellow obstacle {leg['obstacle']['name']} not confirmed in initial RGB-D view")
        bound.append(min(matches, key=lambda d: math.dist(d["position"], center)))
    positions = [state["trunk"][:2]]
    results = []
    for index, leg in enumerate(legs):
        goals = [{"target": "point", "x": p[0], "y": p[1], "speed": "normal", "stop_distance_m": 0.10} for p in leg["points"]]
        trace = output.with_name(output.stem + f"-pass-{index:02d}.jsonl")
        result = execute({"status": "ok", "goals": goals}, executor.robot, executor.observer,
                         trace, scene=executor.scene, timeout_s=35, cancel_file=executor.cancel_file,
                         arrival_tolerance_m=0, cancel_started_ns=executor.started_ns)
        results.append(result)
        if not result["success"]:
            raise RuntimeError("bypass navigation failed or cancelled")
        positions.extend(json.loads(line)["position"] for line in trace.read_text().splitlines())
        time.sleep(0.25)
        positions.append(executor.healthy()["trunk"][:2])
    passed = verify_passes(positions, legs, start)
    if not passed["success"]:
        raise RuntimeError(str(passed))
    home = executor._return_to_pose(start, output.with_name(output.stem + "-return.json"))
    minimum = min([clearance(p, executor.scene) for p in positions]
                  + [g["min_clearance_m"] for g in home["navigation"]["goals"]])
    return {"success": bool(home["success"] and minimum >= 0.025), "task_start_pose": start,
            "legs": legs, "visual_bindings": bound, "navigation": results, "trace_verification": passed,
            "minimum_clearance_m": minimum, "return": home,
            "observation": "RGB-D yellow confirmation; oracle pose/static map; near-to-far order frozen at task start"}
