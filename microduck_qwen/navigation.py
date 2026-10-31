"""Geometric navigation using MuJoCo ground truth, above the existing 50 Hz gait."""

import heapq
import json
import math
import socket
import time

import numpy as np

from .contract import arena, validate

ROBOT_RADIUS = 0.14
PATH_MARGIN = 0.12
WALL_PATH_MARGIN = 0.20
# A guarded approach may walk straight at the object while facing it. That line
# ends stop_distance_m short of the object, so the object's own body is not on
# it; this smaller margin only has to keep real blockers off the line.
FACE_TARGET_MARGIN = 0.05
GRID = 0.05
# These are policy conditioning inputs, NOT measured physical speeds. Commands below
# about 0.2 produce little motion with the installed walking actor.
SPEED = {"slow": 0.30, "normal": 0.35, "fast": 0.40}


class JsonConnection:
    def __init__(self, address, unix=False):
        self.socket = socket.socket(
            socket.AF_UNIX if unix else socket.AF_INET, socket.SOCK_STREAM
        )
        self.socket.settimeout(2)
        self.socket.connect(address)
        self.file = self.socket.makefile("rwb")

    def send(self, value):
        self.file.write((json.dumps(value, allow_nan=False) + "\n").encode())
        self.file.flush()

    def receive(self):
        line = self.file.readline(8 * 1024 * 1024)
        if not line:
            raise ConnectionError("仿真连接已断开")
        value = json.loads(line)
        if "error" in value:
            raise RuntimeError(value["error"])
        return value

    def close(self):
        self.file.close()
        self.socket.close()


class Robot:
    def __init__(self, path):
        self.connection = JsonConnection(tuple(path) if isinstance(path, list) else str(path),
                                         unix=not isinstance(path, list))
        self.request_id = 0

    def call(self, method, params=None):
        self.request_id += 1
        self.connection.send(
            {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params or {},
            }
        )
        answer = self.connection.receive()
        if answer.get("id") != self.request_id:
            raise RuntimeError("RPC 返回的 id 不匹配")
        return answer.get("result")

    def move(self, vx=0.0, vyaw=0.0):
        self.connection.send(
            {
                "jsonrpc": "2.0",
                "method": "robot.move",
                "params": {"vx": vx, "vy": 0.0, "vyaw": vyaw},
            }
        )

    def stop(self):
        self.move()
        return self.call("robot.stop")

    def close(self):
        self.connection.close()


class Observer:
    def __init__(self, port):
        self.connection = JsonConnection(("127.0.0.1", port))
        self.connection.send({"op": "hello", "protocol": 1, "joints": 15})
        self.connection.receive()

    def read(self):
        self.connection.send({"op": "read"})
        state = self.connection.receive()
        w, x, y, z = state["imu"]["quat"]
        state["yaw"] = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return state

    def close(self):
        self.connection.close()


_CACHE = {}


def _cached(owner, tag, build):
    """Memoise per scene object. The entry keeps the owner alive, so its id cannot be reused."""
    hit = _CACHE.get((id(owner), tag))
    if hit is not None and hit[0] is owner:
        return hit[1]
    if len(_CACHE) > 64:
        _CACHE.clear()
    value = build()
    _CACHE[(id(owner), tag)] = (owner, value)
    return value


def _wall_circles(scene):
    """Planned walls as overlapping circles, so one collision model covers every obstacle."""
    def build():
        circles = []
        for wall in scene["walls"]:
            if not wall.get("planning", True):
                continue
            radius = wall["thickness"] / 2
            length = math.dist((wall["x1"], wall["y1"]), (wall["x2"], wall["y2"]))
            pieces = max(1, math.ceil(length / radius))
            for k in range(pieces + 1):
                t = k / pieces
                # Doorways are 1.1 m wide: a wider standoff than the default keeps the gait's
                # lateral wobble from brushing a door post on long, cornered routes.
                circles.append({"name": f"{wall['name']}#{k}", "radius": radius, "path_margin": WALL_PATH_MARGIN,
                                "x": wall["x1"] + (wall["x2"] - wall["x1"]) * t,
                                "y": wall["y1"] + (wall["y2"] - wall["y1"]) * t})
        return circles
    return _cached(scene["walls"], "wall-circles", build)


def obstacles(scene):
    found = scene["obstacles"] + [
        obj for obj in scene["objects"].values() if obj["collidable"]
    ]
    return found + _wall_circles(scene) if "walls" in scene else found


def _circle_arrays(scene, margin):
    """(x, y, radius, inflated radius) arrays; world scenes only, evaluated in one numpy pass."""
    def build():
        items = obstacles(scene)
        xs = np.array([o["x"] for o in items], dtype=float)
        ys = np.array([o["y"] for o in items], dtype=float)
        rs = np.array([o["radius"] for o in items], dtype=float)
        pads = np.array([o.get("path_margin", margin) for o in items], dtype=float)
        return xs, ys, rs, rs + ROBOT_RADIUS + pads
    return _cached(scene, ("circles", margin), build)


def _relaxed_clear(a, b, scene):
    """Short initial bridge only: static obstacles and walls at 3 cm padding, balls ignored."""
    def build():
        items = scene["obstacles"] + (_wall_circles(scene) if "walls" in scene else [])
        xs = np.array([o["x"] for o in items], dtype=float)
        ys = np.array([o["y"] for o in items], dtype=float)
        return xs, ys, np.array([o["radius"] + ROBOT_RADIUS + min(0.03, o.get("path_margin", 0.03)) for o in items], dtype=float)
    xs, ys, limit = _cached(scene, "relaxed", build)
    if not inside(a, scene) or not inside(b, scene):
        return False
    if not len(xs):
        return True
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = np.zeros(len(xs)) if not length2 else np.clip(((xs - a[0]) * dx + (ys - a[1]) * dy) / length2, 0.0, 1.0)
    return bool((np.hypot(a[0] + t * dx - xs, a[1] + t * dy - ys) >= limit).all())


def clearance(point, scene):
    if "walls" in scene:
        xs, ys, rs, _ = _circle_arrays(scene, PATH_MARGIN)
        if not len(xs):
            return 100.0
        return float((np.hypot(xs - point[0], ys - point[1]) - rs - ROBOT_RADIUS).min())
    return min(
        (
            math.dist(point, (o["x"], o["y"])) - o["radius"] - ROBOT_RADIUS
            for o in obstacles(scene)
        ),
        default=100.0,
    )


def inside(point, scene):
    x, y = point
    xmin, xmax, ymin, ymax = scene["bounds"]
    return xmin <= x <= xmax and ymin <= y <= ymax


def segment_clear(a, b, scene, margin=PATH_MARGIN):
    if not inside(a, scene) or not inside(b, scene):
        return False
    if "walls" in scene:
        xs, ys, _, limit = _circle_arrays(scene, margin)
        if not len(xs):
            return True
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = np.zeros(len(xs)) if not length2 else np.clip(((xs - a[0]) * dx + (ys - a[1]) * dy) / length2, 0.0, 1.0)
        return bool((np.hypot(a[0] + t * dx - xs, a[1] + t * dy - ys) >= limit).all())
    for obstacle in obstacles(scene):
        center = (obstacle["x"], obstacle["y"])
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = (
            0.0
            if not length2
            else max(
                0.0,
                min(1.0, ((center[0] - a[0]) * dx + (center[1] - a[1]) * dy) / length2),
            )
        )
        nearest = (a[0] + t * dx, a[1] + t * dy)
        padding = obstacle.get("path_margin", margin)
        if math.dist(nearest, center) < obstacle["radius"] + ROBOT_RADIUS + padding:
            return False
    return True


def face_target_heading(position, target, stop_distance, scene, margin=FACE_TARGET_MARGIN):
    """Heading that walks a clear straight line at the object, or None.

    The body must face the object for its numbered box to stay in frame as the
    distance closes; the path-follower heading aims at a 0.05m waypoint instead,
    which is what used to slide the object out of view on the last metre.
    """
    distance = math.dist(position, target)
    if distance < 1e-6:
        return None
    ratio = max(0.0, (distance - stop_distance) / distance)
    aim = (
        position[0] + (target[0] - position[0]) * ratio,
        position[1] + (target[1] - position[1]) * ratio,
    )
    if not segment_clear(position, aim, scene, margin):
        return None
    if math.dist(position, aim) < 1e-6:
        aim = (target[0], target[1])
    return math.atan2(aim[1] - position[1], aim[0] - position[0])


def _free_grid(scene, grid):
    """Boolean cell map of positions the robot centre may occupy, built once per scene."""
    def build():
        xmin, xmax, ymin, ymax = scene["bounds"]
        nx, ny = round((xmax - xmin) / grid), round((ymax - ymin) / grid)
        gx, gy = xmin + np.arange(nx + 1) * grid, ymin + np.arange(ny + 1) * grid
        free = (gx[:, None] <= xmax + 1e-9) & (gy[None, :] <= ymax + 1e-9)
        xs, ys, _, limit = _circle_arrays(scene, PATH_MARGIN)
        for x, y, reach in zip(xs, ys, limit):
            i0, i1 = max(0, int((x - reach - xmin) // grid)), min(nx, int((x + reach - xmin) // grid) + 1)
            j0, j1 = max(0, int((y - reach - ymin) // grid)), min(ny, int((y + reach - ymin) // grid) + 1)
            if i0 > i1 or j0 > j1:
                continue
            blocked = (gx[i0:i1 + 1, None] - x) ** 2 + (gy[None, j0:j1 + 1] - y) ** 2 < reach * reach
            free[i0:i1 + 1, j0:j1 + 1] &= ~blocked
        return nx, ny, free.tolist()
    return _cached(scene, ("free-grid", grid), build)


def _pathfind_world(start, target, radius, scene):
    """A* over the cached cell map of a world scene (many obstacles, long routes)."""
    xmin, xmax, ymin, ymax = scene["bounds"]
    grid = scene.get("grid", GRID)
    nx, ny, free = _free_grid(scene, grid)
    point = lambda c: (xmin + c[0] * grid, ymin + c[1] * grid)
    cell = lambda p: (min(nx, max(0, round((p[0] - xmin) / grid))), min(ny, max(0, round((p[1] - ymin) / grid))))
    if not inside(start, scene):
        raise ValueError("机器人已离开导航地图范围")
    if math.dist(start, target) <= radius:
        return [start]
    first = cell(start)
    if not free[first[0]][first[1]] or not segment_clear(start, point(first), scene):
        # (balls the robot already stands inside the padding of must not veto the bridge)
        # A valid continuous pose can round into an inflated cell: bridge to a nearby free cell
        # with the same reduced 3 cm padding the small-scene planner uses.
        near = [(first[0] + dx, first[1] + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
        near = [c for c in near if 0 <= c[0] <= nx and 0 <= c[1] <= ny and free[c[0]][c[1]]
                and _relaxed_clear(start, point(c), scene)]
        if not near:
            raise ValueError("机器人距离障碍太近，无法生成安全的初始路径")
        first = min(near, key=lambda c: math.dist(start, point(c)))
    queue, cost, parent, goal = [(0.0, first)], {first: 0.0}, {}, None
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        _, current = heapq.heappop(queue)
        if math.dist(point(current), target) <= radius:
            goal = current
            break
        for dx, dy in steps:
            other = (current[0] + dx, current[1] + dy)
            if not (0 <= other[0] <= nx and 0 <= other[1] <= ny) or not free[other[0]][other[1]]:
                continue
            if dx and dy and not (free[current[0] + dx][current[1]] and free[current[0]][current[1] + dy]):
                continue  # no diagonal corner cutting
            new_cost = cost[current] + math.hypot(dx, dy) * grid
            if new_cost < cost.get(other, float("inf")):
                cost[other], parent[other] = new_cost, current
                heapq.heappush(queue, (new_cost + max(0.0, math.dist(point(other), target) - radius), other))
    if goal is None:
        raise ValueError("目标的停靠区域不可达；请调整目标或停靠距离")
    reverse = [point(goal)]
    while goal != first:
        goal = parent[goal]
        reverse.append(point(goal))
    route = [start] + list(reversed(reverse))
    smooth, index = [start], 0
    while index < len(route) - 1:
        next_index = index + 1
        for candidate in range(len(route) - 1, index, -1):
            if segment_clear(smooth[-1], route[candidate], scene):
                next_index = candidate
                break
        smooth.append(route[next_index])
        index = next_index
    return smooth


def pathfind(start, target, radius, scene):
    """A* to a free cell inside the requested stopping region; no diagonal corner cutting."""
    if "walls" in scene:
        return _pathfind_world(start, target, radius, scene)
    xmin, xmax, ymin, ymax = scene["bounds"]
    nx, ny = round((xmax - xmin) / GRID), round((ymax - ymin) / GRID)

    def point(cell):
        return (xmin + cell[0] * GRID, ymin + cell[1] * GRID)

    def cell(point):
        return (round((point[0] - xmin) / GRID), round((point[1] - ymin) / GRID))

    def free(c):
        return (
            0 <= c[0] <= nx
            and 0 <= c[1] <= ny
            and segment_clear(point(c), point(c), scene)
        )

    if not inside(start, scene):
        raise ValueError("机器人已离开导航地图范围")
    if math.dist(start, target) <= radius:
        return [start]
    first = cell(start)
    if not free(first) or not segment_clear(start, point(first), scene):
        # Rounding a valid continuous pose to a grid cell can put that cell inside
        # the inflated obstacle. Bridge to a nearby free cell instead of stopping
        # at an artificial grid boundary. Only this short initial bridge may use
        # reduced padding; the physical envelope still retains 3 cm clearance.
        relaxed = dict(
            scene,
            obstacles=[
                dict(o, path_margin=min(0.03, o.get("path_margin", PATH_MARGIN)))
                for o in obstacles(scene)
            ],
            objects={},
        )
        candidates = [
            (first[0] + dx, first[1] + dy) for dx in range(-2, 3) for dy in range(-2, 3)
        ]
        candidates = [
            c for c in candidates if free(c) and segment_clear(start, point(c), relaxed)
        ]
        if not candidates:
            raise ValueError("机器人距离障碍太近，无法生成安全的初始路径")
        first = min(candidates, key=lambda c: math.dist(start, point(c)))
    queue = [(0.0, first)]
    cost, parent = {first: 0.0}, {}
    goal = None
    while queue:
        _, current = heapq.heappop(queue)
        if math.dist(point(current), target) <= radius:
            goal = current
            break
        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            other = (current[0] + dx, current[1] + dy)
            if not free(other) or not segment_clear(
                point(current), point(other), scene
            ):
                continue
            new_cost = cost[current] + math.hypot(dx, dy) * GRID
            if new_cost < cost.get(other, float("inf")):
                cost[other], parent[other] = new_cost, current
                # Distance to the goal region is admissible; distance to its center is not.
                heuristic = max(0.0, math.dist(point(other), target) - radius)
                heapq.heappush(queue, (new_cost + heuristic, other))
    if goal is None:
        raise ValueError("目标的停靠区域不可达；请调整目标或停靠距离")
    reverse = [point(goal)]
    while goal != first:
        goal = parent[goal]
        reverse.append(point(goal))
    route = [start] + list(reversed(reverse))
    smooth = [start]
    index = 0
    while index < len(route) - 1:
        next_index = index + 1
        for candidate in range(len(route) - 1, index, -1):
            if segment_clear(smooth[-1], route[candidate], scene):
                next_index = candidate
                break
        smooth.append(route[next_index])
        index = next_index
    return smooth


def target_position(goal, scene):
    target = goal if goal["target"] == "point" else scene["objects"][goal["target"]]
    return (target["x"], target["y"])


def execute(
    plan,
    connection,
    observer,
    trace_file,
    timeout_s=60.0,
    scene=None,
    cancel_file=None,
    guard=None,
    observation_label=None,
    arrival_tolerance_m=0.025,
    cancel_started_ns=None,
):
    if not math.isfinite(arrival_tolerance_m) or not 0 <= arrival_tolerance_m <= 0.025:
        raise ValueError("arrival tolerance must be 0..0.025m")
    scene = scene or arena()
    validate(plan, scene)
    if plan["status"] == "unsupported":
        raise ValueError("当前模型不支持该指令。可导航到红球、蓝色标记或指定世界坐标。")
    if plan["status"] == "stop":
        connection.stop()
        return {"success": True, "stopped": True}
    summary = {
        "success": False,
        "goals": [],
        "observation": "MuJoCo ground-truth pose and fixed scene map",
    }
    if observation_label:
        summary["observation"] = observation_label
    execution_started_ns = time.time_ns() if cancel_started_ns is None else cancel_started_ns
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with trace_file.open("w") as trace:
            for goal in plan["goals"]:
                target = target_position(goal, scene)
                started = time.monotonic()
                last_replan = 0.0
                last_sim_time, last_advance = None, started
                route = []
                min_clearance = float("inf")
                previous = None
                distance_travelled = 0.0
                while True:
                    if cancel_file is not None and cancel_file.exists():
                        if int(cancel_file.read_text()) > execution_started_ns:
                            summary["cancelled"] = True
                            return summary
                    now = time.monotonic()
                    state = observer.read()
                    position = tuple(state["trunk"][:2])
                    if previous is not None:
                        distance_travelled += math.dist(previous, position)
                    previous = position
                    margin = clearance(position, scene)
                    min_clearance = min(min_clearance, margin)
                    if state["sim_time"] != last_sim_time:
                        last_sim_time, last_advance = state["sim_time"], now
                    if now - last_advance > 0.8:
                        raise RuntimeError("物理仿真停止推进")
                    if state["trunk_z"] < 0.085 or state["imu"]["gravity"][2] > -0.65:
                        raise RuntimeError("MicroDuck 跌倒或姿态不稳定，已停止导航")
                    if margin < 0.015:
                        raise RuntimeError("障碍物净空过小，已停止导航")
                    distance = math.dist(position, target)
                    # A guarded approach walks a clear straight line at the
                    # object while facing it, so its numbered box stays in
                    # frame. A blocked leg is a detour around whatever hides
                    # the object: the planned route, not the sight line, drives
                    # that leg and the object is expected to slip out of frame
                    # while the body turns away. The guard therefore only gates
                    # the leg that is actually driven at the object, which is
                    # the only leg where losing sight is real evidence.
                    aimed = (
                        face_target_heading(
                            position, target, goal["stop_distance_m"], scene
                        )
                        if guard is not None
                        else None
                    )
                    visual_check = None
                    if guard is not None:
                        if aimed is None:
                            guard.suspend()
                        else:
                            visual_check = guard(state, goal)
                            if not visual_check.get("safe", False):
                                raise RuntimeError(
                                    "视觉保护停止："
                                    + str(visual_check.get("status", "unknown"))
                                )
                    if distance <= goal["stop_distance_m"] + arrival_tolerance_m:
                        connection.stop()
                        summary["goals"].append(
                            {
                                "goal": goal,
                                "distance_m": distance,
                                "elapsed_s": now - started,
                                "path_length_m": distance_travelled,
                                "min_clearance_m": min_clearance,
                                "position": position,
                            }
                        )
                        print(
                            f"到达 {goal['target']}：距离 {distance:.3f}m", flush=True
                        )
                        break
                    if now - started > timeout_s:
                        raise TimeoutError(f"{goal['target']} 导航超过 {timeout_s}s")
                    if now - last_replan >= 0.5 or not route:
                        route = pathfind(
                            position, target, goal["stop_distance_m"], scene
                        )
                        last_replan = now
                    while len(route) > 1 and math.dist(position, route[1]) < 0.07:
                        route.pop(0)
                    waypoint = route[1] if len(route) > 1 else route[0]
                    heading = (
                        aimed
                        if aimed is not None
                        else math.atan2(
                            waypoint[1] - position[1], waypoint[0] - position[0]
                        )
                    )
                    error = math.atan2(
                        math.sin(heading - state["yaw"]),
                        math.cos(heading - state["yaw"]),
                    )
                    # Turn above the actor's dead zone, with a small zero-command band.
                    # Sending tiny continuous corrections makes this gait stand still.
                    # The official interactive runner permits yaw conditioning up
                    # to 1.5; 1.0 alone can settle into a no-step posture after a stop.
                    vyaw = (
                        math.copysign(1.5 if abs(error) > 0.35 else 1.2, error)
                        if abs(error) > 0.12
                        else 0.0
                    )
                    vx = 0.0 if abs(error) > 0.35 else SPEED[goal["speed"]]
                    connection.move(vx, vyaw)
                    trace.write(
                        json.dumps(
                            {
                                "wall_s": now - started,
                                "sim_time": state["sim_time"],
                                "position": position,
                                "yaw": state["yaw"],
                                "vx": vx,
                                "vyaw": vyaw,
                                "target_distance_m": distance,
                                "clearance_m": margin,
                                "goal": goal,
                                "vision": visual_check,
                            }
                        )
                        + "\n"
                    )
                    trace.flush()
                    time.sleep(max(0.0, 0.1 - (time.monotonic() - now)))
        summary["success"] = True
        return summary
    finally:
        # The native daemon also zeros stale intents after 500 ms if this process disappears.
        try:
            connection.stop()
        except (OSError, RuntimeError, ConnectionError):
            pass
