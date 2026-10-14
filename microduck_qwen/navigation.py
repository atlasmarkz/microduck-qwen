"""Geometric navigation using MuJoCo ground truth, above the existing 50 Hz gait."""

import heapq
import json
import math
import socket
import time

from .contract import arena, validate

ROBOT_RADIUS = 0.14
PATH_MARGIN = 0.12
GRID = 0.05
# These are policy conditioning inputs, NOT measured physical speeds. Commands below
# about 0.2 produce little motion with the installed walking actor. See check_gait.
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
        self.connection = JsonConnection(str(path), unix=True)
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


def obstacles(scene):
    return scene["obstacles"] + [
        obj for obj in scene["objects"].values() if obj["collidable"]
    ]


def clearance(point, scene):
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


def segment_clear(a, b, scene):
    if not inside(a, scene) or not inside(b, scene):
        return False
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
        padding = obstacle.get("path_margin", PATH_MARGIN)
        if math.dist(nearest, center) < obstacle["radius"] + ROBOT_RADIUS + padding:
            return False
    return True


def pathfind(start, target, radius, scene):
    """A* to a free cell inside the requested stopping region; no diagonal corner cutting."""
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
):
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
    execution_started_ns = time.time_ns()
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
                    visual_check = guard(state, goal) if guard is not None else None
                    if visual_check is not None and not visual_check.get("safe", False):
                        raise RuntimeError(
                            "视觉保护停止："
                            + str(visual_check.get("status", "unknown"))
                        )
                    if distance <= goal["stop_distance_m"] + 0.025:
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
                    heading = math.atan2(
                        waypoint[1] - position[1], waypoint[0] - position[0]
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
