"""Experiment: after Qwen has found and approached a ball, kick it or push it back to the start.

Only the walking and kicking use official policies; lining up, choosing the push direction and steering
are the program's. Finding and approaching the ball uses the camera as usual. During the kick and the
push the ball's position is read from the simulator, like the robot pose this project already reads,
because the ball at the robot's feet leaves the camera's view. The verb (kick / push) is recognised by a
rule in `parse`, not by the model: the adapter was never trained on these two skills.
"""

import heapq
import json
import math
from pathlib import Path
import re
import time

from .navigation import ROBOT_RADIUS, clearance, inside
from .skills import SkillExecutor, angle

WALK_VX, TURN_VYAW = 0.4, 1.5     # the simulator's command limits; smaller turns in place barely move the gait
TICK_S = 0.1
CELL_M = 0.10
KICK_AHEAD_M = (0.095, 0.140)     # ball centre ahead of the trunk when the official kick connects well (7 cm ball)
KICK_SIDE_M = 0.045               # ... and this far to the kicking foot's side
KICK_SIDE_TOLERANCE_M = 0.035
DELIVERED_M = 0.55                # the ball counts as brought back within this distance of the start
KICK, PUSH, NONE = "kick", "push", None


def parse(instruction):
    """(what Qwen is told, KICK | PUSH | None): cut the ball verb out of the sentence by rule."""
    for verb, words in ((PUSH, ("把球带", "将球带", "带着球", "带球", "带回", "把球推", "推球", "推回")), (KICK, ("踢",))):
        found = [instruction.find(word) for word in words if instruction.find(word) > 0]
        if found:
            head = re.sub(r"(?:[，,、。\s]|再|然后|接着|之后|并且|并|用脚)+$", "", instruction[:min(found)])
            if not any(word in head for word in ("走到", "走近", "靠近", "过去", "它旁边")):
                head += "，走到它旁边"
            return head, verb
    return instruction, NONE


class Grid:
    """A* over the known safety map; costs more near obstacles so routes keep room for the ball and the robot."""

    def __init__(self, scene, keep=0.06, comfort=0.35, border=0.30):
        self.scene, self.keep, self.comfort, self.border = scene, keep, comfort, border
        self.cache = {}

    def cell(self, x, y):
        return round(x / CELL_M), round(y / CELL_M)

    def room(self, cell):
        """Clearance of a cell for something as wide as the robot, outer shell included."""
        if cell not in self.cache:
            x, y = cell[0] * CELL_M, cell[1] * CELL_M
            xmin, xmax, ymin, ymax = self.scene["bounds"]
            edge = min(x - xmin, xmax - x, y - ymin, ymax - y) - self.border + self.keep
            self.cache[cell] = min(clearance((x, y), self.scene), edge) if inside((x, y), self.scene) else -1.0
        return self.cache[cell]

    def plan(self, start, goal, avoid=None, avoid_radius=0.0, reach=0.15):
        """[(x, y), ...] from start to within `reach` of goal, or None."""
        first, target = self.cell(*start), goal
        queue, cost, parent, closed = [(0.0, first)], {first: 0.0}, {}, set()
        while queue:
            _, here = heapq.heappop(queue)
            if here in closed:
                continue
            closed.add(here)
            x, y = here[0] * CELL_M, here[1] * CELL_M
            if math.dist((x, y), target) <= reach:
                path = [here]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                return [(c[0] * CELL_M, c[1] * CELL_M) for c in reversed(path)] + [tuple(goal)]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    there = here[0] + dx, here[1] + dy
                    if (not dx and not dy) or there in closed:
                        continue
                    room = self.room(there)
                    # The first cells may be tight (the robot can already stand close to something).
                    if room < self.keep and not (room >= 0 and math.dist((there[0] * CELL_M, there[1] * CELL_M), start) < 0.35):
                        continue
                    if avoid is not None and math.dist((there[0] * CELL_M, there[1] * CELL_M), avoid) < avoid_radius \
                            and math.dist((there[0] * CELL_M, there[1] * CELL_M), avoid) <= math.dist((x, y), avoid):
                        continue
                    step = math.hypot(dx, dy) * CELL_M * (1 + 4 * max(0.0, self.comfort - room) / self.comfort)
                    if cost[here] + step < cost.get(there, math.inf):
                        cost[there], parent[there] = cost[here] + step, here
                        heapq.heappush(queue, (cost[there] + math.dist((there[0] * CELL_M, there[1] * CELL_M), target), there))
        return None


def ahead(path, position, distance):
    """The point `distance` further along the path than its nearest point to `position`."""
    nearest = min(range(len(path)), key=lambda i: math.dist(path[i], position))
    left = distance
    for a, b in zip(path[nearest:], path[nearest + 1:]):
        piece = math.dist(a, b)
        if piece >= left:
            return a[0] + (b[0] - a[0]) * left / piece, a[1] + (b[1] - a[1]) * left / piece
        left -= piece
    return path[-1]


class BallPlayer:
    """Skills around one movable ball. `executor` is a SkillExecutor on a sim_ball simulator."""

    def __init__(self, executor, color, on_event=lambda text: None):
        self.executor, self.robot, self.observer = executor, executor.robot, executor.observer
        self.color, self.say = color, on_event
        self.movable = {n: o for n, o in executor.scene["objects"].items() if o["color"] == color and o.get("shape") == "sphere"}
        self.static = dict(executor.scene, objects={n: o for n, o in executor.scene["objects"].items() if n not in self.movable})
        self.grid = Grid(self.static)
        here = self.pose()[:2]
        balls = self.balls()
        self.name = min(balls, key=lambda n: math.dist(balls[n][:2], here))  # the ball Qwen just walked up to

    def balls(self):
        self.observer.connection.send({"op": "balls"})
        return self.observer.connection.receive()["balls"]

    def ball(self):
        return self.balls()[self.name]

    def pose(self):
        state = self.observer.read()
        if state["trunk_z"] < 0.085 or state["imu"]["gravity"][2] > -0.65:
            raise RuntimeError("fallen or unstable; stopped")
        cancel = self.executor.cancel_file
        if cancel is not None and cancel.exists() and int(cancel.read_text()) > self.executor.started_ns:
            raise RuntimeError("cancelled by stop command")
        return state["trunk"][0], state["trunk"][1], state["yaw"]

    def relative(self, direction):
        """(ahead, left) of the ball in a frame at the robot whose x axis points along `direction`."""
        x, y, _ = self.pose()
        bx, by = self.ball()[:2]
        dx, dy = bx - x, by - y
        return math.cos(direction) * dx + math.sin(direction) * dy, -math.sin(direction) * dx + math.cos(direction) * dy

    def steer(self, heading, vx=WALK_VX, turn_first=0.6):
        error = angle(heading - self.pose()[2])
        if abs(error) > turn_first:
            self.robot.move(0.0, math.copysign(TURN_VYAW, error))
        else:
            self.robot.move(vx, max(-1.0, min(1.0, 2.0 * error)))

    def back_off(self, distance=0.25):
        x, y, _ = self.pose()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and math.dist(self.pose()[:2], (x, y)) < distance:
            self.robot.move(-WALK_VX, 0.0)
            time.sleep(TICK_S)
        self.robot.stop()
        time.sleep(0.4)

    def settle_ball(self, timeout=8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and math.hypot(*self.ball()[3:5]) > 0.03:
            time.sleep(TICK_S)

    # -- kick -----------------------------------------------------------------

    def line_up(self, side, timeout=25.0):
        """Walk until the ball is in front of the kicking foot; True when it is within the kick's tolerance."""
        wanted = KICK_SIDE_M if side == "left" else -KICK_SIDE_M
        x, y, _ = self.pose()
        bx, by = self.ball()[:2]
        direction = math.atan2(by - y, bx - x)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            forward, left = self.relative(direction)
            if forward < 0.125:
                break
            correction = max(-0.6, min(0.6, 3.0 * (left - wanted)))
            self.steer(direction + correction, vx=WALK_VX if forward > 0.45 else 0.3, turn_first=0.5)
            time.sleep(TICK_S)
        self.robot.stop()
        time.sleep(0.8)
        # Face the kick direction before judging: the last correction may have left the body turned.
        for _ in range(12):
            error = angle(direction - self.pose()[2])
            if abs(error) < math.radians(8):
                break
            self.robot.move(0.0, math.copysign(TURN_VYAW, error))
            time.sleep(0.12)
        self.robot.stop()
        time.sleep(0.5)
        forward, left = self.relative(self.pose()[2])
        return (KICK_AHEAD_M[0] - 0.02 <= forward <= KICK_AHEAD_M[1] and abs(left - wanted) <= KICK_SIDE_TOLERANCE_M), forward, left

    def kick(self, tries=4):
        attempts = []
        for attempt in range(tries):
            _, left = self.relative(self.pose()[2])
            side = "left" if left >= 0 else "right"
            ready, forward, left = self.line_up(side)
            attempts.append({"side": side, "forward_m": round(forward, 3), "left_m": round(left, 3), "ready": ready})
            self.say(f"对位：球在前方 {forward * 100:.0f} cm、{'左' if left >= 0 else '右'}侧 {abs(left) * 100:.0f} cm，"
                     + (f"用{'左' if side == 'left' else '右'}脚踢" if ready else "不合适，后退重来"))
            if not ready:
                self.back_off(0.30)
                continue
            before = self.ball()[:2]
            self.robot.call("robot.kick", {"side": side})
            fastest, deadline = 0.0, time.monotonic() + 1.5
            while time.monotonic() < deadline:
                fastest = max(fastest, math.hypot(*self.ball()[3:5]))
                time.sleep(0.04)
            self.settle_ball()
            moved = math.dist(before, self.ball()[:2])
            attempts[-1].update(speed_m_s=round(fastest, 2), moved_m=round(moved, 2))
            if moved > 0.2:
                return {"success": True, "attempts": attempts, "ball_moved_m": round(moved, 2), "ball_speed_m_s": round(fastest, 2)}
            self.back_off(0.30)
        return {"success": False, "attempts": attempts, "error": "没有踢到球"}

    # -- push -----------------------------------------------------------------

    def push_home(self, home, timeout=300.0):
        """Walk the ball to `home`, re-planning its route around walls and the other balls."""
        started, route, planned, rounds, tight = time.monotonic(), None, 0.0, 0, 0
        begin = self.ball()[:2]
        going_round = False
        while time.monotonic() - started < timeout:
            ball = self.ball()
            spot = tuple(ball[:2])
            left_to_go = math.dist(spot, home)
            if left_to_go < DELIVERED_M:
                self.robot.stop()
                self.settle_ball()
                return {"success": math.dist(self.ball()[:2], home) < DELIVERED_M + 0.35, "seconds": round(time.monotonic() - started, 1),
                        "ball_from_home_m": round(math.dist(self.ball()[:2], home), 2), "carried_m": round(math.dist(begin, home), 2),
                        "walked_round": rounds}
            now = time.monotonic()
            if route is None or now - planned > 1.5:
                route, planned = self.grid.plan(spot, home, reach=0.3), now
                if route is None:
                    break
            aim = ahead(route, spot, 0.6)
            direction = math.atan2(aim[1] - spot[1], aim[0] - spot[0])
            x, y, yaw = self.pose()
            if clearance((x, y), self.static) < 0.0:
                tight += 1
                if tight > 25:
                    break
                self.back_off(0.12)
                continue
            forward, side = self.relative(direction)
            if forward > 0.10 and abs(side) < 0.05 + 0.25 * max(0.0, forward - 0.2) and forward < 1.2:
                going_round = False
                self.steer(direction + max(-0.6, min(0.6, 3.0 * side)))
            else:
                rounds += not going_round
                going_round = True
                behind = (spot[0] - 0.42 * math.cos(direction), spot[1] - 0.42 * math.sin(direction))
                way = self.grid.plan((x, y), behind, avoid=spot, avoid_radius=ROBOT_RADIUS + 0.16, reach=0.10)
                point = ahead(way, (x, y), 0.30) if way else behind
                self.steer(math.atan2(point[1] - y, point[0] - x), turn_first=0.5)
            time.sleep(TICK_S)
        self.robot.stop()
        return {"success": False, "seconds": round(time.monotonic() - started, 1), "walked_round": rounds,
                "ball_from_home_m": round(math.dist(self.ball()[:2], home), 2),
                "error": "超时" if time.monotonic() - started >= timeout else "球或机器人被卡住"}


def scene_with_moved_balls(info, player, path):
    """A scene file whose movable balls sit where they are now, so the way home plans around them."""
    scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
    for name, state in player.balls().items():
        scene["objects"][name].update(x=state[0], y=state[1])
    path.write_text(json.dumps(scene), encoding="utf-8")
    return dict(info, scene_file=str(path))


def play(info, directory, verb, color, on_event=lambda text: None, cancel_file=None):
    """Kick or push the ball next to the robot, then walk home. Returns a report; never raises for a missed ball."""
    directory.mkdir(parents=True, exist_ok=True)
    report = {"verb": verb, "color": color}
    executor = SkillExecutor(info, directory, cancel_file=cancel_file)
    try:
        player = BallPlayer(executor, color, on_event)
        home = info["home_pose"]
        if verb == KICK:
            on_event("开始对位，准备踢球")
            report["ball"] = player.kick()
        else:
            on_event("开始带球回起点")
            report["ball"] = player.push_home((home["x"], home["y"]))
            if report["ball"]["success"]:
                player.back_off(0.30)
        moved = scene_with_moved_balls(info, player, directory / "moved.scene.json")
    finally:
        executor.close()
    on_event("返回起点")
    walker = SkillExecutor(moved, directory, cancel_file=cancel_file)
    walker.index = 50
    try:
        report["return"] = walker.run({"action": "return_home"})
    finally:
        walker.close()
    report["distance_home_m"] = round(report["return"].get("distance_home_m", math.nan), 3)
    return report
