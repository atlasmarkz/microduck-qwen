"""Bounded simulation skills. Robot pose/map are privileged localization, not VLM object input."""

import json
import math
from pathlib import Path
import time

from .navigation import Robot, Observer, execute, clearance, inside
from .vision import capture, propose_targets, visibility_check, approachable, VisualGuard


class NeedsUserInput(RuntimeError):
    """A pre-execution ambiguity or blocked path; no motion has been issued."""


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"number must be in {low}..{high}")
    return float(value)


def validate_action(action):
    if not isinstance(action, dict):
        raise ValueError("action must be an object")
    name = action.get("action")
    keys = {
        "move": {"action", "direction", "distance_m"}, "turn": {"action", "direction", "degrees"},
        "approach": {"action", "color", "selector" if "selector" in action else "id"} | ({"kind"} & set(action)),
        "search": {"action", "color"},
        "explore": {"action", "color"},
        "return_home": {"action"}, "stop": {"action"}, "done": {"action"},
        "clarify": {"action"}, "unsupported": {"action"},
        "bypass_return": {"action", "side"},
    }
    if name not in keys or set(action) != keys[name]:
        raise ValueError("unknown action or unexpected fields")
    if name == "move":
        if action["direction"] not in ("forward", "backward"):
            raise ValueError("invalid move direction")
        number(action["distance_m"], 0.05, 0.35)
    if name == "turn":
        if action["direction"] not in ("left", "right"):
            raise ValueError("invalid turn direction")
        number(action["degrees"], 5, 120)
    if name in ("approach", "search", "explore") and action["color"] not in ("red", "blue", "green", "yellow"):
        raise ValueError("supported targets are red/blue/green balls and yellow pillars")
    if name == "approach":
        if "kind" in action and action["kind"] not in ("ball", "pillar"):
            raise ValueError("invalid target kind")
        if "selector" in action:
            if action["selector"] not in ("nearest", "leftmost", "rightmost", "only"):
                raise ValueError("invalid target selector")
        elif type(action["id"]) is not int or not 1 <= action["id"] <= 20:
            raise ValueError("invalid current-frame target ID")
        elif action["color"] == "yellow":
            raise ValueError("yellow pillars require selector=nearest/leftmost/rightmost/only, not guessed id")
    if name == "bypass_return" and action["side"] not in ("auto", "left", "right"):
        raise ValueError("invalid bypass side")
    return action


def parse_action(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate action key")
            result[key] = value
        return result
    return validate_action(json.loads(raw.strip(), object_pairs_hook=unique))


def bind_target(action, candidates, position):
    choices = [c for c in candidates or [] if c["color"] == action["color"]
               and ("kind" not in action or c.get("kind", "ball") == action["kind"])]
    if "id" in action:
        selected = next((c for c in choices if c["id"] == action["id"]), None)
    else:
        selector = action["selector"]
        if not choices:
            raise NeedsUserInput("no currently visible target of the requested color")
        if selector == "only":
            if len(choices) != 1:
                raise NeedsUserInput(f"{len(choices)} matching objects; specify nearest, leftmost or rightmost")
            selected = choices[0]
        elif selector == "nearest":
            selected = min(choices, key=lambda c: math.dist(position, c["world_position"][:2]))
        else:
            selected = sorted(choices, key=lambda c: c["pixel"][0], reverse=selector == "rightmost")[0]
    if selected is None or not approachable(selected):
        raise NeedsUserInput("target is not bound to a sufficiently visible current-frame candidate")
    return selected


def memory_state(info, state, completed):
    home = info.get("home_pose")
    return {"home_recorded": home is not None,
            "odometry_relative_home": {"x": round(state["trunk"][0] - home["x"], 2),
                                       "y": round(state["trunk"][1] - home["y"], 2),
                                       "yaw_deg": round(math.degrees(angle(state["yaw"] - home["yaw"])), 1)} if home else None,
            "completed": completed[-4:]}


class SkillExecutor:
    def __init__(self, info, directory, backward_command=0.40, cancel_file=None):
        self.info, self.directory = info, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scene = json.loads(Path(info["scene_file"]).read_text())
        self.robot = Robot(info["socket"])
        try:
            self.observer = Observer(info["port"])
        except BaseException:
            try:
                self.robot.stop()
            finally:
                self.robot.close()
            raise
        self.index = 0
        self.last_sim = None
        self.last_advance = time.monotonic()
        self.backward_command = number(backward_command, 0.20, 0.40)
        self.cancel_file, self.started_ns = cancel_file, time.time_ns()

    def close(self):
        try:
            self.robot.stop()
        finally:
            try:
                self.robot.close()
            finally:
                self.observer.close()

    def healthy(self):
        if self.cancel_file is not None and self.cancel_file.exists() and int(self.cancel_file.read_text()) > self.started_ns:
            raise RuntimeError("cancelled by stop command")
        state = self.observer.read()
        if state["trunk_z"] < 0.085 or state["imu"]["gravity"][2] > -0.65:
            raise RuntimeError("fallen or unstable; stopped")
        position = state["trunk"][:2]
        if not inside(position, self.scene) or clearance(position, self.scene) < 0.025:
            raise RuntimeError("outside map or scene clearance too small; stopped")
        if state["sim_time"] != self.last_sim:
            self.last_sim, self.last_advance = state["sim_time"], time.monotonic()
        elif time.monotonic() - self.last_advance > 0.8:
            raise RuntimeError("stale physics; stopped")
        return state

    def _bounded(self, action, trace):
        start = self.healthy()
        origin = start["trunk"][:2]
        yaw = start["yaw"]
        tick = time.monotonic()
        signed = 1 if action["direction"] in ("forward", "left") else -1
        turning = action["action"] == "turn"
        desired = math.radians(action["degrees"]) if turning else action["distance_m"]
        if not turning:
            target = [origin[0] + signed * desired * math.cos(yaw), origin[1] + signed * desired * math.sin(yaw)]
            # Known-map rear safety in simulation; this is NOT rear visual sensing.
            for i in range(21):
                point = [a + (b - a) * i / 20 for a, b in zip(origin, target)]
                if not inside(point, self.scene) or clearance(point, self.scene) < 0.04:
                    raise NeedsUserInput("requested translation is unsafe in known map")
        while True:
            state = self.healthy()
            displacement = [a - b for a, b in zip(state["trunk"][:2], origin)]
            yaw_delta = angle(state["yaw"] - yaw)
            progress = signed * yaw_delta if turning else signed * (displacement[0] * math.cos(yaw) + displacement[1] * math.sin(yaw))
            trace.write(json.dumps({"sim_time": state["sim_time"], "position": state["trunk"][:2], "yaw": state["yaw"], "progress": progress}) + "\n")
            if progress >= desired - (math.radians(4) if turning else 0.015):
                self.robot.stop()
                time.sleep(0.25)
                end = self.healthy()
                delta = [a - b for a, b in zip(end["trunk"][:2], origin)]
                achieved = signed * angle(end["yaw"] - yaw) if turning else signed * (delta[0] * math.cos(yaw) + delta[1] * math.sin(yaw))
                lateral = abs(-delta[0] * math.sin(yaw) + delta[1] * math.cos(yaw))
                success = abs(achieved - desired) <= (math.radians(12) if turning else 0.05)
                if not turning:
                    success &= lateral < 0.08 and abs(angle(end["yaw"] - yaw)) < 0.35
                return {"success": bool(success), "achieved": achieved, "requested": desired,
                        "units": "radians" if turning else "meters", "lateral_m": lateral,
                        "start": {"position": origin, "yaw": yaw}, "end": {"position": end["trunk"][:2], "yaw": end["yaw"]}}
            elapsed = time.monotonic() - tick
            if elapsed > 15 or (elapsed > 4 and progress < desired * 0.05):
                raise TimeoutError("skill timeout/no progress; stopped")
            if not turning and (abs(yaw_delta) > 0.45 or math.hypot(*displacement) > desired + 0.12):
                raise RuntimeError("translation drift; stopped")
            if turning:
                self.robot.move(0, signed * (1.5 if desired - progress > 0.35 else 1.2))
            else:
                correction = -math.copysign(1.2, yaw_delta) if abs(yaw_delta) > 0.12 else 0
                self.robot.move(0.30 if signed > 0 else -self.backward_command, correction)
            time.sleep(0.08)

    def _return_to_pose(self, home, output):
        plan = {"status": "ok", "goals": [{"target": "point", "x": home["x"], "y": home["y"], "speed": "slow", "stop_distance_m": 0.10}]}
        here = self.healthy()["trunk"][:2]
        # 60 s covers the small arenas; a cross-world return needs time in proportion to the trip.
        budget = max(60, 30 + 12 * math.dist(here, (home["x"], home["y"])))
        result = {"navigation": execute(plan, self.robot, self.observer, output.with_suffix(".jsonl"), timeout_s=budget,
                  scene=self.scene, cancel_file=self.cancel_file, arrival_tolerance_m=0, cancel_started_ns=self.started_ns)}
        delta = angle(home["yaw"] - self.healthy()["yaw"])
        if abs(delta) > math.radians(5):
            with output.with_suffix(".turn.jsonl").open("w") as trace:
                for _ in range(2):
                    delta = angle(home["yaw"] - self.healthy()["yaw"])
                    if abs(delta) <= math.radians(5):
                        break
                    self._bounded({"action": "turn", "direction": "left" if delta > 0 else "right",
                                   "degrees": min(120, abs(math.degrees(delta)))}, trace)
        state = self.healthy()
        result.update(distance_home_m=math.dist(state["trunk"][:2], [home["x"], home["y"]]),
                      home_yaw_error_deg=abs(math.degrees(angle(state["yaw"] - home["yaw"]))))
        result["success"] = result["navigation"]["success"] and result["distance_home_m"] <= 0.125 and result["home_yaw_error_deg"] < 12
        return result

    def run(self, action, candidates=None):
        validate_action(action)
        name = action["action"]
        self.index += 1
        output = self.directory / f"skill-{self.index:03d}.json"
        result = {"action": action, "success": False, "observation": "oracle robot odometry and known safety map"}
        try:
            self.robot.stop()
            if name in ("stop", "done", "clarify", "unsupported"):
                result.update(success=name == "stop", no_motion=True)
            elif name in ("move", "turn"):
                with output.with_suffix(".jsonl").open("w") as trace:
                    result.update(self._bounded(action, trace))
            elif name == "return_home":
                home = self.info.get("home_pose")
                if home is None:
                    raise RuntimeError("home was not recorded at simulator startup; restart simulator")
                result.update(self._return_to_pose(home, output))
            elif name == "bypass_return":
                from .bypass import run_bypass
                result.update(run_bypass(self, action, output))
            elif name == "approach":
                selected = bind_target(action, candidates, self.healthy()["trunk"][:2])
                result["bound_target"] = selected
                target = selected["world_position"]
                if not visibility_check(capture(self.info["port"]), target, action["color"])["safe"]:
                    raise NeedsUserInput("target no longer safely visible")
                distance = 0.45 if selected.get("kind") == "pillar" else 0.30
                plan = {"status": "ok", "goals": [{"target": "point", "x": target[0], "y": target[1], "speed": "slow", "stop_distance_m": distance}]}
                result["navigation"] = execute(plan, self.robot, self.observer, output.with_suffix(".jsonl"), timeout_s=40, scene=self.scene, cancel_file=self.cancel_file,
                    guard=VisualGuard(self.info["port"], target, action["color"], self.directory))
                result["success"] = result["navigation"]["success"]
            elif name == "explore":
                from .explore import run_explore
                result.update(run_explore(self, action, output))
            elif name == "search":
                result["turns"] = []
                for index in range(9):
                    found = [c for c in propose_targets(capture(self.info["port"])) if c["color"] == action["color"] and approachable(c)]
                    if found:
                        result["success"] = True
                        break
                    if index == 8:
                        break
                    with output.with_suffix(".jsonl").open("a") as trace:
                        result["turns"].append(self._bounded({"action": "turn", "direction": "left", "degrees": 45}, trace))
                    if not result["turns"][-1]["success"]:
                        raise RuntimeError("search turn failed")
                result["found"] = result["success"]
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            if isinstance(error, NeedsUserInput):
                result.update(needs_user_input=True, no_motion=True,
                              retry_safe=name in ("move", "turn"))
        finally:
            try:
                self.robot.stop()
            finally:
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
