"""sim_native with movable balls of one colour and the official kick policies (ball-play experiment).

Only this launcher changes the world: balls of the chosen colour become free bodies with the official
kick ball's 15 g mass and a rolling resistance (without one a pushed ball never stops). They keep the
scene's 7 cm radius so the perception pipeline is unchanged; the official ball is 3.5 cm. Everything
else, including the walking gait, motors, cameras and the recorded scene file, is sim_native's.

    python -m microduck_qwen.sim_ball --movable blue <sim_native arguments>
"""

import json
import os
from pathlib import Path
import sys

import numpy as np

from . import sim_body, sim_native
from .skill_showroom import fetch_policy

BALL_MASS_KG = 0.015
FRICTION = (0.5, 0.005, 0.002)  # sliding, torsional, rolling; MuJoCo uses the last only with condim 6
KICK_TICKS = 25  # 0.5 s, the official manifest's duration_s
KICKS = {"left": "ball_kick_left.onnx", "right": "ball_kick_right.onnx"}


def add_movable_balls(spec, balls):
    import mujoco

    for key in list(spec.keys):  # scene.xml's keyframes assume the robot is the only free body
        spec.delete(key)
    for name, ball in balls.items():
        body = spec.worldbody.add_body(name=f"movable_{name}", pos=[ball["x"], ball["y"], ball["radius"]])
        body.add_freejoint(name=f"free_{name}")
        body.add_geom(name=f"goal_{name}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball["radius"], 0, 0],
                      rgba=ball["rgba"], mass=BALL_MASS_KG, friction=list(FRICTION), condim=6, priority=1,
                      contype=1, conaffinity=1)


def configure(color, original=sim_body.configure_body):
    """configure_body with the chosen colour's balls taken out of the static scene and added as free bodies."""
    import mujoco

    scene_file = Path(os.environ["MICRODUCK_VISION_SCENE"])
    scene = json.loads(scene_file.read_text(encoding="utf-8"))
    movable = {n: o for n, o in scene["objects"].items() if o["color"] == color and o.get("shape") == "sphere"}
    if not movable:
        raise ValueError(f"no {color} ball in this scene")
    static_file = scene_file.with_suffix(".static.json")
    static_file.write_text(json.dumps(dict(scene, objects={n: o for n, o in scene["objects"].items() if n not in movable}),
                                      indent=2), encoding="utf-8")
    os.environ["MICRODUCK_VISION_SCENE"] = str(static_file)
    server = original()
    load = mujoco.MjSpec.from_file

    def load_with_balls(path, *args, **kwargs):
        spec = load(path, *args, **kwargs)
        add_movable_balls(spec, movable)
        return spec

    mujoco.MjSpec.from_file = staticmethod(load_with_balls)
    dispatch = server.Handler.dispatch

    def with_balls(handler, body, request):
        if request.get("op") != "balls":
            return dispatch(handler, body, request)
        world = body.world
        with world.lock:
            result = {}
            for name in movable:
                joint = world.model.joint(f"free_{name}")
                position = world.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 3]
                velocity = world.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 2]
                result[name] = [*map(float, position), *map(float, velocity)]
        return {"balls": result}

    server.Handler.dispatch = with_balls
    return server


class KickGait(sim_native.Gait):
    """The walking gait, handed over to an official kick policy for half a second on request."""

    active = None  # the one gait of this process, for the control handler

    def __init__(self, world, body, policy, commands, server):
        super().__init__(world, body, policy, commands, server)
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.walk = self.session
        self.kicks = {side: ort.InferenceSession(str(fetch_policy(name)), sess_options=options,
                                                 providers=["CPUExecutionProvider"]) for side, name in KICKS.items()}
        self.side, self.ticks = None, 0
        KickGait.active = self

    def kick(self, side):
        if side not in self.kicks:
            raise ValueError("kick side must be left or right")
        with self.world.lock:
            self.commands.set({})
            self.side, self.ticks = side, KICK_TICKS

    def update(self):
        if self.ticks > 0:
            self.commands.set({})  # kick policies were trained with an all-zero command
            self.session = self.kicks[self.side]
            self.ticks -= 1
        else:
            self.session = self.walk
        super().update()


class BallControlHandler(sim_native.ControlHandler):
    def handle(self):
        while raw := self.rfile.readline(8193):
            request = {}
            try:
                if len(raw) > 8192:
                    raise ValueError("RPC frame too large")
                request = json.loads(raw)
                if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                    raise ValueError("Expected JSON-RPC 2.0")
                if request.get("method") == "robot.move":
                    self.server.commands.set(request.get("params", {}))
                elif request.get("method") == "robot.stop":
                    self.server.commands.set({})
                elif request.get("method") == "robot.kick":
                    KickGait.active.kick(request.get("params", {}).get("side"))
                else:
                    raise ValueError("Only robot.move, robot.stop and robot.kick are supported")
                answer = {"jsonrpc": "2.0", "id": request.get("id"), "result": {"ok": True}}
            except (ValueError, TypeError) as error:
                self.server.commands.set({})
                answer = {"error": str(error)}
            if not isinstance(request, dict) or "id" in request or "error" in answer:
                self.wfile.write((json.dumps(answer) + "\n").encode())
                self.wfile.flush()


def main():
    arguments = sys.argv[1:]
    color = "blue"
    if "--movable" in arguments:
        at = arguments.index("--movable")
        color = arguments[at + 1]
        del arguments[at:at + 2]
    sys.argv = [sys.argv[0], *arguments]
    original = sim_body.configure_body
    sim_body.configure_body = lambda: configure(color, original)
    sim_native.Gait, sim_native.ControlHandler = KickGait, BallControlHandler
    sim_native.main()


if __name__ == "__main__":
    main()
