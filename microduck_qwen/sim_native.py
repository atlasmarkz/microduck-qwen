"""Native MuJoCo viewer with the official ONNX gait and BAM motors; no robotd."""

import argparse
import json
import math
import os
from pathlib import Path
import socketserver
import sys
import threading
import time

from . import ROOT

POLICY_REVISION = "665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448"
DEFAULT_RL = ROOT / "runs/upstream/microduck_rl-develop"
DEFAULT_POLICY = ROOT / ".cache/hub/models--pollen-robotics--microduck-policies/snapshots" / POLICY_REVISION / "alpha_walking.onnx"


class Commands:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = (0.0, 0.0, 0.0)
        self.updated = 0.0

    def set(self, params):
        if not isinstance(params, dict) or set(params) - {"vx", "vy", "vyaw"}:
            raise ValueError("Expected vx, vy, vyaw")
        values = tuple(params.get(k, 0.0) for k in ("vx", "vy", "vyaw"))
        if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > limit
               for v, limit in zip(values, (0.4, 0.0, 1.5))):
            raise ValueError("Velocity command outside simulation limits")
        with self.lock:
            self.value, self.updated = values, time.monotonic()

    def get(self):
        with self.lock:
            return self.value if time.monotonic() - self.updated < 0.5 else (0.0, 0.0, 0.0)


class ControlHandler(socketserver.StreamRequestHandler):
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
                else:
                    raise ValueError("Only robot.move and robot.stop are supported")
                answer = {"jsonrpc": "2.0", "id": request.get("id"), "result": {"ok": True}}
            except (ValueError, TypeError) as error:
                self.server.commands.set({})
                answer = {"error": str(error)}
            if not isinstance(request, dict) or "id" in request or "error" in answer:
                self.wfile.write((json.dumps(answer) + "\n").encode())
                self.wfile.flush()


class Gait:
    def __init__(self, world, body, policy, commands, server):
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        names = [world.model.actuator(i).name for i in body.actuators]
        if metadata.get("joint_names", "").split(",") != names or self.session.get_inputs()[0].shape != [1, 61]:
            raise ValueError("Policy joint order or 61D observation contract does not match model")
        if (metadata.get("observation_names") != "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"
                or float(metadata.get("action_scale", "nan")) != 1.0):
            raise ValueError("Policy observation order or action scale does not match this runner")
        self.world, self.body, self.commands, self.server = world, body, commands, server
        self.home = np.array([server.HOME_POSE[i] for i in body.to_wire], dtype=np.float32)
        self.last_action = np.zeros(14, dtype=np.float32)
        self.gyro = world.model.sensor("imu_ang_vel").adr[0]

    def update(self):
        import numpy as np

        world, body, data = self.world, self.body, self.world.data
        with world.lock:
            gravity = self.server.gravity_in_trunk(data.qpos[body.trunk + 3:body.trunk + 7])
            if data.qpos[body.trunk + 2] < 0.085 or gravity[2] > -0.65:
                raise RuntimeError("Robot fell or became unstable; restart the simulator")
            command = np.zeros(13, dtype=np.float32)
            command[:3] = self.commands.get()
            obs = np.concatenate((data.sensordata[self.gyro:self.gyro + 3], gravity,
                data.qpos[body.qpos_adr] - self.home, data.qvel[body.qvel_adr], self.last_action, command)).astype(np.float32)
            if obs.shape != (61,) or not np.isfinite(obs).all():
                raise RuntimeError("Invalid policy observation")
            action = self.session.run(None, {self.session.get_inputs()[0].name: obs[None]})[0][0]
            if action.shape != (14,) or not np.isfinite(action).all():
                raise RuntimeError("Invalid policy action")
            self.last_action = action.copy()
            world.bam.q_target[:] = self.home + action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--rl", type=Path, default=DEFAULT_RL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--port", type=int, default=7821)
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--collect-gt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occlusion", choices=["none", "partial", "full", "absent"], default="none")
    parser.add_argument("--clutter", action="store_true", help="duplicate ball colors and add side pillars")
    parser.add_argument("--varied-colors", action="store_true", help="draw ball colours from red/blue/green/yellow instead of a fixed trio")
    parser.add_argument("--yellow-course", action="store_true", help="two/three yellow obstacles with room to pass and return")
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/simulator.json")
    parser.add_argument("--seconds", type=float, default=0, help="bounded smoke run; 0 keeps running")
    args = parser.parse_args()
    if args.yellow_course and (not args.vision or args.clutter or args.occlusion != "none"):
        parser.error("--yellow-course requires --vision and no ball occlusion/clutter flags")
    args.control_port = args.control_port or args.port + 10
    if args.collect_gt and not args.vision:
        parser.error("--collect-gt requires --vision")
    if not args.rl.is_dir() or not args.policy.is_file():
        parser.error("Missing upstream source or policy; see README Windows simulation setup")
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    import mujoco
    import numpy as np
    from .sim_body import configure_body

    sys.path.insert(0, str(args.rl / "src"))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    scene_path = None
    if args.vision:
        from .vision_scene import make_scene

        scene_path = args.manifest.with_suffix(".scene.json")
        scene_path.write_text(json.dumps(make_scene(args.seed, args.occlusion, args.clutter, args.yellow_course, args.varied_colors), indent=2), encoding="utf-8")
        os.environ["MICRODUCK_VISION_SCENE"] = str(scene_path.resolve())
        if args.collect_gt:
            os.environ["MICRODUCK_COLLECT_GT"] = "1"
        else:
            os.environ.pop("MICRODUCK_COLLECT_GT", None)
    server = configure_body()
    world = server.World(server.DEFAULT_SCENE)
    body = server.Body(world, 0)
    body.place(None, server.HOME_TRUNK_Z, 0.0)
    world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    commands = Commands()
    gait = Gait(world, body, args.policy, commands, server)
    if args.vision:
        body.camera = server.Camera(world.model, "vision_camera")
    servers, viewer = [], None
    manifest = args.manifest
    ready = False
    try:
        observer_server = server.Server(("127.0.0.1", args.port), server.Handler)
        observer_server.body = body
        servers.append(observer_server)
        control_server = server.Server(("127.0.0.1", args.control_port), ControlHandler)
        control_server.commands = commands
        servers.append(control_server)
        if args.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(world.model, world.data)
            with viewer.lock():
                viewer.cam.lookat[:] = [0.5, 0.0, 0.12]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 135, -25
        for listener in servers:
            threading.Thread(target=listener.serve_forever, daemon=True).start()
        body.set_torque(True)
        started = time.monotonic()
        tick = started
        next_image = 0.0
        while args.seconds == 0 or time.monotonic() - started < args.seconds:
            if viewer is not None and not viewer.is_running():
                break
            gait.update()
            world.step(4)
            if body.camera is not None and time.monotonic() >= next_image:
                body.camera.render(world)
                next_image = time.monotonic() + 0.2
            if viewer is not None:
                viewer.sync()
            if not ready and world.data.time >= 4.0:
                sensor = body.sensors()
                if sensor["trunk_z"] < 0.095 or sensor["imu"]["gravity"][2] > -0.9:
                    raise RuntimeError("Robot did not stabilize")
                manifest.parent.mkdir(exist_ok=True)
                w, x, y, z = sensor["imu"]["quat"]
                manifest.write_text(json.dumps({"backend": "native-onnx-bam", "port": args.port,
                    "socket": ["127.0.0.1", args.control_port], "viewer": args.viewer,
                    "vision": args.vision, "seed": args.seed, "occlusion": args.occlusion,
                    "clutter": args.clutter,
                    "varied_colors": args.varied_colors,
                    "yellow_course": args.yellow_course,
                    "scene_file": str(scene_path.resolve()) if scene_path else None,
                    "collection_ground_truth": args.collect_gt,
                    "policy": args.policy.resolve().relative_to(ROOT).as_posix(),
                    "home_pose": {"x": sensor["trunk"][0], "y": sensor["trunk"][1],
                        "yaw": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))},
                    "note": "Simulation-only TCP controller; official ONNX gait and BAM motors; no robotd."}, indent=2) + "\n", encoding="utf-8")
                ready = True
                print("READY: native MuJoCo simulation; run CLI commands in a second terminal.", flush=True)
            tick += 0.02
            time.sleep(max(0.0, tick - time.monotonic()))
        print(json.dumps({"ready": ready, "sim_time": world.data.time, "trunk": body.sensors()["trunk"]}), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        commands.set({})
        if viewer is not None:
            viewer.close()
        if body.camera is not None:
            body.camera.renderer.close()
        for listener in servers:
            listener.server_close()
        if ready:
            manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
