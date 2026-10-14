"""Run an isolated official MuJoCo body and robotd. Ctrl+C stops only these child processes."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

from . import ROOT
from .navigation import Observer, Robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--occlusion", choices=["none", "partial", "full", "absent"], default="none"
    )
    parser.add_argument(
        "--manifest", type=Path, help="independent instance metadata path"
    )
    parser.add_argument(
        "--microduck", type=Path, default=Path("/Users/mark/Code/microduck")
    )
    parser.add_argument(
        "--rl", type=Path, default=Path("/Users/mark/Code/microduck_rl")
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=Path("/Users/mark/Code/duck-sim-state/policies/current"),
    )
    parser.add_argument("--walking-policy", default="alpha_walking.onnx")
    args = parser.parse_args()
    args.port = args.port or (7822 if args.vision else 7821)
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", args.port))
    binary = args.microduck / "target/debug/robotd"
    interpreter = args.rl / f".venv/bin/{'mjpython' if args.viewer else 'python'}"
    ort = list(
        (args.rl / ".venv").glob(
            "lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.dylib"
        )
    )
    if not binary.exists() or not interpreter.exists() or not ort:
        raise FileNotFoundError(
            "需要已构建的 microduck、microduck_rl 仿真环境和 ONNX Runtime；参见 README。"
        )
    policies = {
        "walk": args.walking_policy,
        "sitstand": "alpha_sitstand.onnx",
        "ground_pick": "alpha_ground_pick.onnx",
        "kick_left": "ball_kick_left.onnx",
        "kick_right": "ball_kick_right.onnx",
        "roulade": "roulade.onnx",
    }
    for filename in policies.values():
        if not (args.policies / filename).exists():
            raise FileNotFoundError(args.policies / filename)
    state = Path(tempfile.mkdtemp(prefix="mdq-", dir="/private/tmp"))
    socket_path = state / "duck.sock"
    params = state / "robotd.toml"
    # Match the installed actor's proven infer_policy.py execution, not robotd's
    # real-hardware defaults (0.9 action scale and extra joint smoothing).
    params.write_text(
        '[policy]\nenabled = true\nstand = "none"\naction_scale = 1.0\nhead_lowpass = 1.0\nlegs_lowpass = 1.0\n'
        + "".join(
            f'{key} = "{args.policies / file}"\n' for key, file in policies.items()
        )
        + "\n[audio]\nenabled = false\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT) + os.pathsep + str(args.rl / "src"),
        DUCK_RUNTIME_DIR=str(state),
        ORT_DYLIB_PATH=str(ort[0]),
    )
    scene_path = None
    if args.vision:
        from .vision_scene import make_scene

        scene_path = state / "scene.json"
        scene_path.write_text(
            json.dumps(make_scene(args.seed, args.occlusion), indent=2) + "\n"
        )
        env["MICRODUCK_VISION_SCENE"] = str(scene_path)
    command = [
        str(interpreter),
        "-u",
        "-m",
        "microduck_qwen.sim_body",
        "--port",
        str(args.port),
        "--keyframe",
        "HOME" if args.vision else "SIT",
    ]
    if args.vision:
        command += [
            "--cameras",
            "a",
            "--camera-fps",
            "5",
            "--frame-port",
            str(args.port + 100),
        ]
    if not args.viewer:
        command.append("--headless")
    children, logs = [], []

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        body_log = (state / "body.log").open("w")
        logs.append(body_log)
        children.append(
            subprocess.Popen(
                command, env=env, stdout=body_log, stderr=subprocess.STDOUT
            )
        )
        observer = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if children[0].poll() is not None:
                raise RuntimeError((state / "body.log").read_text()[-5000:])
            try:
                observer = Observer(args.port)
                break
            except OSError:
                time.sleep(0.1)
        if observer is None:
            raise TimeoutError(f"MuJoCo 未启动；日志 {state / 'body.log'}")
        if args.vision:
            # Warm up the OpenGL renderer while the robot is still held. The body
            # TCP listener is available before camera initialization completes;
            # enabling torque during that pause can spoil the sit-to-stand policy.
            camera_deadline = time.monotonic() + 20
            while time.monotonic() < camera_deadline:
                try:
                    observer.connection.send({"op": "vision"})
                    observer.connection.receive()
                    break
                except (RuntimeError, OSError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("视觉相机没有完成预热")
        robot_log = (state / "robotd.log").open("w")
        logs.append(robot_log)
        children.append(
            subprocess.Popen(
                [
                    str(binary),
                    "--sim",
                    f"127.0.0.1:{args.port}",
                    "--params",
                    str(params),
                    "--socket",
                    str(socket_path),
                ],
                env=env,
                stdout=robot_log,
                stderr=subprocess.STDOUT,
            )
        )
        deadline = time.monotonic() + 15
        while not socket_path.exists() and time.monotonic() < deadline:
            if children[-1].poll() is not None:
                raise RuntimeError((state / "robotd.log").read_text()[-5000:])
            time.sleep(0.1)
        robot = Robot(socket_path)
        outcome = robot.call("robot.enable", {"on": True})
        enabled_at = time.monotonic()
        print(f"Enable: {outcome}", flush=True)
        deadline = time.monotonic() + 15
        stable_since = None
        while time.monotonic() < deadline:
            sensor = observer.read()
            if sensor["trunk_z"] > 0.095 and sensor["imu"]["gravity"][2] < -0.9:
                stable_since = stable_since or time.monotonic()
                if (
                    time.monotonic() - stable_since > 3
                    and time.monotonic() - enabled_at > 4.5
                ):
                    break
            else:
                stable_since = None
            time.sleep(0.2)
        else:
            raise RuntimeError(f"机器人未站稳：{sensor}；日志 {state}")
        observer.close()
        robot.close()
        if args.vision:
            setup = Observer(args.port)
            setup.connection.send({"op": "vision_face_forward"})
            setup.connection.receive()
            time.sleep(1.0)
            setup.close()
        metadata = {
            "socket": str(socket_path),
            "port": args.port,
            "state_dir": str(state),
            "manager_pid": os.getpid(),
            "viewer": args.viewer,
            "body_pid": children[0].pid,
            "robotd_pid": children[1].pid,
            "vision": args.vision,
            "scene_file": str(scene_path) if scene_path else None,
            "seed": args.seed if args.vision else None,
            "occlusion": args.occlusion if args.vision else None,
            "startup": "held HOME until balancing policy ready"
            if args.vision
            else "sitstand",
        }
        (ROOT / "runs").mkdir(exist_ok=True)
        manifest = "vision-simulator.json" if args.vision else "simulator.json"
        manifest_path = (
            ROOT / args.manifest if args.manifest else ROOT / "runs" / manifest
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"READY: {socket_path}\n{'随机视觉场景 seed=' + str(args.seed) if args.vision else '默认红球/蓝标场景'}\nCtrl+C 关闭本次仿真。",
            flush=True,
        )
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping this simulator.", flush=True)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
