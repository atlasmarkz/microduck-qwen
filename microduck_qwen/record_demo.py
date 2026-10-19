"""Record the current skill agent and real MuJoCo physics for the README."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

from . import ROOT
from .sim_native import Commands, ControlHandler, DEFAULT_POLICY, DEFAULT_RL, Gait
from .sim_body import configure_body
from .sim_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapters/agent-hard-v3-sft300")
    args = parser.parse_args()
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from .agent import AgentPlanner, run_instruction
    from .vision import annotate, capture, propose_balls
    from .vision_scene import make_scene

    instruction = "前进0.2米，再回到初始位置"
    run = ROOT / "runs/agent-demo-recording" / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    scene = run / "scene.json"
    scene.write_text(json.dumps(make_scene(230000, "none", True)), encoding="utf-8")
    os.environ["MICRODUCK_VISION_SCENE"] = str(scene)
    os.environ.pop("MICRODUCK_COLLECT_GT", None)
    planner = AgentPlanner(args.adapter)
    sys.path.insert(0, str(DEFAULT_RL / "src"))
    server = configure_body()
    world = server.World(server.DEFAULT_SCENE)
    body = server.Body(world, 0)
    body.place(None, server.HOME_TRUNK_Z, 0.0)
    world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    commands = Commands()
    gait = Gait(world, body, DEFAULT_POLICY, commands, server)
    body.camera = server.Camera(world.model, "vision_camera")
    world.model.vis.global_.offwidth = 680
    world.model.vis.global_.offheight = 520
    renderer = mujoco.Renderer(world.model, height=300, width=680)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.5, 0, 0.12]
    camera.distance, camera.azimuth, camera.elevation = 2.1, 130, -30
    font = ImageFont.truetype("msyh.ttc", 17)
    small = ImageFont.truetype("msyh.ttc", 14)
    listeners, frames, times, decisions = [], [], [], []
    result, finished = {}, threading.Event()
    predict = planner.predict

    def recorded_predict(text, image, memory):
        entry = {"image": str(image), "memory": memory, "status": "inferencing"}
        decisions.append(entry)
        prediction = predict(text, image, memory)
        entry.update(status="predicted", prediction=prediction)
        return prediction

    planner.predict = recorded_predict

    def execute():
        try:
            result["episode"] = run_instruction(planner, instruction, info)
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            finished.set()

    worker = None
    try:
        observation = server.Server(("127.0.0.1", 0), server.Handler)
        observation.body = body
        control = server.Server(("127.0.0.1", 0), ControlHandler)
        control.commands = commands
        listeners.extend((observation, control))
        for listener in listeners:
            threading.Thread(target=listener.serve_forever, daemon=True).start()
        body.set_torque(True)
        for _ in range(200):
            gait.update()
            world.step(4)
        body.camera.render(world)
        sensor = body.sensors()
        w, x, y, z = sensor["imu"]["quat"]
        home = {"x": sensor["trunk"][0], "y": sensor["trunk"][1],
                "yaw": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))}
        info = {"socket": list(control.server_address), "port": observation.server_address[1],
                "scene_file": str(scene), "home_pose": home, "vision": True}
        started = tick = time.monotonic()
        ended = None
        count = 0
        max_displacement = 0.0
        while time.monotonic() - started < 90:
            elapsed = time.monotonic() - started
            if elapsed >= 1.5 and worker is None:
                worker = threading.Thread(target=execute, daemon=True)
                worker.start()
            gait.update()
            world.step(4)
            pose = body.sensors()["trunk"]
            distance = math.hypot(pose[0] - home["x"], pose[1] - home["y"])
            max_displacement = max(max_displacement, distance)
            if count % 5 == 0:
                body.camera.render(world)
                renderer.update_scene(world.data, camera=camera)
                canvas = Image.new("RGB", (1040, 560), "#0c1424")
                canvas.paste(Image.fromarray(renderer.render()), (350, 20))
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 24), "MICRODUCK × QWEN-VL", font=font, fill="white")
                draw.text((24, 65), "当前技能 agent · 真实本地运行", font=font, fill="#9aabc7")
                draw.text((24, 120), "中文指令", font=font, fill="#61c9ff")
                draw.text((24, 155), "前进0.2米，再回到初始位置", font=font, fill="white")
                draw.text((24, 205), "逐步模型决策（每步重新看图）", font=font, fill="#61c9ff")
                for i, decision in enumerate(decisions):
                    action = decision.get("prediction", {}).get("action", {})
                    label = " / ".join(str(v) for v in action.values()) if action else "看图推理中…"
                    draw.text((24, 240 + i * 33), f"{i + 1}. {label}", font=small, fill="white")
                draw.text((24, 420), f"距起点：{distance:.3f} m", font=font, fill="white")
                status = result.get("episode", {}).get("status", "执行中")
                if status == "model_done" and not any(d.get("prediction", {}).get("action", {}).get("action") == "return_home" for d in decisions):
                    status = "提前结束：未执行回家"
                draw.text((24, 454), f"状态：{status}", font=small, fill="#90edc2")
                draw.text((24, 487), "done 仅为模型声明，非成功认证", font=small, fill="#ffc778")
                live_frame = capture(observation.server_address[1])
                live = annotate(live_frame, propose_balls(live_frame)).resize((320, 180))
                model_image = live
                if decisions:
                    with Image.open(decisions[-1]["image"]) as image:
                        model_image = image.convert("RGB").resize((320, 180))
                canvas.paste(model_image, (350, 350))
                canvas.paste(live, (700, 350))
                draw.text((350, 326), "本步实际模型输入", font=small, fill="#9aabc7")
                draw.text((700, 326), "实时第一视角（编号随帧变化）", font=small, fill="#9aabc7")
                draw.text((350, 540), "Qwen3-VL-2B LoRA · 官方步态 · 仿真位姿与已知地图", font=small, fill="#9aabc7")
                frames.append(canvas.quantize(colors=192, dither=Image.Dither.NONE))
                times.append(time.monotonic())
            if finished.is_set():
                ended = ended or time.monotonic()
                if time.monotonic() - ended > 2:
                    break
            tick += 0.02
            time.sleep(max(0, tick - time.monotonic()))
            count += 1
        if not finished.is_set() or result.get("error"):
            raise RuntimeError(f"recording failed: {result}")
        worker.join(timeout=1)
        assert decisions and len(frames) > 30
        durations = [max(20, round((b - a) * 100) * 10) for a, b in zip(times, times[1:])] + [100]
        output = ROOT / "assets/agent-demo.gif"
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
        with Image.open(output) as gif:
            assert gif.is_animated and gif.n_frames > 30 and gif.size == (1040, 560)
            encoded_ms = 0
            for i in range(gif.n_frames):
                gif.seek(i)
                encoded_ms += gif.info["duration"]
            assert abs(encoded_ms - sum(durations)) <= 10
            result["frame_count"] = gif.n_frames
        for i, name in ((0, "start"), (len(frames) // 2, "middle"), (-1, "end")):
            frames[i].save(run / f"{name}.png")
        result.update(instruction=instruction, seed=230000, clutter=True, occlusion="none",
                      adapter=planner.adapter, decisions=decisions, duration_s=encoded_ms / 1000,
                      max_displacement_m=max_displacement, final_home_distance_m=distance,
                      gif_sha256=digest(output), collection_ground_truth=False,
                      note="Live model inference and physics; model_done is not independently verified task success. No teacher intervention. Wall-clock frame timing preserved.")
        public = json.dumps(result, ensure_ascii=False, indent=2)
        public = public.replace(str(ROOT).replace("\\", "\\\\"), ".").replace(ROOT.as_posix(), ".")
        output.with_suffix(".json").write_text(public + "\n", encoding="utf-8")
        (run / "result.json").write_text(public + "\n", encoding="utf-8")
        print(public, flush=True)
    finally:
        commands.set({})
        if worker and worker.is_alive():
            (ROOT / "runs/cancel_navigation").write_text(str(time.time_ns()))
            worker.join(timeout=15)
        for listener in listeners:
            listener.shutdown()
            listener.server_close()
        renderer.close()
        body.camera.renderer.close()


if __name__ == "__main__":
    main()
