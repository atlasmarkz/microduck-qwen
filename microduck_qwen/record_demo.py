"""Record real model inference and an isolated physics run for the README."""

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

from . import ROOT
from .model import Planner
from .navigation import Observer, Robot, execute
from .sim_native import Commands, ControlHandler, DEFAULT_POLICY, DEFAULT_RL, Gait
from .sim_body import configure_body

INSTRUCTION = "绕开障碍物走到红球旁边，离球30厘米停下"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", action="store_true", help="record the trained Qwen-VL visual navigation branch")
    args = parser.parse_args()
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    visual = args.vision
    instruction = "走到画面左边的球旁边" if visual else INSTRUCTION
    output = ROOT / "assets" / ("vision-navigation-demo.gif" if visual else "navigation-demo.gif")
    run = ROOT / "runs" / ("vision-demo-recording" if visual else "demo-recording")
    run.mkdir(parents=True, exist_ok=True)
    if visual:
        from .vision import annotate, capture, propose_balls, run_instruction
        from .vision_data import digest
        from .vision_scene import make_scene
        from .vision_torch import TorchVisionPlanner

        scene_path = run / "scene.json"
        scene_path.write_text(json.dumps(make_scene(210000, "none")), encoding="utf-8")
        os.environ["MICRODUCK_VISION_SCENE"] = str(scene_path)
        os.environ.pop("MICRODUCK_COLLECT_GT", None)
    sys.path.insert(0, str(DEFAULT_RL / "src"))
    server = configure_body()
    world = server.World(server.DEFAULT_SCENE)
    body = server.Body(world, 0)
    body.place(None, server.HOME_TRUNK_Z, 0.0)
    world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    commands = Commands()
    gait = Gait(world, body, DEFAULT_POLICY, commands, server)
    if visual:
        body.camera = server.Camera(world.model, "vision_camera")
        planner = TorchVisionPlanner(ROOT / "adapters/vision-torch-static-control")
    else:
        planner = Planner(adapter=ROOT / "adapters/nav-torch")
    render_height = 300 if visual else 520
    world.model.vis.global_.offwidth, world.model.vis.global_.offheight = 680, 520
    renderer = mujoco.Renderer(world.model, height=render_height, width=680)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.55, -0.02, 0.12]
    camera.distance, camera.azimuth, camera.elevation = 2.1, 130, -30
    font = ImageFont.truetype("msyh.ttc", 17)
    small = ImageFont.truetype("msyh.ttc", 14)
    title = ImageFont.truetype("msyhbd.ttc", 27)
    listeners, frames, frame_times = [], [], []
    result, finished = {}, threading.Event()
    input_thumbnail = None

    if visual:
        predict = planner.predict

        def predict_visual(text, image, candidates):
            result["model_input"] = str(image)
            prediction = predict(text, image, candidates)
            result["prediction"] = prediction
            return prediction

        planner.predict = predict_visual

    def navigate():
        robot = observer = None
        try:
            if visual:
                command = run_instruction(planner, instruction,
                    {"socket": list(control.server_address), "port": observation.server_address[1],
                        "scene_file": str(scene_path)}, timeout=40)
                result["command"] = command
                result["execution"] = command.get("execution", {})
                return
            prediction = planner.predict(instruction)
            result["prediction"] = {k: v for k, v in prediction.items() if k != "adapter"}
            robot = Robot(list(control.server_address))
            observer = Observer(observation.server_address[1])
            result["execution"] = execute(prediction["plan"], robot, observer,
                run / "trajectory.jsonl", timeout_s=40)
        except BaseException as error:
            result["error"] = str(error)
        finally:
            if robot is not None:
                try:
                    robot.stop()
                finally:
                    robot.close()
            if observer is not None:
                observer.close()
            finished.set()

    def frame(elapsed):
        nonlocal input_thumbnail
        renderer.update_scene(world.data, camera=camera)
        canvas = Image.new("RGB", (1040, 560), "#0c1424")
        canvas.paste(Image.fromarray(renderer.render()), (350, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 24), "MICRODUCK × QWEN-VL" if visual else "MICRODUCK × QWEN", font=title if not visual else font, fill="#f0f6ff")
        draw.text((24, 66), "第一视角选球 → 闭环导航" if visual else "中文指令 → 真实物理导航", font=font, fill="#9aabc7")
        draw.line((24, 110, 322, 110), fill="#27344d", width=2)
        draw.text((24, 132), "01  输入中文指令", font=font, fill="#61c9ff")
        draw.text((24, 168), instruction if visual else "绕开障碍物走到红球旁边，", font=font, fill="white")
        draw.text((24, 196), "输入：标号 RGB + 中文指令" if visual else "离球30厘米停下", font=small if visual else font, fill="white")
        draw.text((24, 248), "02  Qwen3-VL-2B + LoRA" if visual else "02  Qwen3-0.6B + LoRA", font=font, fill="#61c9ff")
        prediction = result.get("prediction")
        if prediction and visual:
            choice = prediction["visual_choice"]
            lines = [f"status: {choice['status']}", f"color: {choice['color']}", f"candidate_id: {choice['id']}"]
        elif prediction:
            goals = prediction["plan"]["goals"]
            goal = goals[0] if goals else {}
            lines = [f"target: {goal.get('target', prediction['plan']['status'])}",
                     f"speed: {goal.get('speed', '—')}",
                     f"stop_distance: {goal.get('stop_distance_m', '—')} m"]
        else:
            lines = ["正在看图选择目标…" if visual and elapsed >= 1.5 else "正在生成 JSON 计划…" if elapsed >= 1.5 else "准备输入指令…"]
        for index, line in enumerate(lines):
            draw.text((24, 284 + index * 26), line, font=small, fill="#e3ebf7")
        draw.text((24, 390), "03  A* 路径 + 闭环控制", font=font, fill="#61c9ff")
        if visual:
            observation_frame = capture(observation.server_address[1])
            candidates = propose_balls(observation_frame)
            live = annotate(observation_frame, candidates).resize((320, 180))
            if input_thumbnail is None and result.get("model_input"):
                with Image.open(result["model_input"]) as source:
                    input_thumbnail = source.convert("RGB").resize((320, 180))
            canvas.paste(input_thumbnail or live, (350, 350))
            canvas.paste(live, (700, 350))
            draw.text((350, 326), "模型输入（固定编号）", font=small, fill="#9aabc7")
            draw.text((700, 326), "实时第一视角（编号随帧更新）", font=small, fill="#9aabc7")
            draw.text((24, 428), "RGB-D 定位 · 0.30 m 停靠", font=font, fill="white")
        else:
            distance = np.linalg.norm(world.data.qpos[body.trunk:body.trunk + 2] - [1.2, 0.25])
            draw.text((24, 428), f"距红球中心：{distance:.3f} m", font=font, fill="white")
        execution = result.get("execution", {})
        status = "到达目标 · 已停车" if execution.get("success") else "正在执行导航" if prediction else "等待模型计划"
        draw.rounded_rectangle((24, 470, 322, 510), radius=8, fill="#163d33" if execution.get("success") else "#1c2d49")
        draw.text((38, 478), status, font=font, fill="#90edc2" if execution.get("success") else "#d8e5fb")
        draw.text((350, 542), "真实 Qwen-VL 推理 · 官方步态 · 仿真位姿与安全地图" if visual else "实时 MuJoCo 渲染 · 官方 ONNX 步态 · 仿真位姿与已知地图", font=small, fill="#9aabc7")
        return canvas

    try:
        observation = server.Server(("127.0.0.1", 0), server.Handler)
        observation.body = body
        listeners.append(observation)
        control = server.Server(("127.0.0.1", 0), ControlHandler)
        control.commands = commands
        listeners.append(control)
        for listener in listeners:
            threading.Thread(target=listener.serve_forever, daemon=True).start()
        body.set_torque(True)
        for _ in range(200):
            gait.update()
            world.step(4)
        if visual:
            body.camera.render(world)
        worker = None
        started = tick = time.monotonic()
        finished_at = None
        steps = 0
        palette = None
        while time.monotonic() - started < 60:
            elapsed = time.monotonic() - started
            if elapsed >= 1.5 and worker is None:
                worker = threading.Thread(target=navigate, daemon=True)
                worker.start()
            gait.update()
            world.step(4)
            if steps % 5 == 0:
                if visual:
                    body.camera.render(world)
                rgb = frame(elapsed)
                if palette is None:
                    palette = rgb.quantize(colors=240, dither=Image.Dither.NONE)
                    colors = palette.getpalette()
                    colors[720:726] = [22, 61, 51, 144, 237, 194]
                    palette.putpalette(colors)
                frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
                frame_times.append(time.monotonic())
            if finished.is_set():
                finished_at = finished_at or time.monotonic()
                if time.monotonic() - finished_at >= 2:
                    break
            tick += 0.02
            time.sleep(max(0, tick - time.monotonic()))
            steps += 1
        if not finished.is_set() or not result.get("execution", {}).get("success"):
            raise RuntimeError(f"Recording did not complete navigation: {result}")
        assert len(frames) > 50
        durations = [max(20, round((b - a) * 100) * 10) for a, b in zip(frame_times, frame_times[1:])] + [100]
        output.parent.mkdir(exist_ok=True)
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
        with Image.open(output) as gif:
            assert gif.is_animated and gif.n_frames > 50 and gif.size == (1040, 560)
            gif_frames = gif.n_frames
            encoded_duration = 0
            for index in range(gif_frames):
                gif.seek(index)
                encoded_duration += gif.info["duration"]
            assert abs(encoded_duration - sum(durations)) <= 10
        for index, name in ((0, "start"), (len(frames) // 2, "middle"), (-1, "end")):
            frames[index].save(run / f"{name}.png")
        result.update(nominal_fps=10, duration_s=sum(durations) / 1000, frame_count=gif_frames,
            captured_frame_count=len(frames), gif=output.relative_to(ROOT).as_posix(),
            note="Live model inference and live MuJoCo physics; text is a status overlay. Frame durations preserve wall-clock capture intervals, rounded to GIF's 10 ms resolution.")
        if visual:
            assert result["command"]["success"] and result["command"]["semantic_check"]["safe"]
            result.update(mode="vision", seed=210000, instruction=instruction, collection_ground_truth=False,
                model_input_sha256=digest(Path(result["model_input"])), gif_sha256=digest(output))
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        public = json.dumps(result, ensure_ascii=False, indent=2).replace(str(ROOT).replace("\\", "\\\\"), ".").replace(ROOT.as_posix(), ".")
        (output.with_suffix(".json")).write_text(public + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"GIF: {output.stat().st_size / 1024 / 1024:.2f} MiB", flush=True)
    finally:
        commands.set({})
        for listener in listeners:
            listener.shutdown()
            listener.server_close()
        renderer.close()
        if visual:
            body.camera.renderer.close()


if __name__ == "__main__":
    main()
