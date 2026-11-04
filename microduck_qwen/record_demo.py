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
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapters/agent-4b-v1-r0")
    parser.add_argument("--yellow-course", action="store_true", help="record ordered yellow-obstacle passing and return")
    parser.add_argument("--world", action="store_true", help="record find-a-ball exploration in a 2x2 multi-room world")
    parser.add_argument("--world-seed", type=int, default=98000)
    parser.add_argument("--color", default="蓝色", help="--world: colour phrase to look for")
    parser.add_argument("--speed", type=int, default=5, help="--world: playback speed-up of the GIF (labelled on screen)")
    args = parser.parse_args()
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from .agent import AgentPlanner, run_instruction
    from .vision_torch import MODEL
    from .vision import annotate, capture, propose_targets
    from .vision_scene import make_scene

    instruction = ("依次绕过黄色障碍物并回到原位" if args.yellow_course else
                   f"去别的房间找{args.color}的球，走到它旁边，然后回起点" if args.world else "前进0.2米，再回到初始位置")
    seed = args.world_seed if args.world else 230001 if args.yellow_course else 230000
    run = ROOT / "runs/agent-demo-recording" / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    scene = run / "scene.json"
    if args.world:
        from .vision_world import make_world
        scene_data = make_world(seed, (2, 2), 1.0)
    else:
        scene_data = make_scene(seed, "none", not args.yellow_course, args.yellow_course)
    scene.write_text(json.dumps(scene_data), encoding="utf-8")
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
    if args.yellow_course:
        camera.lookat[:] = [1.3, 0, 0.12]
        camera.distance, camera.elevation = 3.8, -45
    if args.world:
        xs = [v for w in scene_data["walls"] for v in (w["x1"], w["x2"])]
        ys = [v for w in scene_data["walls"] for v in (w["y1"], w["y2"])]
        camera.lookat[:] = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.0]
        camera.distance, camera.azimuth, camera.elevation = 9.6, 90, -86
    font = ImageFont.truetype("msyh.ttc", 17)
    small = ImageFont.truetype("msyh.ttc", 14)
    wrenderer = mujoco.Renderer(world.model, height=338, width=600) if args.world else None
    try:
        bold = ImageFont.truetype("msyhbd.ttc", 26)
        label_font = ImageFont.truetype("msyhbd.ttc", 17)
    except OSError:
        bold, label_font = ImageFont.truetype("msyh.ttc", 26), ImageFont.truetype("msyh.ttc", 17)
    text_font = ImageFont.truetype("msyh.ttc", 22)
    chip_font = ImageFont.truetype("msyh.ttc", 20)
    note_font = ImageFont.truetype("msyh.ttc", 15)
    trail = []
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

    def world_frame(elapsed, distance, pose, port):
        """Large 16:10 layout for the multi-room demo: top-down world with trail, model input, live view, decisions."""
        body.camera.render(world)
        wrenderer.update_scene(world.data, camera=camera)
        trail.append((pose[0], pose[1]))
        import numpy as np
        cam = wrenderer.scene.camera[0]
        eye, forward, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
        right = np.cross(forward, up)
        tan = math.tan(math.radians(world.model.vis.global_.fovy) / 2)

        def project(x, y, z=0.3):  # world metres -> pixels of the 600x338 top-down render
            v = np.array([x, y, z]) - eye
            depth = v @ forward
            return 300 + (v @ right) / depth / tan * 169, 169 - (v @ up) / depth / tan * 169

        pixels = [project(px, py) for px, py in trail]
        # Only the GL render and the camera grab happen while the simulation runs; drawing is deferred.
        jobs.append((Image.fromarray(wrenderer.render()), pixels, capture(port), elapsed, distance, list(decisions), finished.is_set()))
        times.append(time.monotonic())

    def compose(top, pixels, live_frame, elapsed, distance, decisions, done):
        overlay = ImageDraw.Draw(top)
        for i, (px, py) in enumerate(pixels[:-1]):
            r = 2.5 + 2.0 * i / max(1, len(pixels))
            overlay.ellipse((px - r, py - r, px + r, py + r), fill=(39, 211, 255))
        rx, ry = pixels[-1]
        overlay.ellipse((rx - 11, ry - 11, rx + 11, ry + 11), fill=(255, 154, 47), outline=(255, 255, 255), width=3)
        canvas = Image.new("RGB", (1000, 620), "#0b1220")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 14), "MICRODUCK × QWEN-VL", font=bold, fill="white")
        chip = f"实机仿真 · {args.speed}× 倍速 · {elapsed:.0f} s"
        width = draw.textlength(chip, font=note_font) + 28
        draw.rounded_rectangle((976 - width, 20, 976, 52), 16, fill="#2a1f0b", outline="#ffc778")
        draw.text((990 - width, 25), chip, font=note_font, fill="#ffc778")
        draw.rounded_rectangle((20, 70, 980, 118), 12, fill="#14203a", outline="#2b4170")
        draw.text((36, 79), "指令", font=label_font, fill="#61c9ff")
        draw.text((100, 76), instruction, font=text_font, fill="white")
        canvas.paste(top, (20, 134))
        draw.rounded_rectangle((20, 134, 620, 472), 6, outline="#2b4170", width=2)
        draw.rounded_rectangle((26, 138, 372, 166), 8, fill="#0b1220")
        draw.text((36, 141), "俯视全景 · 橙点为机器人，蓝点为走过的路", font=note_font, fill="#c9d6ee")
        live = annotate(live_frame, propose_targets(live_frame)).resize((340, 191))
        model_image = live
        if decisions:
            with Image.open(decisions[-1]["image"]) as image:
                model_image = image.convert("RGB").resize((340, 191))
        draw.text((640, 134), "本步模型输入（带编号）", font=label_font, fill="#9aabc7")
        canvas.paste(model_image, (640, 160))
        draw.text((640, 360), "机器人第一视角", font=label_font, fill="#9aabc7")
        canvas.paste(live, (640, 386))
        for box in ((640, 160, 980, 351), (640, 386, 980, 577)):
            draw.rounded_rectangle(box, 4, outline="#2b4170", width=2)
        draw.text((20, 486), "模型决策（每步重新看图）", font=label_font, fill="#9aabc7")
        x = 20
        for i, decision in enumerate(decisions):
            action = decision.get("prediction", {}).get("action", {})
            label = " · ".join(str(v) for v in action.values()) if action else "看图推理中…"
            text = f"{i + 1}  {label}"
            chip_w = draw.textlength(text, font=chip_font) + 26
            current = i == len(decisions) - 1 and not done
            draw.rounded_rectangle((x, 516, x + chip_w, 556), 20, fill="#10304a" if current else "#14203a",
                                   outline="#61c9ff" if current else "#2b4170", width=2)
            draw.text((x + 13, 522), text, font=chip_font, fill="white")
            x += chip_w + 10
        status = "已完成" if done else "执行中"
        draw.text((20, 572), f"距起点 {distance:.2f} m   状态：{status}", font=text_font, fill="#90edc2")
        draw.text((20, 598), f"{MODEL.split('/')[-1]} + LoRA · 官方步态 · done 仅为模型声明，非成功认证", font=note_font, fill="#7d8aa6")
        # Small saturated marks (the orange robot dot) vanish in a 96-colour palette unless their colours are
        # forced in: quantize with a swatch strip appended, then crop it away.
        swatches = Image.new("RGB", (1000, 40), "#0b1220")
        keep = ("#ff9a2f", "#27d3ff", "#90edc2", "#ffc778", "#e5322d", "#2a5cff", "#2fd14f", "#ffffff")
        for k, colour in enumerate(keep):  # the ball colours (red/blue/green) must survive next to the orange robot
            swatches.paste(Image.new("RGB", (125, 40), colour), (k * 125, 0))
        extended = Image.new("RGB", (1000, 660))
        extended.paste(canvas, (0, 0))
        extended.paste(swatches, (0, 620))
        frames.append(extended.quantize(colors=96, dither=Image.Dither.NONE).crop((0, 0, 1000, 620)))

    jobs = []  # raw snapshots; all PIL work happens after the run so it cannot slow the simulation down
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
        while time.monotonic() - started < (600 if args.world else 240 if args.yellow_course else 90):
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
                body.camera.render(world)  # keep the robot camera fresh between GIF frames
            if args.world and count % 100 == 0:
                world_frame(elapsed, distance, pose, observation.server_address[1])
            elif not args.world and count % (25 if args.yellow_course else 5) == 0:
                body.camera.render(world)
                renderer.update_scene(world.data, camera=camera)
                canvas = Image.new("RGB", (1040, 560), "#0c1424")
                canvas.paste(Image.fromarray(renderer.render()), (350, 20))
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 24), "MICRODUCK × QWEN-VL", font=font, fill="white")
                draw.text((24, 65), "当前技能 agent · 真实本地运行", font=font, fill="#9aabc7")
                draw.text((24, 120), "中文指令", font=font, fill="#61c9ff")
                draw.text((24, 155), instruction, font=font, fill="white")
                draw.text((24, 205), "逐步模型决策（每步重新看图）", font=font, fill="#61c9ff")
                for i, decision in enumerate(decisions):
                    action = decision.get("prediction", {}).get("action", {})
                    label = " / ".join(str(v) for v in action.values()) if action else "看图推理中…"
                    draw.text((24, 240 + i * 33), f"{i + 1}. {label}", font=small, fill="white")
                draw.text((24, 420), f"距起点：{distance:.3f} m", font=font, fill="white")
                status = result.get("episode", {}).get("status", "执行中")
                if status == "model_done" and not any(d.get("prediction", {}).get("action", {}).get("action") in ("return_home", "bypass_return") for d in decisions):
                    status = "提前结束：未执行回家"
                draw.text((24, 454), f"状态：{status}", font=small, fill="#90edc2")
                if args.world:
                    draw.text((24, 510), f"{args.speed}× 倍速播放 · 已运行 {elapsed:.0f} s", font=small, fill="#ffc778")
                draw.text((24, 487), "done 仅为模型声明，非成功认证", font=small, fill="#ffc778")
                live_frame = capture(observation.server_address[1])
                live = annotate(live_frame, propose_targets(live_frame)).resize((320, 180))
                model_image = live
                if decisions:
                    with Image.open(decisions[-1]["image"]) as image:
                        model_image = image.convert("RGB").resize((320, 180))
                canvas.paste(model_image, (350, 350))
                canvas.paste(live, (700, 350))
                draw.text((350, 326), "本步实际模型输入", font=small, fill="#9aabc7")
                draw.text((700, 326), "实时第一视角（编号随帧变化）", font=small, fill="#9aabc7")
                draw.text((350, 540), f"{MODEL.split('/')[-1]} LoRA · 官方步态 · 仿真位姿与已知地图", font=small, fill="#9aabc7")
                frames.append(canvas.quantize(colors=192, dither=Image.Dither.NONE))
                times.append(time.monotonic())
            if finished.is_set():
                ended = ended or time.monotonic()
                if time.monotonic() - ended > 2:
                    break
            tick += 0.02
            time.sleep(max(0, tick - time.monotonic()))
            count += 1
        for job in jobs:
            compose(*job)
        if not finished.is_set() or result.get("error"):
            raise RuntimeError(f"recording failed: {result}")
        worker.join(timeout=1)
        assert decisions and len(frames) > 30
        if args.yellow_course:
            completed = [s.get("execution", {}) for s in result["episode"]["steps"]
                         if s.get("prediction", {}).get("action", {}).get("action") == "bypass_return"]
            assert len(completed) == 1 and completed[0]["success"] and completed[0]["trace_verification"]["success"], "bypass recording failed physical verification"
        speed = args.speed if args.world else 1
        if args.world:
            actions = [d.get("prediction", {}).get("action", {}).get("action") for d in decisions]
            assert "explore" in actions and "return_home" in actions and distance < 0.4, f"world recording incomplete: {actions}, {distance:.2f} m from start"
        durations = [max(20, round((b - a) * 100 / speed) * 10) for a, b in zip(times, times[1:])] + [100]
        output = ROOT / "assets" / ("explore-rooms-4b-v1.gif" if args.world else
                                    "yellow-obstacles-4b-v1.gif" if args.yellow_course else "agent-demo.gif")
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
        with Image.open(output) as gif:
            assert gif.is_animated and gif.n_frames > 30 and gif.size == frames[0].size
            encoded_ms = 0
            for i in range(gif.n_frames):
                gif.seek(i)
                encoded_ms += gif.info["duration"]
            assert abs(encoded_ms - sum(durations)) <= 10
            result["frame_count"] = gif.n_frames
        for i, name in ((0, "start"), (len(frames) // 2, "middle"), (-1, "end")):
            frames[i].save(run / f"{name}.png")
        result.update(instruction=instruction, seed=seed, clutter=not (args.yellow_course or args.world), yellow_course=args.yellow_course, occlusion="none",
                      adapter=planner.adapter, decisions=decisions, duration_s=encoded_ms / 1000,
                      max_displacement_m=max_displacement, final_home_distance_m=distance,
                      gif_sha256=digest(output), collection_ground_truth=False,
                      note="Live model inference and physics; model_done is not independently verified task success. No teacher intervention. " + (f"Playback is {args.speed}x real time (labelled on screen)." if args.world else "Wall-clock frame timing preserved."))
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
        if wrenderer:
            wrenderer.close()
        body.camera.renderer.close()


if __name__ == "__main__":
    main()
