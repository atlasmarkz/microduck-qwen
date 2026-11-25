"""Colab cell for the ball-play experiment: one sentence, Qwen finds the ball, the program kicks or pushes it home."""

import base64
from concurrent.futures import ThreadPoolExecutor
import html
import io
import json
import math
from pathlib import Path
from queue import Queue
import time

from . import ROOT
from .ball_play import KICK, NONE, parse, play
from .colab_demo import REFRESH_S, AudienceFeed, Board, _close_previous_console, draw_overview, jpeg, simulation

COLORS = {"红": "red", "蓝": "blue", "绿": "green", "黄": "yellow"}
CAPTURE_S, SPEED = 0.6, 6


def ball_color(instruction):
    return next((english for chinese, english in COLORS.items() if chinese in instruction), None)


def clip_frame(spectator, overview, instruction, status, elapsed):
    from PIL import Image, ImageDraw

    from .demo_gif import load_font

    canvas = Image.new("RGB", (960, 452), "#0b1220")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), instruction, font=load_font(20, bold=True), fill="#f8fafc")
    canvas.paste(Image.open(io.BytesIO(spectator)).convert("RGB").resize((640, 360)), (0, 44))
    canvas.paste(overview.resize((316, 316)), (644, 66))
    draw.text((648, 46), "俯视图（白点为起点，球会移动）", font=load_font(13), fill="#94a3b8")
    draw.text((14, 416), status, font=load_font(18, bold=True), fill="#38bdf8")
    draw.text((828, 418), f"{SPEED}× · {elapsed:.0f}s", font=load_font(14), fill="#94a3b8")
    return canvas


def ball_demo(planner, instruction, seed):
    """Run one episode with a live view; returns {"find", "play"} and embeds a small clip in the output."""
    from IPython.display import HTML, display

    from .agent import run_instruction
    from .demo_gif import save_demo_webp
    from .navigation import Observer
    from .sim_ball import KICKS, fetch_policy

    instruction = instruction.strip()
    head, verb = parse(instruction)
    color = ball_color(instruction)
    if verb is NONE or color is None:
        raise ValueError("请在指令里写明球的颜色，并包含“踢”或“带球 / 带回”，例如：去别的房间找蓝色的球，踢一脚，然后回起点")
    _close_previous_console()
    for name in KICKS.values():
        fetch_policy(name)
    print(f"交给 Qwen 的部分：{head}\n程序按规则执行的部分：{'踢一脚，然后返回' if verb == KICK else '把球带回起点'}", flush=True)
    events, log = Queue(), []
    labels = {"explore": "探索找球", "approach": "走近球", "done": "已到球旁", "stop": "已到球旁", "search": "原地找球"}

    class ReportingPlanner:
        def predict(self, text, image, memory):
            prediction = planner.predict(text, image, memory)
            name = (prediction.get("action") or {}).get("action", "无法解析动作")
            events.put(f"Qwen：{labels.get(name, name)}")
            return prediction

    def episode(info):
        report = {"find": run_instruction(ReportingPlanner(), head, info)}
        if report["find"]["status"] not in ("model_done", "model_stop"):
            events.put(f"没有走到球旁：{report['find']['status']}")
            return report
        report["play"] = play(info, Path(report["find"]["directory"]) / "ball", verb, color, events.put,
                              cancel_file=ROOT / "runs/cancel_navigation")
        return report

    board = Board("MicroDuck × Qwen-VL · 踢球与带球实验", instruction, "进度")
    worker, frames, stamps = ThreadPoolExecutor(max_workers=1), [], []
    report = None
    try:
        with simulation(seed, module="microduck_qwen.sim_ball", extra=("--movable", color)) as info:
            scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
            home = info["home_pose"]
            observer, feed = Observer(info["port"]), AudienceFeed(info["port"])
            trail = [(home["x"], home["y"])]
            started = next_map = next_capture = time.monotonic()
            picture = None
            try:
                task = worker.submit(episode, info)
                while report is None:
                    tick = time.monotonic()
                    state = observer.read()
                    observer.connection.send({"op": "balls"})
                    for name, ball in observer.connection.receive()["balls"].items():
                        scene["objects"][name].update(x=ball[0], y=ball[1])
                    pose = tuple(state["trunk"][:2])
                    if math.dist(trail[-1], pose) >= 0.005:
                        trail.append(pose)
                    while not events.empty():
                        log.append(events.get_nowait())
                    if task.done():
                        report = task.result()
                        while not events.empty():
                            log.append(events.get_nowait())
                    overview = None
                    if report is not None or tick >= next_map:
                        picture = draw_overview(scene, home, trail, state["yaw"])
                        overview, next_map = jpeg(picture), tick + 0.3
                    spectator, camera = feed.frames()
                    board.update(log[-6:], tick - started, summary(report), spectator=spectator, camera=camera, overview=overview)
                    if spectator and picture is not None and (report is not None or tick >= next_capture):
                        frames.append(clip_frame(spectator, picture, instruction, summary(report) or (log[-1] if log else "等待第一步…"), tick - started))
                        stamps.append(tick - started)
                        next_capture = tick + CAPTURE_S
                    time.sleep(max(0.0, REFRESH_S - (time.monotonic() - tick)))
            finally:
                feed.close()
                observer.close()
    finally:
        worker.shutdown(wait=False, cancel_futures=True)
    print(summary(report), flush=True)
    if frames:
        clip = save_demo_webp(frames, stamps, ROOT / "runs/ball-play.webp", speed=SPEED)
        data = base64.b64encode(Path(clip["path"]).read_bytes()).decode()
        display(HTML(f'<img alt="{html.escape(instruction)}" width="960" style="max-width:100%" src="data:image/webp;base64,{data}">'))
    return report


def summary(report):
    if report is None:
        return ""
    if "play" not in report:
        return f"没有走到球旁（{report['find']['status']}），没有执行踢球或带球"
    ball, back = report["play"]["ball"], report["play"]["return"]
    if report["play"]["verb"] == KICK:
        text = f"踢出 {ball['ball_moved_m']} m（{ball['ball_speed_m_s']} m/s）" if ball["success"] else f"没踢成：{ball['error']}"
    else:
        text = (f"球带到距起点 {ball['ball_from_home_m']} m，用时 {ball['seconds']:.0f} s" if ball["success"]
                else f"没带成：{ball.get('error', '球滚离了起点')}，球距起点 {ball['ball_from_home_m']} m")
    return text + (f" · 机器人已回起点（{report['play']['distance_home_m']} m）" if back["success"] else f" · 返回失败：{back.get('error')}")
