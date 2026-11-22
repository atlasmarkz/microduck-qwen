"""Record the README demo GIF: third-person chase view of the duck, the model's camera, map and decisions.

Two steps so the layout can be tuned without re-running the model:

    python -m microduck_qwen.record_readme_demo record --adapter adapters/lang-v2
    python -m microduck_qwen.record_readme_demo compose

`record` runs one real episode (same simulator, chase camera and skill agent as the Colab demo)
and stores sampled frames under runs/readme-demo/. `compose` turns them into
assets/microduck-navigation-demo.gif at 5x speed.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

from . import ROOT
from .colab_demo import DEMO_INSTRUCTION, DEMO_SEED, AudienceFeed, draw_overview, simulation

CAPTURE = ROOT / "runs/readme-demo"
OUTPUT = ROOT / "assets/microduck-navigation-demo.gif"
SAMPLE_S = 0.2
GLOSS = {"explore": "去别的房间搜索", "approach": "走到目标旁边", "return_home": "返回起点",
         "done": "模型声明完成", "search": "原地搜索", "move": "直行", "turn": "转向", "stop": "停车",
         "clarify": "请求澄清", "unsupported": "无法执行", "bypass_return": "绕行返回"}


def record(adapter, instruction, seed, out):
    """Run one episode and sample chase view, robot camera and pose every SAMPLE_S seconds."""
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    from .agent import run_instruction
    from .navigation import Observer
    from .vision_torch import TorchAgentPlanner

    planner = TorchAgentPlanner(adapter)
    shutil.rmtree(out, ignore_errors=True)
    (out / "frames").mkdir(parents=True)
    events, started = [], [None]

    class RecordingPlanner:
        def predict(self, text, image, memory):
            events.append({"t": time.monotonic() - started[0], "kind": "thinking"})
            prediction = planner.predict(text, image, memory)
            action = (prediction.get("action") or {}).get("action", "invalid")
            events.append({"t": time.monotonic() - started[0], "kind": "decided", "action": action})
            return prediction

    with simulation(seed) as info:
        shutil.copyfile(info["scene_file"], out / "scene.json")
        observer, feed = Observer(info["port"]), AudienceFeed(info["port"])
        result, samples, last = {}, [], {}
        started[0] = time.monotonic()

        def episode():
            try:
                result["report"] = run_instruction(RecordingPlanner(), instruction, info)
            except BaseException as error:  # keep the frames for diagnosis
                result["error"] = f"{type(error).__name__}: {error}"

        worker = threading.Thread(target=episode)
        worker.start()
        try:
            while True:
                tick = time.monotonic()
                state = observer.read()
                sample = {"t": tick - started[0], "x": state["trunk"][0], "y": state["trunk"][1], "yaw": state["yaw"]}
                for name, data in zip(("spectator", "camera"), feed.frames()):
                    if data is not None and data != last.get(name):
                        last[name] = data
                        path = out / "frames" / f"{len(samples):05d}-{name}.jpg"
                        path.write_bytes(data)
                        last[name + "_file"] = path.name
                    sample[name] = last.get(name + "_file")
                samples.append(sample)
                if not worker.is_alive():
                    break
                time.sleep(max(0.0, SAMPLE_S - (time.monotonic() - tick)))
        finally:
            worker.join()
            feed.close()
            observer.close()
    report = result.get("report") or {}
    home = info["home_pose"]
    capture = {"instruction": instruction, "seed": seed, "adapter": Path(adapter).as_posix(), "home": home,
               "status": report.get("status"), "error": result.get("error") or report.get("error"),
               "final_home_distance_m": math.dist((samples[-1]["x"], samples[-1]["y"]), (home["x"], home["y"])),
               "events": events, "samples": samples}
    (out / "capture.json").write_text(json.dumps(capture, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: capture[key] for key in ("status", "error", "final_home_distance_m")}, ensure_ascii=False))
    return capture


class Painter:
    """Fixed 1220x650 dark layout matching the earlier README GIF."""

    W, H = 1220, 650
    BG, PANEL, LINE = "#0d1422", "#172033", "#2a3650"
    TEXT, MUTED, ACCENT = "#e6edf7", "#8a9ab5", "#46c7d6"

    def __init__(self):
        from PIL import ImageFont

        def font(names, size):
            for name in names:
                for folder in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/noto", "/usr/share/fonts/opentype/noto"):
                    try:
                        return ImageFont.truetype(str(Path(folder) / name), size)
                    except OSError:
                        continue
            return ImageFont.load_default(size)

        cjk = ("msyh.ttc", "NotoSansCJK-Regular.ttc")
        bold = ("msyhbd.ttc", "NotoSansCJK-Bold.ttc")
        mono = ("consola.ttf", "DejaVuSansMono.ttf")
        self.kicker, self.title = font(bold, 15), font(cjk, 30)
        self.caption, self.small = font(cjk, 17), font(cjk, 15)
        self.action, self.chip = font(mono, 30), font(mono, 15)

    def rounded(self, canvas, image, box, radius=12):
        from PIL import Image, ImageDraw

        image = image.resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS)
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), radius, fill=255)
        canvas.paste(image, box[:2], mask)

    def frame(self, capture, spectator, camera, overview, card, chips, active, elapsed):
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (self.W, self.H), self.BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((34, 30), "MICRODUCK × QWEN-VL", font=self.kicker, fill=self.ACCENT)
        draw.text((34, 54), capture["instruction"], font=self.title, fill=self.TEXT)
        badge = "MuJoCo 录制  /  5× 播放"
        width = draw.textlength(badge, font=self.small)
        draw.rounded_rectangle((1186 - width - 32, 42, 1186, 80), 19, outline=self.LINE, fill=self.PANEL)
        draw.text((1186 - width - 16, 51), badge, font=self.small, fill=self.TEXT)
        draw.line((34, 108, 1186, 108), fill=self.LINE, width=1)

        draw.text((34, 122), "第三视角 · 跟随小鸭子", font=self.caption, fill=self.TEXT)
        self.rounded(canvas, spectator, (34, 150, 734, 544))
        draw.text((760, 122), "模型看到的第一视角", font=self.caption, fill=self.TEXT)
        self.rounded(canvas, camera, (760, 150, 1186, 390))
        self.rounded(canvas, overview, (760, 404, 900, 544), radius=10)

        draw.rounded_rectangle((914, 404, 1186, 544), 12, fill=self.PANEL, outline=self.LINE)
        draw.text((932, 416), "当前决策", font=self.small, fill=self.MUTED)
        clock = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        draw.text((1168 - draw.textlength(clock, font=self.small), 416), clock, font=self.small, fill=self.MUTED)
        draw.text((932, 446), card[0], font=self.action, fill=self.ACCENT if card[0] != "…" else self.MUTED)
        draw.text((932, 494), card[1], font=self.caption, fill=self.TEXT)

        gap = 12
        chip_w = (1186 - 34 - gap * (len(chips) - 1)) / max(1, len(chips))
        for index, name in enumerate(chips):
            left = 34 + index * (chip_w + gap)
            box = (left, 572, left + chip_w, 618)
            current, past = index == active, active is not None and index < active
            draw.rounded_rectangle(box, 10, fill="#123a44" if current else self.PANEL,
                                   outline=self.ACCENT if current else self.LINE, width=2 if current else 1)
            color = self.ACCENT if current else (self.TEXT if past else self.MUTED)
            draw.text((left + 14, 587), name, font=self.chip, fill=color)
            gloss = GLOSS.get(name, "")
            draw.text((box[2] - 14 - draw.textlength(gloss, font=self.small), 583), gloss, font=self.small, fill=color)
        return canvas


def compose(out, gif, speed=5.0, fps=4.0, colors=96, scale=0.8):
    """Render the captured episode at `speed`x as a GIF of `fps` frames per second."""
    from PIL import Image

    capture = json.loads((out / "capture.json").read_text(encoding="utf-8"))
    scene = json.loads((out / "scene.json").read_text(encoding="utf-8"))
    samples, events, home = capture["samples"], capture["events"], capture["home"]
    decided = [event for event in events if event["kind"] == "decided"]
    chips = [event["action"] for event in decided]
    painter = Painter()
    cache = {}

    def picture(name):
        if name not in cache:
            cache[name] = Image.open(out / "frames" / name).convert("RGB")
        return cache[name]

    step, frames, index = speed / fps, [], 0
    end = samples[-1]["t"]
    times = [i * step for i in range(int(end / step) + 1)] + [end]
    for t in times:
        while index + 1 < len(samples) and samples[index + 1]["t"] <= t:
            index += 1
        sample = samples[index]
        if not sample.get("spectator") or not sample.get("camera"):
            continue
        trail = [(s["x"], s["y"]) for s in samples[:index + 1]]
        done = [event for event in decided if event["t"] <= t]
        thinking = any(event["kind"] == "thinking" and event["t"] <= t for event in events[len(done) * 2:])
        if t >= end:
            card = (done[-1]["action"] if done else "…", f"距起点 {capture['final_home_distance_m']:.2f} m")
        elif thinking:
            card = ("…", "模型推理中")
        else:
            card = (done[-1]["action"], GLOSS.get(done[-1]["action"], "")) if done else ("…", "启动中")
        active = len(done) - 1 if done else None
        frames.append(painter.frame(capture, picture(sample["spectator"]), picture(sample["camera"]),
                                    draw_overview(scene, home, trail, sample["yaw"]), card, chips, active, t))
    durations = [round(1000 / fps)] * len(frames)
    durations[-1] = 3000  # hold the result
    if scale != 1.0:
        size = (round(Painter.W * scale), round(Painter.H * scale))
        frames = [frame.resize(size, Image.LANCZOS) for frame in frames]
    # One palette for the whole clip, built from frames spread over it: small per-frame palettes
    # drop rare colours (the yellow balls turned white at 64 colours), and a shared palette lets
    # unchanged pixels stay identical between frames, which the GIF delta encoding exploits.
    picks = frames[::max(1, len(frames) // 12)]
    # Object colours cover few pixels, so median cut would merge them (yellow came out white,
    # green teal); seed the palette with shaded swatches of every ball and pillar colour.
    tints = sorted({tuple(round(c * 255) for c in obj["rgba"][:3])
                    for obj in [*scene["obstacles"], *scene["objects"].values()]})
    swatch, shades = 64, (0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
    rows = math.ceil(len(tints) * len(shades) * swatch / picks[0].width)
    montage = Image.new("RGB", (picks[0].width, picks[0].height * len(picks) + rows * swatch * 4))
    for row, frame in enumerate(picks):
        montage.paste(frame, (0, row * frame.height))
    cells = [tuple(round(channel * shade) for channel in tint) for tint in tints for shade in shades]
    per_row = picks[0].width // swatch
    for index, color in enumerate(cells):
        x, y = (index % per_row) * swatch, picks[0].height * len(picks) + (index // per_row) * swatch * 4
        montage.paste(color, (x, y, x + swatch, y + swatch * 4))
    shared = montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette = [frame.quantize(palette=shared, dither=Image.Dither.NONE) for frame in frames]
    gif.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(gif, save_all=True, append_images=palette[1:], duration=durations, loop=0,
                    optimize=True, disposal=1)
    size = gif.stat().st_size
    record_file = gif.with_suffix(".json")
    record_file.write_text(json.dumps({key: capture[key] for key in (
        "instruction", "seed", "adapter", "home", "status", "error", "final_home_distance_m", "events")},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"gif": str(gif), "frames": len(frames), "seconds": sum(durations) / 1000,
                      "megabytes": round(size / 1e6, 2), "status": capture["status"],
                      "final_home_distance_m": round(capture["final_home_distance_m"], 3)}, ensure_ascii=False))
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("record", "compose"))
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapters/lang-v2")
    parser.add_argument("--instruction", default=DEMO_INSTRUCTION)
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--colors", type=int, default=96)
    parser.add_argument("--scale", type=float, default=0.8, help="shrink the 1220x650 canvas; defaults keep the GIF under 6 MB")
    args = parser.parse_args()
    if args.command == "record":
        record(args.adapter, args.instruction, args.seed, args.capture)
    else:
        compose(args.capture, args.output, args.speed, args.fps, args.colors, args.scale)


if __name__ == "__main__":
    main()
