"""Compose the third-person + PiP trajectory README demo GIF."""

from pathlib import Path
import math

ACTION_LABELS = {
    "explore": "搜索目标",
    "search": "原地搜寻",
    "approach": "走到旁边",
    "return_home": "返回起点",
    "bypass_return": "绕行返回",
    "move": "移动",
    "turn": "转向",
    "stop": "停车",
    "done": "模型结束",
    "clarify": "请求澄清",
    "unsupported": "暂不支持",
}

FONT_CANDIDATES_REGULAR = (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

FONT_CANDIDATES_BOLD = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "msyhbd.ttc",
    "msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def load_font(size, *, bold=False):
    from PIL import ImageFont

    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rounded_image(image, size, radius=12):
    from PIL import Image, ImageDraw

    resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return resized, mask


def _color_label(color):
    return {"blue": "蓝球", "red": "红球", "green": "绿球", "yellow": "黄标"}.get(color, "目标")


def _decision_text(action):
    if not isinstance(action, dict):
        return "thinking", "看图推理中…"
    name = action.get("action", "unknown")
    if name in ("explore", "search") and action.get("color"):
        return name, f"搜索{_color_label(action['color'])}"
    if name == "approach" and action.get("color"):
        return name, f"走向{_color_label(action['color'])}"
    return name, ACTION_LABELS.get(name, name)


def _step_stages(instruction):
    target = "蓝球" if "蓝" in instruction else "红球" if "红" in instruction else "绿球" if "绿" in instruction else "目标"
    return [
        ("explore", f"搜索{target}"),
        ("approach", "走到旁边"),
        ("return_home", "返回起点"),
        ("done", "模型结束"),
    ]


def compose_demo_frame(
    spectator_img,
    camera_img,
    minimap_img,
    instruction,
    decisions,
    elapsed,
    distance,
    *,
    speed=5,
    done=False,
):
    """Compose one 1220x650 frame: 3rd-person duck hero + PiP trail map + 1st-person camera + step bar."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1220, 650), "#0b1220")
    draw = ImageDraw.Draw(canvas)

    f_eyebrow = load_font(15, bold=True)
    f_title = load_font(30, bold=True)
    f_label = load_font(16, bold=False)
    f_pill = load_font(15, bold=False)
    f_action = load_font(30, bold=True)
    f_sub = load_font(19, bold=False)
    f_step_code = load_font(17, bold=True)
    f_step_zh = load_font(16, bold=False)
    f_mini = load_font(12, bold=True)

    # Header
    draw.text((30, 18), "MICRODUCK × QWEN-VL", font=f_eyebrow, fill="#38bdf8")
    short_title = instruction if len(instruction) <= 28 else instruction[:27] + "…"
    draw.text((30, 44), short_title, font=f_title, fill="#f8fafc")

    badge = f"MuJoCo 实景 · {speed}× 播放 · {elapsed:.0f}s"
    badge_w = int(draw.textlength(badge, font=f_pill)) + 32
    draw.rounded_rectangle((1190 - badge_w, 34, 1190, 72), radius=19, fill="#111b2e", outline="#334155", width=1)
    draw.text((1190 - badge_w + 16, 43), badge, font=f_pill, fill="#cbd5e1")
    draw.line((30, 102, 1190, 102), fill="#1e293b", width=1)

    # Left Hero: 3rd-person chase view of the real 3D MicroDuck (700 x 394, 16:9)
    draw.text((30, 116), "第三视角 · 跟随 MicroDuck（左下为 2×2 房间轨迹）", font=f_label, fill="#94a3b8")
    hero_box = (30, 146, 730, 540)
    hero_img, hero_mask = _rounded_image(spectator_img, (700, 394), radius=12)
    canvas.paste(hero_img, (30, 146), hero_mask)
    draw.rounded_rectangle(hero_box, radius=12, outline="#263752", width=2)

    # Bottom-left PiP mini-map inside the hero view
    if minimap_img is not None:
        pip_x, pip_y, pip_s = 44, 360, 166
        draw.rounded_rectangle((pip_x - 4, pip_y - 22, pip_x + pip_s + 4, pip_y + pip_s + 4),
                               radius=10, fill="#0b1220", outline="#38bdf8", width=2)
        draw.text((pip_x + 4, pip_y - 19), "2×2 轨迹 · 白点起点", font=f_mini, fill="#cbd5e1")
        mini_img, mini_mask = _rounded_image(minimap_img, (pip_s, pip_s), radius=7)
        canvas.paste(mini_img, (pip_x, pip_y), mini_mask)

    # Right Top: Model first-person camera (436 x 245, 16:9)
    draw.text((754, 116), "模型看到的第一视角", font=f_label, fill="#94a3b8")
    cam_box = (754, 146, 1190, 391)
    cam_img, cam_mask = _rounded_image(camera_img, (436, 245), radius=12)
    canvas.paste(cam_img, (754, 146), cam_mask)
    draw.rounded_rectangle(cam_box, radius=12, outline="#263752", width=2)

    # Right Bottom: Current decision + distance to home
    card_box = (754, 410, 1190, 540)
    draw.rounded_rectangle(card_box, radius=12, fill="#111b2e", outline="#263752", width=2)
    draw.text((776, 426), "当前决策", font=f_label, fill="#94a3b8")

    dist_text = f"距起点 {distance:.2f} m"
    dist_color = "#34d399" if (done and distance < 0.3) else "#38bdf8"
    dist_w = int(draw.textlength(dist_text, font=f_pill)) + 24
    draw.rounded_rectangle((1170 - dist_w, 422, 1170, 452), radius=14, fill="#0b172a", outline=dist_color, width=1)
    draw.text((1170 - dist_w + 12, 428), dist_text, font=f_pill, fill=dist_color)

    last_action = decisions[-1] if decisions else None
    code, subtitle = _decision_text(last_action)
    draw.text((776, 456), code, font=f_action, fill="#38bdf8" if not done else "#34d399")
    draw.text((776, 498), subtitle, font=f_sub, fill="#e2e8f0")

    # Bottom 4-step pipeline bar
    stages = _step_stages(instruction)
    seen_names = [a.get("action") if isinstance(a, dict) else str(a) for a in decisions]
    active_name = seen_names[-1] if seen_names else None
    bar_w = (1160 - 3 * 16) // 4
    for idx, (stage_code, stage_zh) in enumerate(stages):
        x0 = 30 + idx * (bar_w + 16)
        x1 = x0 + bar_w
        y0, y1 = 568, 622
        is_active = active_name == stage_code
        is_completed = stage_code in seen_names and not is_active
        fill = "#0f2942" if is_active else "#111b2e"
        outline = "#38bdf8" if is_active else ("#22c55e" if is_completed else "#1e293b")
        width = 2 if (is_active or is_completed) else 1
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=fill, outline=outline, width=width)
        code_color = "#38bdf8" if is_active else ("#4ade80" if is_completed else "#94a3b8")
        zh_color = "#f8fafc" if is_active else ("#cbd5e1" if is_completed else "#64748b")
        draw.text((x0 + 16, y0 + 16), stage_code, font=f_step_code, fill=code_color)
        zh_w = int(draw.textlength(stage_zh, font=f_step_zh))
        draw.text((x1 - 16 - zh_w, y0 + 17), stage_zh, font=f_step_zh, fill=zh_color)

    return canvas


def quantize_demo_frame(canvas, colors=128):
    """Quantize a composed frame while preserving bright UI/ball/robot accent colors."""
    from PIL import Image

    w, h = canvas.size
    swatches = Image.new("RGB", (w, 24), "#0b1220")
    keep = ("#38bdf8", "#fb923c", "#34d399", "#e5322d", "#2a5cff", "#2fd14f", "#facc15", "#f8fafc")
    step = w // len(keep)
    for i, color in enumerate(keep):
        swatches.paste(Image.new("RGB", (step, 24), color), (i * step, 0))
    extended = Image.new("RGB", (w, h + 24))
    extended.paste(canvas, (0, 0))
    extended.paste(swatches, (0, h))
    return extended.quantize(colors=colors, dither=Image.Dither.NONE).crop((0, 0, w, h))


def save_demo_gif(frames, timestamps, output_path, *, speed=5, final_hold_ms=1500):
    """Encode quantized PIL frames into an animated GIF with accurate speed-scaled timing."""
    if not frames:
        raise ValueError("No frames to save")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [
        max(40, round((b - a) * 1000 / speed / 10) * 10)
        for a, b in zip(timestamps, timestamps[1:])
    ] + [final_hold_ms]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return {
        "path": str(output),
        "frames": len(frames),
        "size_bytes": output.stat().st_size,
        "duration_s": round(sum(durations) / 1000.0, 2),
    }
