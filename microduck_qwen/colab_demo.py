"""Run a live simulation episode in a notebook using the existing skill agent."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
import base64
import html
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time

from . import ROOT

DEMO_INSTRUCTION = "去别的房间找蓝色的球，走到它旁边，然后回起点"
DEMO_SEED = 98107


NVIDIA_EGL = Path("/usr/lib64-nvidia/libEGL_nvidia.so.0")  # Colab GPU runtimes ship it here
EGL_VENDORS = Path("/usr/share/glvnd/egl_vendor.d")


def prepare_rendering(backend):
    """Make MuJoCo render on the GPU when one exists, and report the OpenGL renderer.

    Colab GPU runtimes register only Mesa's EGL vendor, so EGL silently falls back to
    llvmpipe: about 0.6-1.0 s per camera frame on a T4 instead of about 0.04 s.
    """
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = backend
    if backend == "egl" and NVIDIA_EGL.is_file() and not list(EGL_VENDORS.glob("*nvidia*.json")):
        EGL_VENDORS.mkdir(parents=True, exist_ok=True)
        (EGL_VENDORS / "10_nvidia.json").write_text(json.dumps(
            {"file_format_version": "1.0.0", "ICD": {"library_path": str(NVIDIA_EGL)}}))
    probe = ("import mujoco\n"
             "from OpenGL import GL\n"
             "context = mujoco.GLContext(64, 64)\n"
             "context.make_current()\n"
             "print(GL.glGetString(GL.GL_RENDERER).decode())\n"
             "context.free()\n")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"MuJoCo {backend} 渲染初始化失败：\n{result.stderr[-2000:]}")
    renderer = result.stdout.strip().splitlines()[-1]
    if "llvmpipe" in renderer or backend == "osmesa":
        # Software GL needs 0.4-1 s per frame; let the agent accept frames that old.
        os.environ.setdefault("MICRODUCK_CAMERA_MAX_AGE", "2.0")
    return renderer


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def display_frame(port):
    """Latest camera frame for the audience view; never fails on render latency."""
    from .vision import capture

    return capture(port, max_age=math.inf)


@contextmanager
def simulation(seed):
    """Start the headless simulator, then stop it on success, failure or interruption."""
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colab-", dir=runs) as directory:
        directory = Path(directory)
        manifest = directory / "simulator.json"
        log_path = directory / "simulator.log"
        # Fresh ports: a simulator orphaned by a crashed kernel must not block the next run.
        # Physics runs on its own thread so software rendering cannot freeze simulated time.
        command = [sys.executable, "-u", "-m", "microduck_qwen.sim_native", "--vision",
                   "--world", "2", "2", "--seed", str(seed), "--manifest", str(manifest),
                   "--port", str(free_port()), "--control-port", str(free_port()), "--async-physics"]
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
        try:
            deadline = time.monotonic() + 45
            while process.poll() is None and time.monotonic() < deadline:
                if manifest.exists():
                    try:
                        info = json.loads(manifest.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass  # The simulator may still be writing the manifest.
                    else:
                        break
                time.sleep(0.1)
            else:
                log = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"仿真未能启动。\n{log}")
            info["log_file"] = str(log_path)
            yield info
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def draw_overview(scene, home, trail, yaw):
    """Draw an audience-only map from the scene and measured simulator poses."""
    from PIL import Image, ImageDraw

    width, margin = 420, 24
    image = Image.new("RGB", (width, width), "#0f172a")
    draw = ImageDraw.Draw(image)
    xmin, xmax, ymin, ymax = scene["bounds"]
    scale = (width - 2 * margin) / max(xmax - xmin, ymax - ymin)

    def point(x, y):
        return margin + scale * (x - xmin), width - margin - scale * (y - ymin)

    def circle(x, y, radius, color):
        px, py = point(x, y)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    for wall in scene["walls"]:
        draw.line((point(wall["x1"], wall["y1"]), point(wall["x2"], wall["y2"])),
                  fill="#94a3b8", width=5)
    for obj in [*scene["obstacles"], *scene["objects"].values()]:
        color = tuple(round(channel * 255) for channel in obj["rgba"][:3])
        circle(obj["x"], obj["y"], max(4, obj["radius"] * scale), color)
    if len(trail) > 1:
        draw.line([point(*pose) for pose in trail], fill="#38bdf8", width=3)
    circle(home["x"], home["y"], 5, "#f8fafc")
    x, y = trail[-1]
    circle(x, y, 6, "#fb923c")
    draw.line((point(x, y), point(x + 0.22 * math.cos(yaw), y + 0.22 * math.sin(yaw))),
              fill="#fb923c", width=3)
    return image


def dashboard(frame, overview, instruction, decisions, elapsed, summary="", *,
              title="MicroDuck × Qwen-VL", decision_label="模型决策"):
    from IPython.display import HTML
    from PIL import Image

    def img(image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78)
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()

    actions = " → ".join(decisions) or "等待第一步…"
    return HTML(f"""<div style="font:16px system-ui;max-width:1000px">
      <p><b>{html.escape(title)}</b>　{html.escape(instruction)}</p>
      <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
        <div>仿真俯视图 · 白点为起点<br><img width="380" src="{img(overview)}"></div>
        <div>机器人实时第一视角<br><img width="480" src="{img(Image.fromarray(frame.rgb))}"></div>
      </div>
      <p><b>{html.escape(decision_label)}</b>　{html.escape(actions)}　·　{elapsed:.0f} 秒</p>
      <p>{html.escape(summary)}</p>
    </div>""")


def live_demo(planner, instruction=DEMO_INSTRUCTION, seed=DEMO_SEED):
    """Execute fresh model decisions while streaming simulated RGB and odometry."""
    from IPython.display import display

    from .agent import run_instruction
    from .navigation import Observer

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("请输入一条中文指令。")
    decisions = Queue()

    class ReportingPlanner:
        def predict(self, text, image, memory):
            prediction = planner.predict(text, image, memory)
            action = prediction.get("action") or {}
            decisions.put(action.get("action", "无法解析动作"))
            return prediction

    output = display("正在启动 MuJoCo…", display_id=True)
    # On interrupt, close the simulation before waiting for the in-flight inference.
    with ThreadPoolExecutor(max_workers=1) as worker, simulation(seed) as info:
        scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
        home = info["home_pose"]
        observer = Observer(info["port"])
        trail, actions = [], []
        started = time.monotonic()
        try:
            frame, state = display_frame(info["port"]), observer.read()
            task = worker.submit(run_instruction, ReportingPlanner(), instruction, info)
            while True:
                pose = tuple(state["trunk"][:2])
                trail.append(pose)
                while not decisions.empty():
                    actions.append(decisions.get_nowait())
                overview = draw_overview(scene, home, trail, state["yaw"])
                elapsed = time.monotonic() - started
                report = task.result() if task.done() else None
                summary = ""
                if report is not None:
                    distance = math.dist(pose, (home["x"], home["y"]))
                    summary = f"结束状态 {report['status']} · 距起点 {distance:.2f} m · done 是模型声明"
                    error = next((step["execution"]["error"] for step in reversed(report["steps"])
                                  if step.get("execution", {}).get("error")), None)
                    if error:
                        summary += f" · {error}"
                output.update(dashboard(frame, overview, instruction, actions, elapsed, summary))
                if report is not None:
                    return report
                time.sleep(0.5)
                frame, state = display_frame(info["port"]), observer.read()
        finally:
            observer.close()


def live_baseline(motion="forward", seed=DEMO_SEED):
    """Run an explicit manual motion with the official gait, with no model inference."""
    from IPython.display import display

    from .navigation import Observer
    from .skills import SkillExecutor

    motions = {
        "forward": ("手动前进 0.20 m", {"action": "move", "direction": "forward", "distance_m": 0.2}),
        "left": ("手动左转 45°", {"action": "turn", "direction": "left", "degrees": 45}),
        "right": ("手动右转 45°", {"action": "turn", "direction": "right", "degrees": 45}),
        "stop": ("停止", {"action": "stop"}),
    }
    label, action = motions[motion]

    def execute(info, directory):
        executor = SkillExecutor(info, directory)
        try:
            return executor.run(action, [])
        finally:
            executor.close()

    output = display("正在启动 MicroDuck…", display_id=True)
    with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=1) as worker, simulation(seed) as info:
        observer = Observer(info["port"])
        try:
            scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
            trail, started = [], time.monotonic()
            task = worker.submit(execute, info, Path(directory))
            while True:
                state = observer.read()
                frame = display_frame(info["port"])
                trail.append(tuple(state["trunk"][:2]))
                overview = draw_overview(scene, info["home_pose"], trail, state["yaw"])
                result = task.result() if task.done() else None
                summary = ""
                if result is not None:
                    distance = math.dist((info["home_pose"]["x"], info["home_pose"]["y"]), trail[-1])
                    summary = f"执行结束 · 位移 {distance:.2f} m" if result["success"] else result.get("error", "已停车")
                output.update(dashboard(frame, overview, label, [action["action"]],
                                        time.monotonic() - started, summary,
                                        title="MicroDuck · 官方步态", decision_label="手动命令"))
                if result is not None:
                    return result
                time.sleep(0.3)
        finally:
            observer.close()
