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


class AudienceFeed:
    """(chase view, robot camera) as JPEG bytes for the audience; (None, None) until rendered.

    One small request per refresh over one kept-open connection: the simulator pre-encodes
    both pictures, so the dashboard never pulls the agent's full RGB-D packet (about 1 MB
    with depth), and the simulator log is not flooded with connect/disconnect lines.
    """

    def __init__(self, port):
        self.port, self.connection = port, None

    def frames(self):
        from .navigation import JsonConnection

        try:
            if self.connection is None:
                self.connection = JsonConnection(("127.0.0.1", self.port))
            self.connection.send({"op": "spectator"})
            packet = self.connection.receive()
        except RuntimeError:
            return None, None  # not rendered yet
        except OSError:
            self.close()  # cosmetic view: reconnect next time, never stop the demo over it
            return None, None
        camera = packet.get("camera_jpeg_b64")
        return base64.b64decode(packet["jpeg_b64"]), base64.b64decode(camera) if camera else None

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except OSError:
                pass
            self.connection = None


@contextmanager
def simulation(seed):
    """Start the headless simulator, then stop it on success, failure or interruption."""
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colab-", dir=runs, ignore_cleanup_errors=True) as directory:
        directory = Path(directory)
        manifest = directory / "simulator.json"
        log_path = directory / "simulator.log"
        # Fresh ports: a simulator orphaned by a crashed kernel must not block the next run.
        # Physics runs on its own thread so software rendering cannot freeze simulated time.
        command = [sys.executable, "-u", "-m", "microduck_qwen.sim_native", "--vision",
                   "--world", "2", "2", "--seed", str(seed), "--manifest", str(manifest),
                   "--port", str(free_port()), "--control-port", str(free_port()), "--async-physics", "--spectator"]
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


def jpeg(image, quality=78):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class Board:
    """Dashboard whose pictures update in place, so motion reads as video, not page reloads.

    Uses ipywidgets (preinstalled on Colab); without it, falls back to re-rendering HTML.
    Only changed pictures are sent, which keeps a 10 Hz refresh light.
    """

    PANELS = (("spectator", "第三视角 · 跟随小鸭子（仅供观众）", 480),
              ("camera", "机器人实时第一视角 · 模型看到的画面", 480),
              ("overview", "仿真俯视图 · 白点为起点", 300))

    def __init__(self, title, subtitle, decision_label):
        from IPython.display import HTML, display

        self.header = f"<b>{html.escape(title)}</b>　{html.escape(subtitle)}"
        self.decision_label, self.latest, self.status_text = decision_label, {}, None
        try:
            import ipywidgets as widgets
        except ImportError:
            widgets = None
        self.widgets = widgets
        if widgets is None:
            self.handle = display(HTML(self.header), display_id=True)
            return
        self.images, boxes = {}, []
        for name, caption, width in self.PANELS:
            self.images[name] = widgets.Image(format="jpeg", layout=widgets.Layout(width=f"{width}px"))
            boxes.append(widgets.VBox([widgets.HTML(html.escape(caption)), self.images[name]]))
        self.status = widgets.HTML()
        display(widgets.VBox([widgets.HTML(f'<div style="font-size:16px">{self.header}</div>'),
                              widgets.HBox(boxes, layout=widgets.Layout(flex_flow="row wrap", grid_gap="16px")),
                              self.status]))

    def update(self, decisions, elapsed, summary="", **pictures):
        """pictures: spectator / camera / overview as JPEG bytes; None keeps the previous one."""
        from IPython.display import HTML

        changed = {name: data for name, data in pictures.items()
                   if data is not None and data != self.latest.get(name)}
        self.latest.update(changed)
        actions = " → ".join(decisions) or "等待第一步…"
        status = (f"<p><b>{html.escape(self.decision_label)}</b>　{html.escape(actions)}　·　{elapsed:.0f} 秒</p>"
                  f"<p>{html.escape(summary)}</p>")
        if self.widgets is not None:
            for name, data in changed.items():
                self.images[name].value = data
            if status != self.status_text:
                self.status.value = status
        elif changed or status != self.status_text:
            pictures_html = "".join(
                f'<div>{html.escape(caption)}<br><img width="{width}" '
                f'src="data:image/jpeg;base64,{base64.b64encode(self.latest[name]).decode()}"></div>'
                for name, caption, width in self.PANELS if name in self.latest)
            self.handle.update(HTML(f"""<div style="font:16px system-ui;max-width:1000px"><p>{self.header}</p>
              <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">{pictures_html}</div>
              {status}</div>"""))
        self.status_text = status


REFRESH_S = 1 / 15  # dashboard cadence; on a GPU the simulator renders the chase view at 15 Hz, the robot camera at 10 Hz


def live_demo(planner, instruction=DEMO_INSTRUCTION, seed=DEMO_SEED):
    """Execute fresh model decisions while streaming simulated RGB and odometry."""
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

    board = Board("MicroDuck × Qwen-VL", instruction, "模型决策")
    # On interrupt, close the simulation before waiting for the in-flight inference.
    with ThreadPoolExecutor(max_workers=1) as worker, simulation(seed) as info:
        scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
        home = info["home_pose"]
        observer, feed = Observer(info["port"]), AudienceFeed(info["port"])
        trail, actions = [], []
        started = next_map = time.monotonic()
        try:
            task = worker.submit(run_instruction, ReportingPlanner(), instruction, info)
            while True:
                tick = time.monotonic()
                state = observer.read()
                pose = tuple(state["trunk"][:2])
                trail.append(pose)
                while not decisions.empty():
                    actions.append(decisions.get_nowait())
                report = task.result() if task.done() else None
                summary, overview = "", None
                if report is not None or tick >= next_map:
                    overview, next_map = jpeg(draw_overview(scene, home, trail, state["yaw"])), tick + 0.3
                if report is not None:
                    distance = math.dist(pose, (home["x"], home["y"]))
                    summary = f"结束状态 {report['status']} · 距起点 {distance:.2f} m · done 是模型声明"
                    error = next((step["execution"]["error"] for step in reversed(report["steps"])
                                  if step.get("execution", {}).get("error")), None)
                    if error:
                        summary += f" · {error}"
                spectator, camera = feed.frames()
                board.update(actions, tick - started, summary, spectator=spectator, camera=camera, overview=overview)
                if report is not None:
                    return report
                time.sleep(max(0.0, REFRESH_S - (time.monotonic() - tick)))
        finally:
            feed.close()
            observer.close()


def live_baseline(motion="forward", seed=DEMO_SEED):
    """Run an explicit manual motion with the official gait, with no model inference."""
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

    board = Board("MicroDuck · 官方步态", label, "手动命令")
    with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=1) as worker, simulation(seed) as info:
        observer, feed = Observer(info["port"]), AudienceFeed(info["port"])
        try:
            scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
            trail, started = [], time.monotonic()
            next_map = started
            task = worker.submit(execute, info, Path(directory))
            while True:
                tick = time.monotonic()
                state = observer.read()
                trail.append(tuple(state["trunk"][:2]))
                result = task.result() if task.done() else None
                overview = None
                if result is not None or tick >= next_map:
                    overview = jpeg(draw_overview(scene, info["home_pose"], trail, state["yaw"]))
                    next_map = tick + 0.3
                summary = ""
                if result is not None:
                    distance = math.dist((info["home_pose"]["x"], info["home_pose"]["y"]), trail[-1])
                    summary = f"执行结束 · 位移 {distance:.2f} m" if result["success"] else result.get("error", "已停车")
                spectator, camera = feed.frames()
                board.update([action["action"]], tick - started, summary,
                             spectator=spectator, camera=camera, overview=overview)
                if result is not None:
                    return result
                time.sleep(max(0.0, REFRESH_S - (time.monotonic() - tick)))
        finally:
            feed.close()
            observer.close()
