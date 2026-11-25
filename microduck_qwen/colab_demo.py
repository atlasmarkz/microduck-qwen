"""Live, interactive, and GIF-recording MuJoCo console for Colab notebooks."""

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
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

from . import ROOT

DEMO_INSTRUCTION = "去别的房间找蓝色的球，走到它旁边，然后回起点"
DEMO_SEED = 98107

PRESET_INSTRUCTIONS = [
    ("找蓝球并回起点", "去别的房间找蓝色的球，走到它旁边，然后回起点"),
    ("找红球", "去别的房间找红色的球，走到它旁边"),
    ("找绿球", "找绿色的球并走到它旁边"),
    ("前进 0.3m", "向前走0.3米"),
    ("左转 90°", "向左转90度"),
    ("回起点", "回到起点"),
]

NVIDIA_EGL = Path("/usr/lib64-nvidia/libEGL_nvidia.so.0")  # Colab GPU runtimes ship it here
EGL_VENDORS = Path("/usr/share/glvnd/egl_vendor.d")

_ACTIVE_CONSOLE = None


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


def _free_port_pair():
    sockets, ports = [], []
    try:
        for _ in range(2):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()
    return ports[0], ports[1]


def _format_action(action):
    if not isinstance(action, dict):
        return "无法解析动作"
    name = action.get("action", "unknown")
    params = [f"{k}={v}" for k, v in action.items() if k != "action"]
    return f"{name}({', '.join(params)})" if params else name


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


class SimSession:
    """Persistent headless MuJoCo simulator with async physics and 15 Hz spectator camera."""

    def __init__(self, seed=DEMO_SEED):
        self.seed = int(seed)
        self._lock = threading.Lock()
        self._tempdir = None
        self._process = None
        self._observer = None
        self._feed = None
        self.info = None
        self.scene = None
        self.home = None
        self.trail = []
        self._last_yaw = 0.0
        self._start(self.seed)

    def _start(self, seed):
        from .navigation import Observer

        runs = ROOT / "runs"
        runs.mkdir(exist_ok=True)
        tempdir = tempfile.TemporaryDirectory(prefix="colab-", dir=runs, ignore_cleanup_errors=True)
        directory = Path(tempdir.name)
        manifest = directory / "simulator.json"
        log_path = directory / "simulator.log"
        port, control_port = _free_port_pair()
        command = [
            sys.executable, "-u", "-m", "microduck_qwen.sim_native", "--vision",
            "--world", "2", "2", "--seed", str(seed), "--manifest", str(manifest),
            "--port", str(port), "--control-port", str(control_port),
            "--async-physics", "--spectator",
        ]
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
            )
        try:
            deadline = time.monotonic() + 45
            info = None
            while process.poll() is None and time.monotonic() < deadline:
                if manifest.exists():
                    try:
                        info = json.loads(manifest.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                    else:
                        break
                time.sleep(0.1)
            if info is None:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"仿真未能启动。\n{log_tail}")
            info["log_file"] = str(log_path)
            observer = Observer(info["port"])
            feed = AudienceFeed(info["port"])
            scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            tempdir.cleanup()
            raise

        self._tempdir = tempdir
        self._process = process
        self._observer = observer
        self._feed = feed
        self.info = info
        self.scene = scene
        self.home = info["home_pose"]
        self.seed = int(seed)
        self.trail = [(self.home["x"], self.home["y"])]
        self._last_yaw = float(self.home.get("yaw", 0.0))

    @property
    def alive(self):
        return self._process is not None and self._process.poll() is None

    def read_log_tail(self, chars=1200):
        if self._tempdir is None:
            return ""
        log_path = Path(self._tempdir.name) / "simulator.log"
        if not log_path.is_file():
            return ""
        return log_path.read_text(encoding="utf-8", errors="replace")[-chars:]

    def sample(self, *, draw_map=False):
        """Read pose + spectator/camera JPEGs over kept-open connections."""
        with self._lock:
            if not self.alive or self._observer is None or self._feed is None:
                raise RuntimeError("仿真进程已停止，请点击「重置场景」重新启动。")
            state = self._observer.read()
            pose = (float(state["trunk"][0]), float(state["trunk"][1]))
            self._last_yaw = float(state["yaw"])
            if not self.trail or math.dist(self.trail[-1], pose) >= 0.005:
                self.trail.append(pose)
            distance = math.dist(pose, (self.home["x"], self.home["y"]))
            spectator_jpg, camera_jpg = self._feed.frames()
            overview_jpg = jpeg(draw_overview(self.scene, self.home, self.trail, self._last_yaw)) if draw_map else None
            return spectator_jpg, camera_jpg, overview_jpg, state, distance

    def cancel(self):
        """Signal active skill/agent execution to stop and immediately brake the robot."""
        runs = ROOT / "runs"
        runs.mkdir(exist_ok=True)
        (runs / "cancel_navigation").write_text(str(time.time_ns()), encoding="utf-8")
        if self.info and self.alive:
            from .navigation import Robot
            try:
                robot = Robot(self.info["socket"])
                try:
                    robot.stop()
                finally:
                    robot.close()
            except OSError:
                pass

    def reset(self, seed=None):
        """Stop the current simulator and start a fresh one with the given seed."""
        target_seed = self.seed if seed is None else int(seed)
        self.cancel()
        with self._lock:
            self._close_unlocked()
            self._start(target_seed)

    def _close_unlocked(self):
        if self._feed is not None:
            self._feed.close()
            self._feed = None
        if self._observer is not None:
            try:
                self._observer.close()
            except OSError:
                pass
            self._observer = None
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process_kill = self._process
                    process_kill.kill()
                    process_kill.wait()
            self._process = None
        if self._tempdir is not None:
            try:
                self._tempdir.cleanup()
            except OSError:
                pass
            self._tempdir = None
        self.info = None

    def close(self):
        with self._lock:
            self._close_unlocked()


@contextmanager
def simulation(seed):
    """Start the headless simulator, then stop it on success, failure or interruption."""
    session = SimSession(seed=seed)
    try:
        yield session.info
    finally:
        session.close()


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
    Only changed pictures are sent, which keeps a 10-15 Hz refresh light.
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


class UiBridge:
    """Delivers widget updates made by worker threads while no cell is running.

    Colab drops widget messages the kernel sends on its own between cells (2026-11: an update from
    a thread or an event-loop callback never reaches the page), but delivers those sent while it
    answers the page. So on Colab the page polls, and each poll applies the newest pending updates.
    Elsewhere the kernel's event loop applies them.
    """

    LANES = 5  # polls in flight at once, so the refresh rate is not capped by one round trip

    def __init__(self):
        self.main_thread = threading.get_ident()
        self.lock = threading.Lock()
        self.pending = {}
        self.closed = False
        self.polls = 0
        self.ioloop = None
        try:
            from google.colab import output

            self.colab = output
        except ImportError:
            self.colab = None
            try:
                from tornado.ioloop import IOLoop

                self.ioloop = IOLoop.current(instance=False)
            except Exception:
                pass

    def post(self, key, fn):
        """Run fn on the main thread; a newer update with the same key replaces an undelivered one."""
        if threading.get_ident() == self.main_thread:
            with self.lock:
                self.pending.pop(key, None)
            fn()
            return
        with self.lock:
            self.pending[key] = fn
        if self.colab is None:
            if self.ioloop is not None:
                self.ioloop.add_callback(self.flush)
            else:
                self.flush()

    def flush(self):
        with self.lock:
            updates, self.pending = list(self.pending.values()), {}
        for update in updates:
            update()

    def start_polling(self):
        """Call right after displaying the widgets: the polling script lives in the same output."""
        if self.colab is None:
            return
        from IPython.display import JSON, Javascript, display

        def poll():
            self.polls += 1
            self.flush()
            return JSON({"stop": self.closed})

        name = f"microduck.poll.{id(self)}"
        self.colab.register_callback(name, poll)
        display(Javascript("""
            (() => {
              const wait = ms => new Promise(done => setTimeout(done, ms));
              const lane = async offset => {
                await wait(offset);
                for (let failures = 0; failures < 20;) {
                  try {
                    const reply = await google.colab.kernel.invokeFunction(%s, [], {});
                    if (reply.data["application/json"].stop) return;
                    failures = 0;
                  } catch (error) {  // kernel restarting, or gone
                    failures += 1;
                    await wait(1000);
                  }
                  await wait(30);
                }
              };
              for (let i = 0; i < %d; i++) lane(i * 60);
            })();
        """ % (json.dumps(name), self.LANES)))


class InteractiveConsole:
    """Native ipywidgets interactive console built on 15 Hz Spectator + async physics."""

    def __init__(self, planner=None, seed=DEMO_SEED):
        import ipywidgets as widgets

        self.planner = planner
        self.session = SimSession(seed=seed)
        self.ui = UiBridge()
        self._stop_refresh = threading.Event()
        self._wake_refresh = threading.Event()
        self._busy_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._busy = False
        self._resetting = False
        self._inferring = False
        self._task_label = ""
        self._task_color = "#2563eb"
        self._task_started = 0.0
        self._worker_thread = None
        self._ask_event = threading.Event()
        self._ask_answer = None
        self._logs = []
        self._last_report = None
        self._last_refresh_warn = 0.0
        self._latest_bytes = {}
        self._next_map = 0.0

        title_text = "MicroDuck × Qwen-VL · 用中文指挥机器人" if planner is not None else "MicroDuck · 官方步态交互遥控"
        self.title_html = widgets.HTML(f"<h3 style='margin:4px 0 8px 0'>{html.escape(title_text)}</h3>")
        self.seed_input = widgets.IntText(value=int(seed), description="Seed:", layout=widgets.Layout(width="160px"))
        self.reset_btn = widgets.Button(description="重置场景", icon="refresh", button_style="info",
                                        layout=widgets.Layout(width="115px"))
        self.stop_btn = widgets.Button(description="紧急停止", icon="stop", button_style="danger",
                                       layout=widgets.Layout(width="115px"))
        self.status_html = widgets.HTML()

        self.spectator_img = widgets.Image(format="jpeg", layout=widgets.Layout(width="450px"))
        self.camera_img = widgets.Image(format="jpeg", layout=widgets.Layout(width="450px"))
        self.overview_img = widgets.Image(format="jpeg", layout=widgets.Layout(width="255px"))
        self.log_html = widgets.HTML()

        # Manual control widgets (shown only in the official baseline console)
        self.dist_slider = widgets.FloatSlider(
            value=0.20, min=0.05, max=0.35, step=0.05, description="步长 (m):",
            readout_format=".2f", layout=widgets.Layout(width="320px"),
        )
        self.deg_slider = widgets.IntSlider(
            value=45, min=15, max=120, step=15, description="转角 (°):",
            layout=widgets.Layout(width="320px"),
        )
        btn_layout = widgets.Layout(width="105px", height="36px")
        self.btn_fwd = widgets.Button(description="前进", icon="arrow-up", button_style="primary", layout=btn_layout)
        self.btn_back = widgets.Button(description="后退", icon="arrow-down", layout=btn_layout)
        self.btn_left = widgets.Button(description="左转", icon="rotate-left", layout=btn_layout)
        self.btn_right = widgets.Button(description="右转", icon="rotate-right", layout=btn_layout)
        self.btn_halt = widgets.Button(description="停止", icon="stop-circle", button_style="warning", layout=btn_layout)
        spacer = widgets.Box(layout=widgets.Layout(width="105px", height="36px"))
        dpad = widgets.VBox([
            widgets.HBox([spacer, self.btn_fwd, spacer]),
            widgets.HBox([self.btn_left, self.btn_halt, self.btn_right]),
            widgets.HBox([spacer, self.btn_back, spacer]),
        ], layout=widgets.Layout(gap="6px"))
        sliders = widgets.VBox([self.dist_slider, self.deg_slider], layout=widgets.Layout(justify_content="center"))
        self.manual_panel = widgets.HBox([dpad, sliders], layout=widgets.Layout(gap="24px", padding="8px 0"))

        self.btn_fwd.on_click(lambda _: self._run_manual("forward"))
        self.btn_back.on_click(lambda _: self._run_manual("backward"))
        self.btn_left.on_click(lambda _: self._run_manual("left"))
        self.btn_right.on_click(lambda _: self._run_manual("right"))
        self.btn_halt.on_click(lambda _: self._handle_stop())

        self.reset_btn.on_click(lambda _: self._handle_reset())
        self.stop_btn.on_click(lambda _: self._handle_stop())

        if planner is not None:
            preset_buttons = []
            for label, prompt_text in PRESET_INSTRUCTIONS:
                btn = widgets.Button(description=label, layout=widgets.Layout(width="auto"))
                btn.on_click(lambda _, text=prompt_text: self._fill_and_send(text))
                preset_buttons.append(btn)
            self.preset_box = widgets.HBox(preset_buttons, layout=widgets.Layout(flex_flow="row wrap", gap="6px"))
            self.prompt_input = widgets.Text(
                value=DEMO_INSTRUCTION,
                placeholder="输入中文指令，例如：去别的房间找蓝色的球，走到它旁边，然后回起点",
                layout=widgets.Layout(width="760px"),
            )
            self.send_btn = widgets.Button(description="发送指令", icon="paper-plane", button_style="success",
                                           layout=widgets.Layout(width="120px"))
            self.send_btn.on_click(lambda _: self._handle_send())

            self.ask_prompt_html = widgets.HTML()
            self.ask_input = widgets.Text(placeholder="输入补充说明、新目标（如：改为找红球）…",
                                          layout=widgets.Layout(width="520px"))
            self.ask_send_btn = widgets.Button(description="回复并继续", button_style="primary",
                                               layout=widgets.Layout(width="110px"))
            self.ask_retry_btn = widgets.Button(description="重试", button_style="info",
                                                layout=widgets.Layout(width="80px"))
            self.ask_cancel_btn = widgets.Button(description="取消", button_style="warning",
                                                 layout=widgets.Layout(width="80px"))
            self.ask_send_btn.on_click(lambda _: self._submit_ask(self.ask_input.value.strip()))
            self.ask_retry_btn.on_click(lambda _: self._submit_ask("重试"))
            self.ask_cancel_btn.on_click(lambda _: self._submit_ask("取消"))
            self.ask_box = widgets.VBox([
                self.ask_prompt_html,
                widgets.HBox([self.ask_input, self.ask_send_btn, self.ask_retry_btn, self.ask_cancel_btn],
                             layout=widgets.Layout(gap="6px")),
            ], layout=widgets.Layout(
                display="none", border="1px solid #f59e0b", padding="8px 12px",
                margin="8px 0", background_color="#fffbeb",
            ))

            qwen_panel = widgets.VBox([
                widgets.HTML("<div style='font-size:13px;color:#475569;margin-bottom:4px'>快捷指令（点击即执行）：</div>"),
                self.preset_box,
                widgets.HBox([self.prompt_input, self.send_btn], layout=widgets.Layout(gap="8px", margin="8px 0 4px 0")),
                self.ask_box,
            ])
            language, manual = [qwen_panel], []
        else:
            language, manual = [], [self.manual_panel]

        header = widgets.HBox(
            [self.seed_input, self.reset_btn, self.stop_btn, self.status_html],
            layout=widgets.Layout(align_items="center", gap="10px", margin="0 0 8px 0"),
        )
        views = widgets.HBox([
            widgets.VBox([widgets.HTML("<b>第三视角 · 跟随小鸭子</b>"), self.spectator_img]),
            widgets.VBox([widgets.HTML("<b>机器人实时第一视角</b>"), self.camera_img]),
            widgets.VBox([widgets.HTML("<b>仿真俯视图 · 白点为起点</b>"), self.overview_img]),
        ], layout=widgets.Layout(flex_flow="row wrap", grid_gap="12px"))

        self.root = widgets.VBox([
            self.title_html,
            header,
            *language,
            views,
            *manual,
            widgets.HTML("<div style='font-weight:600;margin-top:8px'>运行日志</div>"),
            self.log_html,
        ], layout=widgets.Layout(max_width="1220px"))

        self._append_log("仿真已就绪（2×2 房间，Seed=%d）。场景常驻运行，可连续操控或点击「重置场景」。" % self.session.seed)
        self._refresh_once(force_map=True)
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _status_markup(self, text, color="#16a34a"):
        return (
            f"<span style='display:inline-block;padding:3px 10px;border-radius:12px;"
            f"background:{color};color:#fff;font-size:13px;font-weight:600'>{html.escape(text)}</span>"
        )

    def _set_status(self, text, color="#16a34a"):
        self._task_label = text
        self._task_color = color
        markup = self._status_markup(text, color)
        self.ui.post("status", lambda: setattr(self.status_html, "value", markup))

    def _append_log(self, message, level="info"):
        colors = {"info": "#334155", "action": "#0369a1", "ok": "#15803d", "warn": "#b45309", "error": "#b91c1c"}
        stamp = time.strftime("%H:%M:%S")
        color = colors.get(level, "#334155")
        with self._log_lock:
            self._logs.append(f"<div style='color:{color};margin:2px 0'>[{stamp}] {html.escape(message)}</div>")
            self._logs = self._logs[-40:]
            html_value = (
                "<div style='max-height:200px;overflow-y:auto;border:1px solid #cbd5e1;"
                "border-radius:6px;padding:8px 10px;background:#f8fafc;font:13px/1.5 ui-monospace,monospace'>"
                + "".join(reversed(self._logs))
                + "</div>"
            )
            self.ui.post("log", lambda: setattr(self.log_html, "value", html_value))

    def _refresh_once(self, *, force_map=False):
        if self._resetting:
            return
        now = time.monotonic()
        need_map = force_map or now >= self._next_map
        try:
            spectator_jpg, camera_jpg, overview_jpg, state, dist = self.session.sample(draw_map=need_map)
            if need_map:
                self._next_map = now + 0.35
            yaw_deg = round(math.degrees(state["yaw"]))
            if self._busy:
                elapsed = max(0, int(now - self._task_started))
                status_markup = self._status_markup(
                    f"{self._task_label or '执行中…'} · {elapsed}s · 朝向 {yaw_deg}° · 距起点 {dist:.2f} m",
                    self._task_color,
                )
            else:
                status_markup = self._status_markup(f"待命 · 朝向 {yaw_deg}° · 距起点 {dist:.2f} m", "#16a34a")

            for key, data, image in (("spectator", spectator_jpg, self.spectator_img),
                                     ("camera", camera_jpg, self.camera_img),
                                     ("overview", overview_jpg, self.overview_img)):
                if data is not None and data != self._latest_bytes.get(key):
                    self._latest_bytes[key] = data
                    self.ui.post(key, lambda image=image, data=data: setattr(image, "value", data))
            self.ui.post("status", lambda: setattr(self.status_html, "value", status_markup))
        except Exception as error:
            if self._resetting:
                return
            if not self.session.alive:
                self._set_status("仿真已退出，请点「重置场景」", color="#dc2626")
            else:
                if now - self._last_refresh_warn >= 10.0:
                    self._last_refresh_warn = now
                    self._append_log(f"画面刷新异常：{error}", level="warn")

    def _refresh_loop(self):
        while not self._stop_refresh.is_set():
            tick = time.monotonic()
            if self.session.alive and not self._inferring and not self._resetting:
                self._refresh_once()
            target_dt = 0.08 if self._busy else 0.25  # ~12 Hz when moving, ~4 Hz when idle
            wait_s = max(0.02, target_dt - (time.monotonic() - tick))
            self._wake_refresh.wait(wait_s)
            self._wake_refresh.clear()

    def _start_worker(self, target):
        with self._busy_lock:
            if self._resetting:
                self._append_log("场景正在重置中，请等待几秒完成后再发送指令。", level="warn")
                return False
            if self._busy:
                self._append_log("当前有任务正在执行，请先等待完成或点击「紧急停止」。", level="warn")
                return False
            if not self.session.alive or self.session.info is None:
                self._append_log("仿真未就绪，请先点击「重置场景」。", level="warn")
                return False
            self._busy = True
            self._task_started = time.monotonic()
        self._wake_refresh.set()
        self._worker_thread = threading.Thread(target=self._run_guarded, args=(target,), daemon=True)
        self._worker_thread.start()
        return True

    def _run_guarded(self, target):
        try:
            target()
        except Exception as error:
            self._set_status("执行出错", color="#dc2626")
            self._append_log(f"执行异常：{type(error).__name__}: {error}", level="error")
            if not self.session.alive:
                tail = self.session.read_log_tail(400).strip()
                if tail:
                    self._append_log(f"仿真日志：{tail}", level="error")
        finally:
            self._hide_ask_box()
            with self._busy_lock:
                self._busy = False
            if self.session.alive and not self._resetting:
                self._refresh_once(force_map=True)

    def _hide_ask_box(self):
        if self.planner is not None:
            self.ui.post("ask", lambda: setattr(self.ask_box.layout, "display", "none"))

    def _submit_ask(self, answer):
        self._ask_answer = answer
        self._ask_event.set()

    def _ask_callback(self, prompt_text):
        self._ask_event.clear()
        self._ask_answer = None
        prompt_markup = (
            f"<div style='color:#92400e;font-weight:600;white-space:pre-wrap'>{html.escape(prompt_text)}</div>"
        )

        def show_ask():
            self.ask_prompt_html.value = prompt_markup
            self.ask_input.value = ""
            self.ask_box.layout.display = "flex"

        self.ui.post("ask", show_ask)
        self._set_status("等待你的回复…", color="#d97706")
        self._append_log(f"机器人询问：{prompt_text.splitlines()[0]}", level="warn")
        self._ask_event.wait()
        self._hide_ask_box()
        answer = self._ask_answer if self._ask_answer is not None else "取消"
        self._append_log(f"你的回复：{answer}", level="info")
        self._set_status("继续执行中…", color="#2563eb")
        return answer

    def _run_manual(self, direction):
        from .skills import SkillExecutor

        if direction in ("forward", "backward"):
            dist = round(float(self.dist_slider.value), 2)
            action = {"action": "move", "direction": direction, "distance_m": dist}
            label = f"手动{'前进' if direction == 'forward' else '后退'} {dist:.2f} m"
        else:
            deg = int(self.deg_slider.value)
            action = {"action": "turn", "direction": direction, "degrees": deg}
            label = f"手动{'左转' if direction == 'left' else '右转'} {deg}°"

        def task():
            self._set_status(f"执行中：{label}", color="#2563eb")
            self._append_log(f"▶ {label}", level="action")
            with tempfile.TemporaryDirectory() as directory:
                executor = SkillExecutor(self.session.info, Path(directory),
                                         cancel_file=ROOT / "runs/cancel_navigation")
                try:
                    result = executor.run(action, [])
                finally:
                    executor.close()
            if result.get("success"):
                self._append_log(f"✓ {label} 完成", level="ok")
            else:
                self._append_log(f"⚠ {label} 未完成：{result.get('error', '已停车')}", level="warn")

        self._start_worker(task)

    def _fill_and_send(self, text):
        self.prompt_input.value = text
        self._handle_send()

    def _handle_send(self):
        instruction = self.prompt_input.value.strip()
        if not instruction:
            self._append_log("请输入一条中文指令。", level="warn")
            return

        from .agent import run_instruction

        console = self

        class LiveReportingPlanner:
            def predict(self, text, image, memory):
                console._set_status("Qwen 推理中…", color="#7c3aed")
                console._append_log("⏳ 正在调用 Qwen3-VL 推理下一步技能…", level="info")
                console._inferring = True
                try:
                    prediction = console.planner.predict(text, image, memory)
                finally:
                    console._inferring = False
                action = prediction.get("action")
                latency = prediction.get("latency_s", 0.0)
                if action is not None:
                    console._append_log(
                        f"🤖 模型决策：{_format_action(action)} ({latency:.1f}s) | {prediction.get('raw', '').strip()}",
                        level="action",
                    )
                    if action.get("action") == "explore":
                        console._append_log("🔍 开始原地环视 360° 并跨房间搜索目标…", level="info")
                    console._set_status(f"执行动作：{action.get('action')}", color="#2563eb")
                else:
                    console._append_log(
                        f"⚠ 模型输出无法解析：{prediction.get('error', prediction.get('raw'))}",
                        level="warn",
                    )
                return prediction

        def task():
            self._append_log(f"▶ 收到中文指令：{instruction}", level="info")
            report = run_instruction(
                LiveReportingPlanner(), instruction, self.session.info,
                ask=self._ask_callback,
            )
            self._last_report = report
            pose = self.session.trail[-1]
            dist = math.dist(pose, (self.session.home["x"], self.session.home["y"]))
            error = next(
                (step["execution"]["error"] for step in reversed(report["steps"])
                 if step.get("execution", {}).get("error")),
                None,
            )
            summary = f"结束状态：{report['status']} · 距起点 {dist:.2f} m"
            if error:
                summary += f" · {error}"
            level = "ok" if report["status"] in ("model_done", "model_stop") else "warn"
            self._append_log(summary, level=level)

        self._start_worker(task)

    def _handle_stop(self):
        self._append_log("⏹ 收到紧急停止命令", level="warn")
        self.session.cancel()
        if not self._ask_event.is_set():
            self._ask_answer = "取消"
            self._ask_event.set()
        self._set_status("已请求停止", color="#d97706")

    def _handle_reset(self):
        with self._busy_lock:
            if self._resetting:
                self._append_log("场景正在重置中，请稍候…", level="warn")
                return
            self._resetting = True
        seed = int(self.seed_input.value)
        self._handle_stop()

        def reset_task():
            try:
                self._set_status(f"正在重置场景 (Seed={seed})…", color="#0284c7")
                if self._worker_thread is not None and self._worker_thread.is_alive():
                    self._worker_thread.join(timeout=8)
                self._latest_bytes.clear()
                self.session.reset(seed)
                self._append_log(f"🔄 场景已重置（Seed={seed}）", level="ok")
            except Exception as error:
                self._set_status("重置失败", color="#dc2626")
                self._append_log(f"重置场景失败：{type(error).__name__}: {error}", level="error")
            finally:
                with self._busy_lock:
                    self._resetting = False
                if self.session.alive:
                    self._refresh_once(force_map=True)
                    self._wake_refresh.set()

        threading.Thread(target=reset_task, daemon=True).start()

    def display(self):
        from IPython.display import display
        display(self.root)
        self.ui.start_polling()
        return self

    def close(self):
        self.ui.closed = True
        self._stop_refresh.set()
        self._wake_refresh.set()
        self._handle_stop()
        self.session.close()


def _close_previous_console():
    global _ACTIVE_CONSOLE
    prev = getattr(sys, "_microduck_active_console", None) or _ACTIVE_CONSOLE
    if prev is not None:
        try:
            prev.close()
        except Exception:
            pass
    _ACTIVE_CONSOLE = None
    setattr(sys, "_microduck_active_console", None)


def interactive_demo(planner, seed=DEMO_SEED):
    """Launch the interactive ipywidgets console driven by Chinese instructions."""
    global _ACTIVE_CONSOLE
    _close_previous_console()
    _ACTIVE_CONSOLE = InteractiveConsole(planner=planner, seed=seed)
    setattr(sys, "_microduck_active_console", _ACTIVE_CONSOLE)
    return _ACTIVE_CONSOLE.display()


def interactive_baseline(seed=DEMO_SEED):
    """Launch the interactive ipywidgets manual remote-control console (CPU-friendly)."""
    global _ACTIVE_CONSOLE
    _close_previous_console()
    _ACTIVE_CONSOLE = InteractiveConsole(planner=None, seed=seed)
    setattr(sys, "_microduck_active_console", _ACTIVE_CONSOLE)
    return _ACTIVE_CONSOLE.display()


def record_colab_gif(
    planner,
    instruction=DEMO_INSTRUCTION,
    seed=DEMO_SEED,
    output_path="/content/microduck-navigation-demo.gif",
    *,
    speed=5,
    capture_interval_s=0.45,
    drive_folder="microduck-qwen",
):
    """Run one clean episode and export the 1220x650 third-person + PiP trajectory README GIF."""
    from IPython.display import Image as IPyImage, display
    from PIL import Image

    from .agent import run_instruction
    from .demo_gif import compose_demo_frame, quantize_demo_frame, save_demo_gif
    from .navigation import Observer

    _close_previous_console()
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("请输入一条中文指令。")

    decisions_log = []
    decision_lock = threading.Lock()

    class RecordingPlanner:
        def predict(self, text, image, memory):
            prediction = planner.predict(text, image, memory)
            action = prediction.get("action")
            with decision_lock:
                decisions_log.append({"action": action, "image": str(image)})
            return prediction

    board = Board("MicroDuck × Qwen-VL · 正在录制第三视角 GIF", instruction, "模型决策")
    raw_frames = []
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        with simulation(seed) as info:
            scene = json.loads(Path(info["scene_file"]).read_text(encoding="utf-8"))
            home = info["home_pose"]
            observer, feed = Observer(info["port"]), AudienceFeed(info["port"])
            trail = [(home["x"], home["y"])]
            started = time.monotonic()
            next_capture = started
            next_map = started
            try:
                task = worker.submit(run_instruction, RecordingPlanner(), instruction, info)
                while True:
                    tick = time.monotonic()
                    state = observer.read()
                    pose = (float(state["trunk"][0]), float(state["trunk"][1]))
                    if math.dist(trail[-1], pose) >= 0.005:
                        trail.append(pose)
                    spectator_jpg, camera_jpg = feed.frames()
                    with decision_lock:
                        cur_decisions = list(decisions_log)
                    action_names = [
                        (d["action"].get("action", "?") if isinstance(d["action"], dict) else "?")
                        for d in cur_decisions
                    ]
                    report = task.result() if task.done() else None
                    overview_img = None
                    overview_jpg = None
                    if report is not None or tick >= next_map or tick >= next_capture:
                        overview_img = draw_overview(scene, home, trail, state["yaw"])
                        overview_jpg = jpeg(overview_img)
                        next_map = tick + 0.35
                    if spectator_jpg and camera_jpg and (report is not None or tick >= next_capture):
                        dist = math.dist(pose, (home["x"], home["y"]))
                        raw_frames.append({
                            "spectator_jpg": spectator_jpg,
                            "camera_jpg": camera_jpg,
                            "overview_img": overview_img or draw_overview(scene, home, trail, state["yaw"]),
                            "decisions": [d["action"] for d in cur_decisions if d["action"] is not None],
                            "model_image": cur_decisions[-1]["image"] if cur_decisions else None,
                            "elapsed": tick - started,
                            "distance": dist,
                            "done": report is not None,
                        })
                        next_capture = tick + capture_interval_s
                    summary = ""
                    if report is not None:
                        dist = math.dist(pose, (home["x"], home["y"]))
                        summary = f"录制采样完成 ({len(raw_frames)} 帧) · 状态 {report['status']} · 距起点 {dist:.2f} m"
                    board.update(action_names, tick - started, summary,
                                 spectator=spectator_jpg, camera=camera_jpg, overview=overview_jpg)
                    if report is not None:
                        break
                    time.sleep(max(0.0, REFRESH_S - (time.monotonic() - tick)))
            finally:
                feed.close()
                observer.close()
    finally:
        worker.shutdown(wait=False, cancel_futures=True)

    print(f"正在合成 {len(raw_frames)} 帧第三视角 README GIF…", flush=True)
    composed, timestamps = [], []
    for item in raw_frames:
        spec_img = Image.open(io.BytesIO(item["spectator_jpg"])).convert("RGB")
        cam_img = Image.open(io.BytesIO(item["camera_jpg"])).convert("RGB")
        canvas = compose_demo_frame(
            spec_img, cam_img, item["overview_img"], instruction,
            item["decisions"], item["elapsed"], item["distance"],
            speed=speed, done=item["done"],
        )
        composed.append(quantize_demo_frame(canvas, colors=128))
        timestamps.append(item["elapsed"])

    gif_info = save_demo_gif(composed, timestamps, output_path, speed=speed)
    drive_dest = None
    if drive_folder:
        drive_dir = Path("/content/drive/MyDrive") / drive_folder
        if drive_dir.is_dir():
            drive_dest = drive_dir / Path(output_path).name
            shutil.copyfile(output_path, drive_dest)
            gif_info["drive_path"] = str(drive_dest)

    size_mb = gif_info["size_bytes"] / (1024 * 1024)
    print(f"GIF 导出完成：{gif_info['path']} ({gif_info['frames']} 帧, {size_mb:.2f} MB, {gif_info['duration_s']}s)")
    if drive_dest:
        print(f"已同步保存到 Google Drive：{drive_dest}")
    display(IPyImage(filename=gif_info["path"]))
    return {"report": report, "gif": gif_info}


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
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        with simulation(seed) as info:
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
                    report, exc = None, None
                    if task.done():
                        try:
                            report = task.result()
                        except Exception as error:
                            exc = error
                    summary, overview = "", None
                    if report is not None or exc is not None or tick >= next_map:
                        overview, next_map = jpeg(draw_overview(scene, home, trail, state["yaw"])), tick + 0.3
                    if report is not None:
                        distance = math.dist(pose, (home["x"], home["y"]))
                        summary = f"结束状态 {report['status']} · 距起点 {distance:.2f} m · done 是模型声明"
                        error = next((step["execution"]["error"] for step in reversed(report["steps"])
                                      if step.get("execution", {}).get("error")), None)
                        if error:
                            summary += f" · {error}"
                    elif exc is not None:
                        summary = f"执行异常：{type(exc).__name__}: {exc}"
                    spectator, camera = feed.frames()
                    board.update(actions, tick - started, summary, spectator=spectator, camera=camera, overview=overview)
                    if exc is not None:
                        raise exc
                    if report is not None:
                        return report
                    time.sleep(max(0.0, REFRESH_S - (time.monotonic() - tick)))
            finally:
                feed.close()
                observer.close()
    finally:
        worker.shutdown(wait=False, cancel_futures=True)


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
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        with tempfile.TemporaryDirectory() as directory, simulation(seed) as info:
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
                    result, exc = None, None
                    if task.done():
                        try:
                            result = task.result()
                        except Exception as error:
                            exc = error
                    overview = None
                    if result is not None or exc is not None or tick >= next_map:
                        overview = jpeg(draw_overview(scene, info["home_pose"], trail, state["yaw"]))
                        next_map = tick + 0.3
                    summary = ""
                    if result is not None:
                        distance = math.dist((info["home_pose"]["x"], info["home_pose"]["y"]), trail[-1])
                        summary = f"执行结束 · 位移 {distance:.2f} m" if result["success"] else result.get("error", "已停车")
                    elif exc is not None:
                        summary = f"执行异常：{type(exc).__name__}: {exc}"
                    spectator, camera = feed.frames()
                    board.update([action["action"]], tick - started, summary,
                                 spectator=spectator, camera=camera, overview=overview)
                    if exc is not None:
                        raise exc
                    if result is not None:
                        return result
                    time.sleep(max(0.0, REFRESH_S - (time.monotonic() - tick)))
            finally:
                feed.close()
                observer.close()
    finally:
        worker.shutdown(wait=False, cancel_futures=True)
