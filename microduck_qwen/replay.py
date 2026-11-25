"""Side-by-side replays of one recorded task under four control setups, for the Qwen notebook.

Reads evaluation runs (agent_eval). The trajectory is the logged simulator pose, the text is the model's
raw output, and the third-person view re-renders the same scene at that pose. Runs with a pose log
(state_log.py) show every joint as recorded; older runs logged only position and heading, so their
panels show the default stance and say so. Waiting for inference is shortened to a fixed hold; motion
is sped up by the same factor in all four panels.

    python -m microduck_qwen.replay --out assets/replay                     # cut runs
    python -m microduck_qwen.replay --out assets/replay --uncut             # free-run runs (harness arms)
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

from . import ROOT
from .colab_demo import draw_overview
from .sim_native import DEFAULT_RL
from .truth_score import score_run

SEEDS = (99200, 99300)
# label, cut run, uncut run. The no-harness run never had a teacher cut.
ARMS = (
    ("raw", "① 裸 Qwen：只有基本动作", "raw", "raw"),
    ("frozen", "② 加技能框架", "frozen", "free-frozen"),
    ("frozterm", "③ 再加结束规则", "frozterm", "free-frozterm"),
    ("lora", "④ 再加 LoRA（完整系统）", "mapoff-v2", "free-lora"),
)
OK, FAIL, CUT = "success", "fail", "cut"
TASK_TYPES = {3: "走近并返回", 4: "定距移动"}  # anything else is a cross-room search
PANEL, GAP, HEADER = (600, 330), 10, 46
FRAME_S, HOLD_S, END_S = 0.15, 1.1, 2.5
MAX_MOTION_S = 11.0  # the slowest panel's motion is sped up to fit this

REASON_ZH = {
    "step budget of 40 exhausted without done or stop": "40 次决策用完，没有结束任务",
    "step limit of 5 reached without a terminal decision": "步数用完，没有结束任务",
    "no terminal decision": "没有给出结束判断", "invalid model output": "输出无法执行",
    "touched the scene": "碰到了障碍", "target never reached": "始终没走到目标",
    "left the target before ending": "结束前离开了目标",
    "ended with clarify": "反问用户，没有完成任务", "ended with unsupported": "认为做不到，没有完成任务",
    "failed physical skill": "技能没有执行成功", "skill failed": "技能没有执行成功",
}


def outcome(verdict):
    return CUT if verdict.startswith("cut") else verdict


def load_scores(uncut=False, runs=ROOT / "runs/lang"):
    scores = {}
    for key, _label, cut_run, uncut_run in ARMS:
        merged = {}
        for seed in SEEDS:
            merged.update(score_run(runs / f"{uncut_run if uncut else cut_run}-{seed}"))
        scores[key] = merged
    return scores


def task_type(task_id):
    return TASK_TYPES.get(task_id, "跨房间找球")


def select(scores, total=8):
    """One task per outcome pattern seen at least twice, then fill by a fixed rule; no hand picking.

    Returns [{"key": (seed, task), "pattern": (...), "count": n, "why": text}].
    """
    order = [arm[0] for arm in ARMS]
    keys = sorted(set.intersection(*(set(scores[arm]) for arm in order)))
    pattern = {key: tuple(outcome(scores[arm][key]["verdict"]) for arm in order) for key in keys}
    counts = Counter(pattern.values())
    ranked = sorted(counts, key=lambda p: (-counts[p], p))
    chosen = []

    def add(key, why):
        if len(chosen) < total and key not in [item["key"] for item in chosen]:
            chosen.append({"key": key, "pattern": pattern[key], "count": counts[pattern[key]], "why": why})

    for p in ranked:
        if counts[p] >= 2:
            add(next(k for k in keys if pattern[k] == p), "该结果组合的第一题")
    seen = {task_type(item["key"][1]) for item in chosen if item["pattern"] == ranked[0]}
    for key in keys:
        if pattern[key] == ranked[0] and task_type(key[1]) not in seen:
            seen.add(task_type(key[1]))
            add(key, "最常见组合中的另一类任务")
    for key in keys:
        if pattern[key][-1] != OK:
            add(key, "完整系统没有成功的题")
    return sorted(chosen, key=lambda item: (-item["count"], item["key"])), keys


def action_text(prediction):
    action = prediction.get("action")
    if action is None:
        return "输出无法执行"
    names = {"move": "移动", "turn": "转向", "approach": "走近", "explore": "探索找", "search": "原地找", "return_home": "返回起点",
             "bypass_return": "绕行返回", "done": "任务完成", "stop": "到位停下", "clarify": "请求澄清", "unsupported": "不支持"}
    colours = {"red": "红球", "blue": "蓝球", "green": "绿球", "yellow": "黄球"}
    text = names.get(action["action"], action["action"])
    if action["action"] in ("move", "turn"):
        side = {"forward": "前", "backward": "后", "left": "左", "right": "右"}[action["direction"]]
        amount = f'{action["distance_m"] * 100:.0f} 厘米' if action["action"] == "move" else f'{action["degrees"]:.0f}°'
        return f"向{side}{text} {amount}"
    return text + colours.get(action.get("color"), "")


def reason_text(verdict, episode):
    last = (episode["steps"] or [{}])[-1]
    if verdict["verdict"] == "cut_deviation":
        return "评测在此截断：与参考动作不同"
    if verdict["verdict"] == "cut_teacher":
        return "评测在此截断：参考动作无法给出"
    if verdict["verdict"] == "cut_infra":
        return "仿真故障，不计"
    if verdict["verdict"] == OK:
        return f'成功：{action_text(last["prediction"])}'
    error = str(episode.get("error") or "")
    for source in (verdict["reason"], error):
        for key, text in REASON_ZH.items():
            if source.startswith(key):
                return "失败：" + text
    if " m from home" in verdict["reason"]:
        return f'失败：结束时离起点 {verdict["reason"].split()[1]} m'
    if "clearance too small" in error or "unsafe" in error:
        return "失败：前方不安全，已停车"
    return "失败：" + (verdict["reason"] or error)[:28]


class ThirdPerson:
    """Chase view of a recorded pose in its own scene; the same camera as sim_native.Spectator."""

    def __init__(self, scene_file, width=640, height=360):
        import sys

        import mujoco

        from .sim_body import configure_body

        if str(DEFAULT_RL / "src") not in sys.path:
            sys.path.insert(0, str(DEFAULT_RL / "src"))
        os.environ["MICRODUCK_VISION_SCENE"] = str(scene_file)
        self.mj = mujoco
        self.server = configure_body()
        self.model = self.server.build_world(self.server.DEFAULT_SCENE, 1)
        self.data = mujoco.MjData(self.model)
        self.root = int(self.model.joint("trunk_base_freejoint").qposadr[0])
        self.joints = [(i, int(self.model.joint(name).qposadr[0])) for i, name in enumerate(self.server.JOINT_NAMES)
                       if name != "mouth"]
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance, self.camera.elevation = 1.1, -48

    def shot(self, pose, azimuth):
        """pose = (x, y, yaw, recorded state or None); returns (picture, eased azimuth)."""
        from PIL import Image

        x, y, yaw, state = pose
        if state is None:
            xyz, quat, joints = (x, y, 0.1165), (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)), self.server.HOME_POSE
        else:
            xyz, quat, joints = state["xyz"], state["quat"], state["joints"]
        self.data.qpos[self.root:self.root + 3] = xyz
        self.data.qpos[self.root + 3:self.root + 7] = quat
        for wire, address in self.joints:
            self.data.qpos[address] = joints[wire]
        self.mj.mj_forward(self.model, self.data)
        degrees = math.degrees(yaw)
        azimuth = degrees if azimuth is None else azimuth + 0.35 * ((degrees - azimuth + 180) % 360 - 180)
        self.camera.lookat[:] = [x, y, 0.08]
        self.camera.azimuth = azimuth
        self.renderer.update_scene(self.data, camera=self.camera)
        return Image.fromarray(self.renderer.render()), azimuth

    def close(self):
        self.renderer.close()


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_states(directory):
    path = directory.with_name(directory.name + "-states.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def load_arm(directory):
    """Decisions with the motion each one caused: path is (sim_time, x, y, yaw, recorded state or None)."""
    episode = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    recorded = [step for step in episode["steps"] if step.get("memory")]
    states = load_states(directory)
    steps = []
    for index, step in enumerate(recorded):
        path = []
        if states and "decided_sim_time" in step:
            until = recorded[index + 1]["sim_time"] if index + 1 < len(recorded) else math.inf
            path = [(row["t"], *row["xyz"][:2], yaw_of(row["quat"]), row) for row in states
                    if step["sim_time"] <= row["t"] < until]
            # Standing still while the model thinks is shown as a fixed hold instead.
            path = [row for row in path if row[0] >= step["decided_sim_time"]] or path[-1:]
        else:
            # skill-001 is the return to the task start, so decision 0 is skill-002.
            for log in directory.glob(f"skill-{index + 2:03d}*.jsonl"):
                for line in log.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if "position" in row and "sim_time" in row:
                        path.append((row["sim_time"], *row["position"], row["yaw"], None))
            path.sort(key=lambda row: row[0])
        steps.append({"raw": step["prediction"].get("raw", ""), "text": action_text(step["prediction"]),
                      "path": path, "odometry": step["memory"]["odometry_relative_home"]})
    return episode, steps


def timeline(steps, home, speed):
    """[(duration_s, step, pose, trail_length)] plus the whole trail, in display time."""
    trail, frames = [], []
    for index, step in enumerate(steps):
        o = step["odometry"]
        rest = (home["x"] + o["x"], home["y"] + o["y"], home["yaw"] + math.radians(o["yaw_deg"]),
                step["path"][0][4] if step["path"] else trail[-1][3] if trail else None)
        trail.append(rest)
        frames.append((HOLD_S, index, rest, len(trail)))
        previous = None
        for time_s, x, y, yaw, state in step["path"]:
            trail.append((x, y, yaw, state))
            if previous is not None:
                frames.append((min(1.0, time_s - previous) / speed, index, trail[-1], len(trail)))
            previous = time_s
    return frames, trail


def motion_seconds(steps):
    return sum(min(1.0, b[0] - a[0]) for s in steps for a, b in zip(s["path"], s["path"][1:]))


def sample(frames, t):
    for duration, index, pose, length in frames:
        if t < duration:
            return index, pose, length
        t -= duration
    return frames[-1][1:]


def panel(arm, state, finished, view, fonts):
    from PIL import Image, ImageDraw

    width, height = PANEL
    image = Image.new("RGB", PANEL, "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), 10, outline="#cbd5e1", width=1)
    draw.text((14, 9), arm["label"], font=fonts["bold"], fill="#0f172a")
    index, pose, length = state
    trail = [p[:2] for p in arm["trail"][:length]] or [pose[:2]]
    image.paste(draw_overview(arm["scene"], arm["home"], trail, pose[2]).resize((232, 232)), (14, 42))
    picture, arm["azimuth"] = view.shot(pose, arm["azimuth"])
    picture = picture.resize((326, 183), Image.LANCZOS)
    image.paste(picture, (260, 42))
    if pose[3] is None:
        draw.text((268, 206), "腿部姿态未记录，画为默认站姿", font=fonts["small"], fill="#e2e8f0")
    step = arm["steps"][index]
    draw.text((260, 231), f"第 {index + 1}/{len(arm['steps'])} 次决策：{step['text']}", font=fonts["bold"], fill="#0369a1")
    raw = " ".join(step["raw"].split())
    draw.text((260, 255), raw[:46] + ("…" if len(raw) > 46 else ""), font=fonts["small"], fill="#64748b")
    draw.text((14, 283), "俯视图（白点为起点）", font=fonts["small"], fill="#64748b")
    if finished:
        draw.rounded_rectangle((260, 286, width - 14, 318), 8, fill=arm["colour"])
        draw.text((272, 292), arm["badge"], font=fonts["bold"], fill="#ffffff")
    return image


def render(key, scores, uncut, path, runs=ROOT / "runs/lang"):
    """Write one animated WebP; returns the facts the notebook prints next to it."""
    from PIL import Image, ImageDraw

    from .demo_gif import load_font

    seed, task_id = key
    fonts = {"title": load_font(20, bold=True), "bold": load_font(15, bold=True), "small": load_font(12)}
    group = next(s for s in SEEDS if s <= seed < s + 8)
    arms, speed = [], 1.0
    for name, label, cut_run, uncut_run in ARMS:
        seed_dir = runs / f"{uncut_run if uncut else cut_run}-{group}" / f"seed-{seed}"
        episode, steps = load_arm(seed_dir / f"task-{task_id:02d}")
        simulator = json.loads((seed_dir / "simulator-0.json").read_text(encoding="utf-8"))
        scene = json.loads((seed_dir / "simulator-0.scene.json").read_text(encoding="utf-8"))
        verdict = scores[name][key]
        speed = max(speed, motion_seconds(steps) / MAX_MOTION_S)
        arms.append({"label": label, "steps": steps, "scene": scene, "home": simulator["home_pose"], "verdict": verdict,
                     "badge": reason_text(verdict, episode),
                     "colour": {OK: "#15803d", FAIL: "#b91c1c", CUT: "#b45309"}[outcome(verdict["verdict"])],
                     "decisions": len(steps), "motion_s": motion_seconds(steps), "azimuth": None,
                     "posed": any(row[4] is not None for step in steps for row in step["path"]),
                     "scene_file": seed_dir / "simulator-0.scene.json"})
    instruction = scores["lora"][key]["instruction"]
    for arm in arms:
        arm["frames"], arm["trail"] = timeline(arm["steps"], arm["home"], speed)
        arm["length"] = sum(f[0] for f in arm["frames"])
    # A panel with many decisions gets shorter holds so the whole clip stays short.
    total = max(arm["length"] for arm in arms)
    stretch = min(1.0, 16.0 / total)
    width, height = 2 * PANEL[0] + GAP, HEADER + 2 * PANEL[1] + GAP
    view = ThirdPerson(arms[0]["scene_file"])  # the four arms share the layout of this seed
    if any(arm["scene"]["objects"] != arms[0]["scene"]["objects"] for arm in arms):
        raise ValueError("the arms were recorded in different scenes")
    pictures, clock = [], 0.0
    while clock <= total * stretch + END_S:
        canvas = Image.new("RGB", (width, height), "#f1f5f9")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 9), f"指令：{instruction}", font=fonts["title"], fill="#0f172a")
        for i, arm in enumerate(arms):
            t = clock / stretch
            done = t >= arm["length"]
            part = panel(arm, sample(arm["frames"], t), done, view, fonts)
            canvas.paste(part, ((i % 2) * (PANEL[0] + GAP), HEADER + (i // 2) * (PANEL[1] + GAP)))
        pictures.append(canvas.resize((width * 4 // 5, height * 4 // 5), Image.LANCZOS))
        clock += FRAME_S
    view.close()
    pictures[0].save(path, save_all=True, append_images=pictures[1:], loop=0, duration=round(FRAME_S * 1000), quality=50, method=4)
    return {"seed": seed, "task": task_id, "type": task_type(task_id), "instruction": instruction, "file": path.name,
            "speed": round(speed / stretch, 1), "frames": len(pictures), "bytes": path.stat().st_size,
            "arms": [{"label": a["label"], "outcome": outcome(a["verdict"]["verdict"]), "badge": a["badge"],
                      "decisions": a["decisions"], "posed": a["posed"], "motion_s": round(a["motion_s"], 1)} for a in arms]}


def totals(scores, keys):
    """Success / fail / cut per arm over the tasks every arm was scored on."""
    return {arm[0]: dict(Counter(outcome(scores[arm[0]][k]["verdict"]) for k in keys)) for arm in ARMS}


IMAGES = "https://raw.githubusercontent.com/atlasmarkz/microduck-qwen/main/assets/replay/"
NOTEBOOK = ROOT / "notebooks/microduck-live-demo.ipynb"
CELL = "replay-results"


def markdown(index):
    """The notebook's results cell: totals, clips and the selection rule, all from index.json."""
    cut = any(counts.get(CUT) for counts in index["totals"].values())
    lines = ["## 先看效果：同一句话，四种控制方式", "",
             "不用运行，也不用 GPU。四个面板是**同一个任务、同一个布局**的仿真实录，从左上到右下逐层加上本仓库的方法。"
             "每个面板左边是俯视轨迹，右边是**跟随机器人的第三视角**（只给观众看，模型看不到），下面是模型的决策和原始输出。"
             + ("" if all(arm["posed"] for clip in index["clips"] for arm in clip["arms"]) else
                "这批录像只记录了机器人的位置和朝向，腿部画为默认站姿；带完整关节姿态的补跑完成后会替换。"), "",
             "| 控制方式 | 成功 | 失败 |" + (" 评测截断 |" if cut else ""), "|---|---:|---:|" + ("---:|" if cut else "")]
    for key, label, *_runs in ARMS:
        counts = index["totals"][key]
        lines.append(f'| {label} | **{counts.get(OK, 0)}/{index["tasks"]}** | {counts.get(FAIL, 0)} |'
                     + (f" {counts.get(CUT, 0)} |" if cut else ""))
    lines += ["", f'{index["tasks"]} 个任务来自 16 个布局，模型训练和框架开发都没有用过（③ 的结束规则例外，见文末）。成败由仿真真值判定：是否到达目标、是否返回、有无碰撞、是否自己结束任务。', ""]
    if cut:
        lines += ["> **关于“评测截断”**：②③④ 用的评测会在模型第一次偏离参考动作时停止，所以这些题看不到自然结局，也不算成功。"
                  "② 的 27 题因此是下限，不是它自由运行的成绩。不截断的补跑完成后会替换这里的数字和录像。", ""]
    for clip in index["clips"]:
        lines += [f'**{clip["instruction"]}**　（{clip["type"]}；{index["tasks"]} 题中 {clip["pattern_count"]} 题是这种结果组合）', "",
                  f'![{clip["instruction"]}]({IMAGES}{clip["file"]})', ""]
    lines += ["<small>选题规则：出现不少于 2 次的每种结果组合各取编号最小的一题；再从最常见的组合里补上其他任务类型；最后补完整系统没有成功的题。没有手工挑选。"
              "录像里推理等待压缩为固定停顿，运动按同一倍数加速。</small>", "",
              "<small>适用范围：仿真、单一 4B 模型、红绿蓝黄四色球的 2×2 房间。③ 的结束规则是看过同一批种子的失败后写的，它和 ④ 的差别在这 80 题上不显著（p=0.18，截断口径）。</small>"]
    return "\n".join(lines)


def update_notebook(index, path=NOTEBOOK):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["id"] == CELL)
    cell["source"] = markdown(index).splitlines(keepends=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "assets/replay")
    parser.add_argument("--uncut", action="store_true", help="use the free-run recordings for the three harness arms")
    parser.add_argument("--list", action="store_true", help="only print the selection")
    args = parser.parse_args()
    scores = load_scores(args.uncut)
    chosen, keys = select(scores)
    for item in chosen:
        print(item["key"], item["pattern"], f'{item["count"]}/{len(keys)}', item["why"], scores["lora"][item["key"]]["instruction"])
    if args.list:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    clips = []
    for number, item in enumerate(chosen, 1):
        clip = render(item["key"], scores, args.uncut, args.out / f"task-{number}.webp")
        clips.append(dict(clip, pattern=item["pattern"], pattern_count=item["count"], why=item["why"]))
        print(json.dumps({k: clip[k] for k in ("file", "frames", "bytes", "speed")}), flush=True)
    index = {"uncut": args.uncut, "tasks": len(keys), "totals": totals(scores, keys), "clips": clips}
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    update_notebook(index)


if __name__ == "__main__":
    main()
