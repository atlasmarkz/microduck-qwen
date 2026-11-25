"""Run Pollen Robotics' published MicroDuck policies headless and render one short clip per skill.

Audience-only: open floor, official scenes, official ONNX weights, BAM M6 motors. Nothing here is
used by the navigation agent. Motor conversion and observation layout mirror
microduck_rl/scripts/infer_policy.py; upstream files are untouched.
"""

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

from . import ROOT
from .sim_body import reset_bam
from .sim_native import DEFAULT_POLICY, DEFAULT_RL

# Official set v10 on the Hub, 2026-11-12 (velstand rolled back to the v7 weights, new roller). The remote-control
# console keeps its own, older POLICY_REVISION so earlier experiments stay reproducible.
SHOWROOM_REVISION = "236d1835b03ac7b45d962efbf2fe3a6940206112"
POLICY_REPO = "pollen-robotics/microduck-policies"
SCENES = Path("src/mjlab_microduck/robot/microduck")
CONTROL_S, TIMESTEP, DECIMATION = 0.02, 0.005, 4
FRAME_TICKS = 3  # one picture every 60 ms, which is also an exact animation delay
BALL_AHEAD_M, BALL_SIDE_M, BALL_RADIUS_M = 0.09, 0.042, 0.035  # microduck_ball_kick_env_cfg
ROLLER_ACTION_SCALE = 0.8  # manifest.json action_scale for the two roller policies


def zeros(_t):
    return [0.0] * 13


def twist(vx=0.0, vy=0.0, wz=0.0):
    return lambda _t: [vx, vy, wz] + [0.0] * 10


def posture(sit):
    return lambda _t: [float(sit)] + [0.0] * 12


def phase(period):
    """One-shot phase clock in the twist slot: [cos, sin, 0]."""
    return lambda t: [math.cos(2 * math.pi * t / period), math.sin(2 * math.pi * t / period)] + [0.0] * 11


def body_sway(t):
    """Body-pose slots are [x, y, z, roll, pitch, yaw]; sway roll, then pitch, by 10 degrees."""
    angle = math.radians(10) * math.sin(2 * math.pi * t / 2.0)
    return [0.0] * 10 + ([angle, 0.0, 0.0] if t < 2.0 else [0.0, angle, 0.0])


@dataclass(frozen=True)
class Segment:
    policy: str  # "" = no policy, motors hold the default pose
    seconds: float
    caption: str
    command: object = zeros
    event: str = ""  # "kick_left" | "kick_right": put the ball in front of that foot first
    main: bool = False  # the policy this skill is about


@dataclass(frozen=True)
class Skill:
    title: str
    task: str
    segments: tuple
    scene: str = "scene.xml"
    start: str = "stand"  # stand | back | front
    note: str = ""
    checks: tuple = field(default_factory=tuple)


STAND = "velstand.onnx"


def settle(seconds=1.0, policy=STAND):
    return Segment(policy, seconds, "站稳")


def recover(seconds=1.5, policy=STAND):
    return Segment(policy, seconds, "交回默认步态")


def kick(side, label):
    name = f"ball_kick_{side}.onnx"
    return Skill(f"{label}踢球", "Mjlab-BallKick-Flat-MicroDuck",
                 (settle(), Segment(name, 0.5, f"{label}踢球", event=f"kick_{side}", main=True), recover(2.5)),
                 scene="scene_ball.xml", note="策略看不到球：观测里没有球的位置，靠操作者对准。当前源码的目标球速是 0.25 m/s，已发布权重踢得更快。", checks=("ball",))


SKILLS = {
    "walk": Skill("行走", "Mjlab-Velocity2-Rough-Backlash-MicroDuck（旧名，现为 Velocity）", (
        Segment("alpha_walking.onnx", 0.5, "零速度指令"),
        Segment("alpha_walking.onnx", 3.0, "前进 vx=0.4 m/s", twist(0.4), main=True),
        Segment("alpha_walking.onnx", 2.5, "原地左转 1.5 rad/s", twist(wz=1.5), main=True),
        Segment("alpha_walking.onnx", 1.0, "停下")),
        note="遥控台用的就是这个步态。实测速度明显低于指令，小指令（如 vx=0.15）下几乎不动。", checks=("travel", "turn")),
    "velstand": Skill("走站一体（含倒地起身）", "Mjlab-VelStand-Rough-Backlash-MicroDuck", (
        Segment("", 1.2, "仰面躺倒，无策略，电机保持默认姿势"),
        Segment(STAND, 2.0, "从仰面倒地起身", main=True),
        Segment(STAND, 3.0, "起身后前进 vx=0.4 m/s", twist(0.4), main=True),
        Segment(STAND, 1.0, "停下站稳", main=True)),
        start="back", note="真机当前的默认步态，一个网络同时负责走、站、起身。", checks=("rise", "travel")),
    "stand": Skill("站立与身体姿态", "Mjlab-StandUp-Flat-MicroDuck", (
        Segment("alpha_stand.onnx", 1.0, "站稳", main=True),
        Segment("alpha_stand.onnx", 4.0, "身体左右、前后倾斜 ±10°", body_sway, main=True),
        Segment("alpha_stand.onnx", 1.0, "回正", main=True)), checks=("tilt",)),
    "sitstand": Skill("坐下 / 起立", "Mjlab-SitStand-Flat-MicroDuck", (
        settle(), Segment("alpha_sitstand.onnx", 3.0, "坐下 (flag=1)", posture(1), main=True),
        Segment("alpha_sitstand.onnx", 3.0, "起立 (flag=0)", posture(0), main=True), recover(1.0)),
        checks=("low",)),
    "ground_pick": Skill("低头啄地", "Mjlab-GroundPick-Flat-MicroDuck", (
        settle(), Segment("alpha_ground_pick.onnx", 2.8, "啄地（相位 0→0.7）", phase(4.0), main=True), recover()),
        checks=("mouth",)),
    "kick_left": kick("left", "左脚"),
    "kick_right": kick("right", "右脚"),
    "roulade": Skill("前滚翻", "Mjlab-Roulade-Flat-MicroDuck", (
        settle(), Segment("roulade.onnx", 2.0, "前滚翻", main=True), recover(1.5)),
        note="这里按 infer_policy.py 的默认值跑 2.0 s，由策略自己滚完；真机 manifest 设为 1.0 s 后就交回默认步态。", checks=("roll",)),
    "roller": Skill("轮滑", "官方未公开（轮滑任务族）", (
        Segment("roller.onnx", 1.0, "站稳", main=True),
        Segment("roller.onnx", 5.0, "轮滑前进 vx=0.5 m/s", twist(0.5), main=True),
        Segment("roller.onnx", 1.5, "滑行停下", main=True)),
        scene="scene_rollers.xml", checks=("travel",)),
    "roller_crouch": Skill("轮滑下蹲滑行", "Mjlab-RollerCrouch-Flat-MicroDuck", (
        Segment("roller.onnx", 1.0, "站稳"),
        Segment("roller.onnx", 3.0, "加速 vx=0.5 m/s", twist(0.5)),
        Segment("roller_crouch.onnx", 3.5, "下蹲滑行（相位 0→0.7）", phase(5.0), main=True),
        Segment("roller.onnx", 1.5, "交回轮滑步态")),
        scene="scene_rollers.xml", checks=("low",)),
}


def fetch_policy(name):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(POLICY_REPO, name, revision=SHOWROOM_REVISION,
                                cache_dir=DEFAULT_POLICY.parents[3], token=False))


class Stage:
    """One duck on an official scene with BAM motors, driven by whichever policy is active."""

    def __init__(self, scene, rl=DEFAULT_RL, edit=None):
        import mujoco
        import numpy as np
        from bam.model import load_model
        from bam.mujoco import MujocoController

        self.mj, self.np = mujoco, np
        spec = mujoco.MjSpec.from_file(str(rl / SCENES / scene))
        if edit is not None:
            edit(spec)
        motor = load_model(motor_name="xl330", model="m6")
        motor.actuator.kp, motor.actuator.vin, motor.actuator.max_current = 200.0, 7.4, None
        limit = motor.actuator.vin * motor.kt.value / motor.R.value
        for act in spec.actuators:
            target = act.target.name if hasattr(act.target, "name") else str(act.target)
            act.set_to_motor()
            act.forcelimited, act.forcerange, act.ctrllimited = True, (-limit, limit), False
            act.gear = [1.0, 0, 0, 0, 0, 0]
            for joint in spec.joints:
                if joint.name == target:
                    joint.damping = np.zeros((3, 1))
                    joint.frictionloss = 0.0
                    joint.solref_friction = (-5.0e4, -2.0e2)
                    joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)
        self.model = spec.compile()
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.qpos = np.array([self.model.joint(n).qposadr[0] for n in self.names])
        self.qvel = np.array([self.model.joint(n).dofadr[0] for n in self.names])
        self.root = int(self.model.joint("trunk_base_freejoint").qposadr[0])
        self.trunk = int(self.model.body("trunk_base").id)
        self.gyro = int(self.model.sensor("imu_ang_vel").adr[0])
        # Zero sag, as in sim_body.py: keeps calibration across BAM's parameter rename.
        self.bam = MujocoController(motor, self.names, self.model, self.data, 0.0, vin_min=6.0)
        self.sessions, self.home = {}, None
        self.last_action = np.zeros(14, dtype=np.float32)

    def load(self, name):
        import onnxruntime as ort

        if name in self.sessions:
            return self.sessions[name]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(fetch_policy(name)), sess_options=options, providers=["CPUExecutionProvider"])
        meta = session.get_modelmeta().custom_metadata_map
        if meta.get("joint_names", "").split(",") != self.names or session.get_inputs()[0].shape != [1, 61]:
            raise ValueError(f"{name}: joint order or 61D observation contract does not match the model")
        home = self.np.array([float(v) for v in meta["default_joint_pos"].split(",")], dtype=self.np.float32)
        if self.home is not None and not self.np.allclose(self.home, home):
            raise ValueError(f"{name}: different default pose")
        self.home = home
        self.sessions[name] = session
        return session

    def place(self, start):
        data, height = self.data, 0.12
        quat = {"stand": [1, 0, 0, 0], "back": [math.sqrt(0.5), 0, -math.sqrt(0.5), 0],
                "front": [math.sqrt(0.5), 0, math.sqrt(0.5), 0]}[start]
        self.mj.mj_resetData(self.model, data)
        data.qpos[self.root:self.root + 7] = [0, 0, height, *quat]
        data.qpos[self.qpos] = self.home
        self.mj.mj_forward(self.model, data)
        reset_bam(self.bam, data)
        self.bam.q_target[:] = self.home

    def place_ball(self, side):
        """Same offset as training's reset_ball_in_front_of_foot, in the duck's yaw frame."""
        x, y = self.data.qpos[self.root:self.root + 2]
        yaw, lateral = self.yaw(), BALL_SIDE_M if side == "left" else -BALL_SIDE_M
        ball = self.model.joint("ball_free")
        self.data.qpos[ball.qposadr[0]:ball.qposadr[0] + 7] = [
            x + math.cos(yaw) * BALL_AHEAD_M - math.sin(yaw) * lateral,
            y + math.sin(yaw) * BALL_AHEAD_M + math.cos(yaw) * lateral, BALL_RADIUS_M, 1, 0, 0, 0]
        self.data.qvel[ball.dofadr[0]:ball.dofadr[0] + 6] = 0.0

    def yaw(self):
        w, x, y, z = self.data.qpos[self.root + 3:self.root + 7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def gravity(self):
        """World down in the trunk frame; z = -1 is perfectly upright."""
        return -self.data.xmat[self.trunk].reshape(3, 3)[2]

    def tick(self, name, command, scale):
        np, data = self.np, self.data
        if not name:
            self.bam.q_target[:] = self.home
            return self.step()
        obs = np.concatenate((data.sensordata[self.gyro:self.gyro + 3], self.gravity(), data.qpos[self.qpos] - self.home,
                              data.qvel[self.qvel], self.last_action, command)).astype(np.float32)
        if obs.shape != (61,) or not np.isfinite(obs).all():
            raise RuntimeError("Invalid policy observation")
        session = self.sessions[name]
        action = session.run(None, {session.get_inputs()[0].name: obs[None]})[0][0]
        self.last_action = action.astype(np.float32)
        self.bam.q_target[:] = self.home + scale * action
        self.step()

    def step(self):
        for _ in range(DECIMATION):
            self.bam.update()
            self.mj.mj_step(self.model, self.data)


class Camera:
    """Follows the duck (and leaves the heading alone, so turns and rolls read clearly)."""

    def __init__(self, stage, width=480, height=360):
        mj = stage.mj
        self.stage = stage
        self.renderer = mj.Renderer(stage.model, height=height, width=width)
        self.renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0  # doubles software-GL time
        self.view = mj.MjvCamera()
        self.view.type = mj.mjtCamera.mjCAMERA_FREE
        self.view.distance, self.view.elevation, self.view.azimuth = 0.85, -16, 120

    def shot(self, policy, seconds, main):
        from PIL import Image, ImageDraw, ImageFont

        x, y = self.stage.data.qpos[self.stage.root:self.stage.root + 2]
        self.view.lookat[:] = [x, y, 0.09]
        self.renderer.update_scene(self.stage.data, camera=self.view)
        picture = Image.fromarray(self.renderer.render())
        # ASCII only: Colab has no CJK font. The notebook lists the Chinese steps next to the clip.
        ImageDraw.Draw(picture).text((10, 8), f"{policy or 'no policy'}  {seconds:4.1f}s", font=ImageFont.load_default(16),
                                     fill="#fbbf24" if main else "#e2e8f0", stroke_width=2, stroke_fill="#0f172a")
        return picture


def run_skill(key, frames=True):
    """Roll one skill out; returns (pictures, metrics). Physics is deterministic."""
    import numpy as np

    skill = SKILLS[key]
    stage = Stage(skill.scene)
    for name in dict.fromkeys(segment.policy for segment in skill.segments if segment.policy):
        stage.load(name)
    stage.place(skill.start)
    kicks = [segment.event.split("_")[1] for segment in skill.segments if segment.event]
    if kicks:
        stage.place_ball(kicks[0])
    camera = Camera(stage) if frames else None
    images, log = [], []
    ball = stage.model.joint("ball_free") if "ball" in skill.checks else None
    mouth = stage.model.site("mouth_tip").id
    scale = ROLLER_ACTION_SCALE if skill.scene == "scene_rollers.xml" else 1.0
    tick = 0
    for segment in skill.segments:
        if segment.event:
            stage.place_ball(segment.event.split("_")[1])
        for step in range(round(segment.seconds / CONTROL_S)):
            stage.tick(segment.policy, np.array(segment.command(step * CONTROL_S), dtype=np.float32), scale)
            data = stage.data
            log.append({"main": segment.main, "caption": segment.caption,
                        "xyz": data.qpos[stage.root:stage.root + 3].tolist(), "yaw": stage.yaw(),
                        "gz": float(stage.gravity()[2]), "gyro_y": float(data.sensordata[stage.gyro + 1]),
                        "mouth": float(data.site_xpos[mouth][2]),
                        "ball": data.qvel[ball.dofadr[0]:ball.dofadr[0] + 3].tolist() if ball is not None else None})
            if camera is not None and tick % FRAME_TICKS == 0:
                images.append(camera.shot(segment.policy, tick * CONTROL_S, segment.main))
            tick += 1
    if camera is not None:
        camera.renderer.close()
    return images, measure(skill, log)


def measure(skill, log):
    """Plain measured numbers; upright means the trunk is within 30 degrees of vertical."""
    upright = lambda row: row["gz"] < -math.cos(math.radians(30))
    main = [row for row in log if row["main"]]
    result = {"秒": round(len(log) * CONTROL_S, 1), "结束时直立": all(map(upright, log[-25:])),
              "结束躯干高度_m": round(log[-1]["xyz"][2], 3)}
    if "travel" in skill.checks:
        moving = [row for row in main if "前进" in row["caption"]]
        seconds = len(moving) * CONTROL_S
        distance = math.dist(moving[0]["xyz"][:2], moving[-1]["xyz"][:2])
        result.update({"前进位移_m": round(distance, 2), "平均速度_m/s": round(distance / seconds, 2)})
    if "turn" in skill.checks:
        yaws = [row["yaw"] for row in main if "转" in row["caption"]]
        turned = sum((b - a + math.pi) % (2 * math.pi) - math.pi for a, b in zip(yaws, yaws[1:]))
        result["转角_deg"] = round(math.degrees(turned))
    if "rise" in skill.checks:
        first = next((i for i in range(len(main) - 25) if all(map(upright, main[i:i + 25]))), None)
        result["起身用时_s"] = None if first is None else round(first * CONTROL_S, 2)
    if "tilt" in skill.checks:
        result["最大倾斜_deg"] = round(math.degrees(max(math.acos(max(-1.0, min(1.0, -r["gz"]))) for r in main)), 1)
    if "low" in skill.checks:
        result["躯干最低_m"] = round(min(row["xyz"][2] for row in main), 3)
    if "mouth" in skill.checks:
        result["嘴尖最低_m"] = round(min(row["mouth"] for row in main), 3)
    if "ball" in skill.checks:
        start = next(i for i, row in enumerate(log) if row["main"])
        result["球最高速度_m/s"] = round(max(math.hypot(*row["ball"][:2]) for row in log[start:]), 2)
    if "roll" in skill.checks:
        start = next(i for i, row in enumerate(log) if row["main"])
        result["俯仰累计转角_deg"] = round(math.degrees(sum(row["gyro_y"] for row in log[start:]) * CONTROL_S))
    return result


def save_clip(images, path):
    """Animated WebP in real time (60 ms per picture); a GIF of the same clip is several times larger."""
    images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                   duration=round(FRAME_TICKS * CONTROL_S * 1000), quality=72, method=2)
    return path


def record(key, directory):
    images, metrics = run_skill(key)
    clip = save_clip(images, Path(directory) / f"{key}.webp")
    report = {"skill": key, "clip": str(clip), "metrics": metrics, "frames": len(images)}
    (Path(directory) / f"{key}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=[*SKILLS, "all"], default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/showroom")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for key in SKILLS if args.skill == "all" else [args.skill]:
        print(json.dumps(record(key, args.out), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
