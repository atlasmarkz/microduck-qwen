"""Headless MuJoCo rollouts of the gait policy, matching microduck_qwen.sim_native.Gait.update.

One World per process, reset between episodes. Everything that varies between episodes is drawn from
``numpy.random.default_rng(seed)`` in a fixed order, so the same seed gives the same command,
friction, heading and observation-noise sequence for every policy: base and candidate are paired.

Protocol (PROTOCOL.md section 2): 1 s zero command, then the episode command for 6 s; the measured
window is seconds 2-6 after the command starts. Fall: trunk z < 0.085 m or gravity z > -0.65.
"""

import math
import os
import sys

import numpy as np

from . import MICRODUCK_ROOT

CONTROL_DT = 0.02
SETTLE_S = 1.0
COMMAND_S = 6.0
WINDOW_S = (2.0, 6.0)
# Training observation noise (microduck_velocity_env_cfg.py): uniform, per element.
OBS_NOISE = np.concatenate([np.full(3, 0.03), np.full(3, 0.01), np.full(14, 0.001), np.full(14, 0.25),
                            np.zeros(14 + 13)]).astype(np.float32)
FRICTION_RANGE = (0.7, 1.3)

_SIM = None


class Sim:
    """The MicroDuck world with BAM motors on an empty floor (no arena objects or walls)."""

    def __init__(self):
        import mujoco

        sys.path.insert(0, str(MICRODUCK_ROOT / "runs/upstream/microduck_rl-develop/src"))
        sys.path.insert(0, str(MICRODUCK_ROOT))
        os.environ.pop("MICRODUCK_VISION_SCENE", None)
        os.environ.pop("MICRODUCK_COLLECT_GT", None)
        from microduck_qwen import sim_body

        # Empty floor: a 6 s walk covers up to 2.4 m, which would reach the arena's ball and obstacle.
        sim_body.arena = lambda: {"objects": {}, "obstacles": [], "walls": []}
        self.mujoco = mujoco
        self.server = sim_body.configure_body()
        self.reset_bam = sim_body.reset_bam
        self.world = self.server.World(self.server.DEFAULT_SCENE)
        self.body = self.server.Body(self.world, 0)
        self.world.bodies.append(self.body)
        model = self.world.model
        self.home = np.array([self.server.HOME_POSE[i] for i in self.body.to_wire], dtype=np.float32)
        self.gyro = model.sensor("imu_ang_vel").adr[0]
        # MuJoCo combines contact friction with max(), so scale the floor and both feet together.
        self.friction_geoms = [i for i in range(model.ngeom)
                               if model.geom(i).name in ("floor", "left_foot_collision", "right_foot_collision")]
        if len(self.friction_geoms) != 3:
            raise RuntimeError("floor or foot geoms not found")
        self.base_friction = model.geom_friction[self.friction_geoms, 0].copy()

    def reset(self, yaw, friction_scale):
        mujoco, world, body = self.mujoco, self.world, self.body
        mujoco.mj_resetData(world.model, world.data)
        world.model.geom_friction[self.friction_geoms, 0] = self.base_friction * friction_scale
        body.place(None, self.server.HOME_TRUNK_Z, 0.0)
        world.data.qpos[body.trunk + 3:body.trunk + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(world.model, world.data)
        body.remember()
        body.set_torque(True)
        self.reset_bam(world.bam, world.data)

    def trunk_state(self):
        data, trunk, dof = self.world.data, self.body.trunk, self.body.trunk_dof
        quat = data.qpos[trunk + 3:trunk + 7].copy()
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(data.qpos[trunk + 2]), quat, yaw, data.qvel[dof:dof + 3].copy()


def get_sim():
    global _SIM
    if _SIM is None:
        _SIM = Sim()
    return _SIM


LEAD_WZ = 1.5
LEAD_S = 1.5
OLD_KINDS = ("stand", "walk", "spin")


def episode_spec(seed, condition):
    """Command, friction and heading for one seed; the rng continues into the noise stream.

    new (PROTOCOL 2, revised 2026-11-21): turning already under way - LEAD_S seconds at
    sign * LEAD_WZ in place, then the target sign * U(0.3, 1.0) in place.
    old: one third each, from standstill - stand (zero command), walk (vx U(0.3, 0.4)),
    spin (in place, sign * U(1.5, 2.0)).
    """
    rng = np.random.default_rng(seed)
    lead, kind = None, None
    if condition == "new":
        sign = 1.0 if rng.random() < 0.5 else -1.0
        command = (0.0, 0.0, float(sign * rng.uniform(0.3, 1.0)))
        lead = ((0.0, 0.0, sign * LEAD_WZ), LEAD_S)
    elif condition == "old":
        kind = OLD_KINDS[int(rng.integers(3))]
        if kind == "stand":
            command = (0.0, 0.0, 0.0)
        elif kind == "walk":
            command = (float(rng.uniform(0.3, 0.4)), 0.0, 0.0)
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            command = (0.0, 0.0, float(sign * rng.uniform(1.5, 2.0)))
    else:
        raise ValueError(condition)
    friction = float(rng.uniform(*FRICTION_RANGE))
    yaw0 = float(rng.uniform(0.0, 2 * math.pi))
    return rng, {"seed": int(seed), "condition": condition, "kind": kind, "command": command,
                 "lead": lead, "friction_scale": friction, "yaw0": yaw0}


def rollout(policy, seed, condition, command=None, record=False, lead="spec", noise=True):
    """Run one episode. ``policy(obs[61]) -> action[14]``.

    Timeline: SETTLE_S zero command, then the lead-in (if any), then the target command for
    COMMAND_S; the measured window is WINDOW_S after the target starts. Probes and diagnostics only:
    ``command`` overrides the target, ``lead`` = (command, seconds) or None overrides the lead-in,
    ``noise=False`` drops obs noise. Everything else still comes from the seed.
    """
    sim = get_sim()
    rng, spec = episode_spec(seed, condition)
    if command is not None:
        spec["command"] = tuple(float(c) for c in command)
    if lead != "spec":
        spec["lead"] = lead
    sim.reset(spec["yaw0"], spec["friction_scale"])
    world, body, data = sim.world, sim.body, sim.world.data
    last_action = np.zeros(14, dtype=np.float32)
    settle_ticks = round(SETTLE_S / CONTROL_DT)
    lead_ticks = round(spec["lead"][1] / CONTROL_DT) if spec["lead"] else 0
    target_tick = settle_ticks + lead_ticks
    total_ticks = target_tick + round(COMMAND_S / CONTROL_DT)
    w0, w1 = (target_tick + round(t / CONTROL_DT) for t in WINDOW_S)
    yaws, vel_body, tilts, frames = [], [], [], []
    fell_at = None
    for tick in range(total_ticks):
        z, quat, yaw, lin_vel = sim.trunk_state()
        gravity = sim.server.gravity_in_trunk(quat)
        if z < 0.085 or gravity[2] > -0.65:
            fell_at = tick * CONTROL_DT
            break
        command = np.zeros(13, dtype=np.float32)
        if tick >= target_tick:
            command[:3] = spec["command"]
        elif tick >= settle_ticks:
            command[:3] = spec["lead"][0]
        clean = np.concatenate((data.sensordata[sim.gyro:sim.gyro + 3], gravity,
                                data.qpos[body.qpos_adr] - sim.home, data.qvel[body.qvel_adr],
                                last_action, command)).astype(np.float32)
        drawn = rng.uniform(-1.0, 1.0, 61).astype(np.float32) * OBS_NOISE
        obs = clean + drawn if noise else clean
        action = np.asarray(policy(obs), dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise RuntimeError("invalid policy action")
        if record:
            frames.append((obs, action))
        last_action = action
        world.bam.q_target[:] = sim.home + action
        world.step(4)
        if w0 <= tick + 1 <= w1:
            _, quat_after, yaw_after, vel = sim.trunk_state()
            yaws.append(yaw_after)
            c, s = math.cos(yaw_after), math.sin(yaw_after)
            vel_body.append((c * vel[0] + s * vel[1], -s * vel[0] + c * vel[1]))
            tilts.append(math.degrees(math.acos(min(1.0, -float(sim.server.gravity_in_trunk(quat_after)[2])))))
    result = dict(spec, command=list(spec["command"]), fell=fell_at is not None, fell_at_s=fell_at)
    if spec["lead"]:
        result["lead"] = [list(spec["lead"][0]), spec["lead"][1]]
    if fell_at is None:
        unwrapped = np.unwrap(np.array(yaws))
        span = (len(yaws) - 1) * CONTROL_DT
        vb = np.array(vel_body)
        result.update(yaw_rate=float((unwrapped[-1] - unwrapped[0]) / span), vx=float(vb[:, 0].mean()),
                      vy=float(vb[:, 1].mean()), max_tilt_deg=float(max(tilts)))
    if record:
        result["frames"] = frames
    return result


def new_success(result, ratio=0.7):
    """PROTOCOL.md section 2: no fall, same direction, |yaw rate| >= 70 % of |command|."""
    if result["fell"]:
        return False
    wz = result["command"][2]
    return result["yaw_rate"] * math.copysign(1.0, wz) >= ratio * abs(wz)
