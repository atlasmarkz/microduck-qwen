"""Extend the official body server with an arena and training-matched BAM M6 motors.

Motor conversion mirrors microduck_rl/scripts/infer_policy.py; upstream files are untouched.
"""

from .contract import arena
import json
import os
import time
from pathlib import Path


def reset_bam(controller, data):
    if hasattr(controller, "reset"):
        controller.reset(data.qpos)
    else:
        controller.model.reset()
        controller.last_ts = data.time
        controller.q_target[:] = data.qpos[controller.qpos_indexes]


def configure_body():
    import mujoco
    import numpy as np
    from bam.model import load_model
    from bam.mujoco import MujocoController
    from mjlab_microduck.sim import body_server

    scene_path = os.environ.get("MICRODUCK_VISION_SCENE")
    scene = json.loads(Path(scene_path).read_text()) if scene_path else arena()
    if scene_path:
        from .vision_sensor import RGBDCamera, add_robot_camera, run_rgbd

        body_server.Camera = RGBDCamera
        body_server.CAMERA_FPS = 5
        body_server.run = run_rgbd
        original_dispatch = body_server.Handler.dispatch

        def dispatch(handler, body, request):
            if request.get("op") in ("collection_frame", "collection_stats"):
                if os.environ.get("MICRODUCK_COLLECT_GT") != "1":
                    raise RuntimeError("ground truth endpoint disabled outside collection")
                if request["op"] == "collection_frame":
                    return body.camera.collection_packet()
                with body.world.lock:
                    return dict(body.world.collection_stats)
            if request.get("op") == "vision":
                if body.camera is None:
                    raise RuntimeError("camera disabled")
                return body.camera.packet()
            if request.get("op") == "vision_face_forward":
                # Simulation setup only: stabilize a reproducible initial view after
                # sit-to-stand. No target or scene labels enter camera observations.
                with body.world.lock:
                    data = body.world.data
                    q = data.qpos[body.trunk + 3 : body.trunk + 7].copy()
                    w, x, y, z = q
                    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    inv_yaw = np.array([np.cos(yaw / 2), 0, 0, -np.sin(yaw / 2)])
                    mujoco.mju_mulQuat(
                        data.qpos[body.trunk + 3 : body.trunk + 7], inv_yaw, q
                    )
                    data.qvel[body.trunk_dof : body.trunk_dof + 6] = 0.0
                    mujoco.mj_forward(body.world.model, data)
                return {"accepted": True}
            return original_dispatch(handler, body, request)

        body_server.Handler.dispatch = dispatch

    def build_world(path, count):
        if count != 1:
            raise ValueError("导航实验目前只支持一只 MicroDuck")
        spec = mujoco.MjSpec.from_file(str(path))
        for name, obj in scene["objects"].items():
            ball = obj.get("shape") == "sphere" or name == "red_ball"
            spec.worldbody.add_geom(
                name="goal_" + name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE
                if ball
                else mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[obj["x"], obj["y"], obj["radius"] if ball else 0.003],
                size=[obj["radius"], 0.003, 0.0],
                rgba=obj.get(
                    "rgba", [0.9, 0.12, 0.08, 1] if ball else [0.1, 0.3, 0.95, 1]
                ),
                contype=1 if obj["collidable"] else 0,
                conaffinity=1 if obj["collidable"] else 0,
            )
        for obj in scene["obstacles"]:
            spec.worldbody.add_geom(
                name=obj["name"],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[obj["x"], obj["y"], obj["height"] / 2],
                size=[obj["radius"], obj["height"] / 2, 0],
                rgba=obj.get("rgba", [0.9, 0.65, 0.12, 1]),
                contype=1,
                conaffinity=1,
            )
        if scene_path:
            add_robot_camera(spec)
        motor = load_model(motor_name="xl330", model="m6")
        motor.actuator.vin = 7.4
        force_limit = motor.actuator.vin * motor.kt.value / motor.R.value
        for act in spec.actuators:
            target = act.target.name if hasattr(act.target, "name") else str(act.target)
            if target.startswith("passive_"):
                continue
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-force_limit, force_limit)
            act.ctrllimited = False
            act.gear = [1.0, 0, 0, 0, 0, 0]
            for joint in spec.joints:
                if joint.name == target:
                    joint.damping = np.zeros((3, 1))
                    joint.frictionloss = 0.0
                    joint.solref_friction = (-5.0e4, -2.0e2)
                    joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)
                    break
        return spec.compile()

    class BamWorld(body_server.World):
        def __init__(self, path, count=1):
            super().__init__(path, count)
            motor = load_model(motor_name="xl330", model="m6")
            motor.actuator.kp = 200.0
            motor.actuator.vin = 7.4
            motor.actuator.max_current = None
            names = [self.model.actuator(i).name for i in range(self.model.nu)]
            # Zero sag preserves calibration across BAM's gain/resistance API rename.
            self.bam = MujocoController(
                motor, names, self.model, self.data, 0.0, vin_min=6.0
            )
            self.collection_stats = {"steps_with_scene_contact": 0, "minimum_contact_distance_m": None}
            self.scene_geoms = {i for i in range(self.model.ngeom)
                                if (self.model.geom(i).name or "").startswith("goal_")
                                or (self.model.geom(i).name or "") in {o["name"] for o in scene["obstacles"]}}

        def step(self, times=1):
            with self.lock:
                for _ in range(times):
                    if self.bodies and self.bodies[0].torque_on:
                        self.bam.update()
                    else:
                        self.data.ctrl[:] = 0.0
                    mujoco.mj_step(self.model, self.data)
                    if os.environ.get("MICRODUCK_COLLECT_GT") == "1":
                        contacts = [float(contact.dist) for contact in self.data.contact
                                    if ((contact.geom1 in self.scene_geoms and self.model.geom_bodyid[contact.geom2] > 0)
                                        or (contact.geom2 in self.scene_geoms and self.model.geom_bodyid[contact.geom1] > 0))
                                    and contact.dist <= 0]
                        if contacts:
                            self.collection_stats["steps_with_scene_contact"] += 1
                            previous = self.collection_stats["minimum_contact_distance_m"]
                            self.collection_stats["minimum_contact_distance_m"] = min(contacts + ([previous] if previous is not None else []))
                    for body in self.bodies:
                        if not body.released or time.monotonic() < getattr(
                            body, "hold_until", 0.0
                        ):
                            body.restore()

    class BamBody(body_server.Body):
        def place(self, pose, trunk_z, offset_y):
            super().place(pose, trunk_z, offset_y)
            reset_bam(self.world.bam, self.world.data)
            if list(self.actuators) != self.world.bam.act_indexes:
                raise RuntimeError("BAM 与 body 的电机索引不一致")

        def set_targets(self, wire_targets):
            if len(wire_targets) != len(body_server.JOINT_NAMES):
                raise ValueError("关节目标数量错误")
            with self.world.lock:
                self.world.bam.q_target[:] = [wire_targets[i] for i in self.to_wire]

        def set_torque(self, on):
            with self.world.lock:
                self.torque_on = bool(on)
                if on:
                    if scene_path and not self.released:
                        # Vision starts at HOME. Hold through robotd's 2 s home
                        # ramp, so renderer scheduling cannot topple a biped
                        # before its balancing network actually takes control.
                        self.hold_until = time.monotonic() + 2.5
                    self.released = True
                    reset_bam(self.world.bam, self.world.data)
                self._apply_torque()

        def _apply_torque(self):
            self.world.bam.model.actuator.kp = self.kp if self.torque_on else 0.0

    body_server.build_world = build_world
    body_server.World = BamWorld
    body_server.Body = BamBody
    return body_server


def main():
    configure_body().main()


if __name__ == "__main__":
    main()
