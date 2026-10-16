"""Robot-mounted RGB-D virtual sensor; rendering stays on the physics/main thread.

Only the RGB image reaches the VLM. Calibrated depth/extrinsics support metric
localization; extrinsics currently use MuJoCo pose (explicit oracle localization).
"""

import base64
import os
import threading
import time
import zlib

import numpy as np


def add_robot_camera(spec):
    import mujoco

    # Camera looks down -Z, with +Y up. Body +X forward, +Y left, +Z up.
    pitch = 0.10
    c, s = np.cos(pitch), np.sin(pitch)
    rotation = np.array([[0, s, -c], [-1, 0, 0], [0, c, s]], dtype=float)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    spec.body("trunk_base").add_camera(
        name="vision_camera", pos=[0.095, 0, 0.04], quat=quat, fovy=65
    )


class RGBDCamera:
    """Same hook as official Camera, but upright calibrated robot-front RGB-D."""

    def __init__(self, model, name, width=640, height=360):
        import mujoco

        self.camera = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "vision_camera"
        )
        if self.camera < 0:
            raise RuntimeError("vision_camera missing")
        self.model = model
        self.collect_gt = os.environ.get("MICRODUCK_COLLECT_GT") == "1"
        # MSAA blends encoded geom IDs at edges and creates false distant segmentation pixels.
        if self.collect_gt:
            model.vis.quality.offsamples = 0
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.width, self.height = width, height
        self.lock = threading.Lock()
        self.latest = None
        self.latest_uyvy = None
        self.latest_gt = None

    def render(self, world):
        from mjlab_microduck.sim.camera import to_uyvy

        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera)
            metadata = {
                "sim_time": float(world.data.time),
                "wall_ns": time.time_ns(),
                "camera_position": world.data.cam_xpos[self.camera].tolist(),
                "camera_rotation": world.data.cam_xmat[self.camera]
                .reshape(3, 3)
                .tolist(),
            }
        self.renderer.disable_depth_rendering()
        ground_truth = None
        if self.collect_gt:
            # Same renderer scene as RGB/depth. Never included in packet().
            self.renderer.enable_segmentation_rendering()
            segmentation = self.renderer.render().copy()
            self.renderer.disable_segmentation_rendering()
            objects = {}
            import mujoco

            for ident in range(self.model.ngeom):
                name = self.model.geom(ident).name or ""
                if not name.startswith("goal_"):
                    continue
                mask = (segmentation[:, :, 0] == ident) & (
                    segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                )
                yy, xx = np.nonzero(mask)
                objects[name.removeprefix("goal_")] = {
                    "visible_pixels": int(mask.sum()),
                    "bbox": [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
                    if len(xx) else None,
                    "pixel": [float(xx.mean()), float(yy.mean())] if len(xx) else None,
                }
            ground_truth = {"sim_time": metadata["sim_time"], "objects": objects,
                            "source": "MuJoCo geom segmentation; collection/scoring only"}
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy().astype("<f4")
        self.renderer.disable_depth_rendering()
        focal = self.height / (
            2 * np.tan(np.deg2rad(self.model.cam_fovy[self.camera]) / 2)
        )
        metadata.update(
            width=self.width,
            height=self.height,
            fx=float(focal),
            fy=float(focal),
            cx=self.width / 2,
            cy=self.height / 2,
            rgb_zlib=base64.b64encode(zlib.compress(rgb.tobytes(), 1)).decode(),
            depth_zlib=base64.b64encode(zlib.compress(depth.tobytes(), 1)).decode(),
            sensor="robot-mounted front RGB-D; MuJoCo camera extrinsics",
        )
        with self.lock:
            self.latest = metadata
            self.latest_uyvy = to_uyvy(rgb)
            self.latest_gt = ground_truth

    def collection_packet(self):
        with self.lock:
            if self.latest is None or self.latest_gt is None:
                raise RuntimeError("collection ground truth unavailable")
            return {"sensor": dict(self.latest), "ground_truth": self.latest_gt}

    def packet(self):
        with self.lock:
            if self.latest is None:
                raise RuntimeError("camera frame not ready")
            return dict(self.latest)

    def frame(self):
        with self.lock:
            return self.latest_uyvy


def run_rgbd(world, headless):
    """Keep 50 Hz physics independent of slow OpenGL/depth/image work.

    Official run() renders on the physics loop. A RGB+depth frame can pause that
    loop for several control ticks, invalidating robotd's wall-clock policy inputs.
    OpenGL stays on the main thread, while physics has its own real-time worker.
    """
    import mujoco.viewer

    stopped = threading.Event()
    errors = []
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(
            world.model, world.data, show_left_ui=False, show_right_ui=False
        )

    def physics():
        deadline = time.perf_counter()
        try:
            while not stopped.is_set():
                world.step(4)
                deadline += 0.02
                slack = deadline - time.perf_counter()
                if slack > 0:
                    stopped.wait(slack)
                elif slack < -0.25:
                    deadline = time.perf_counter()
        except BaseException as error:
            errors.append(error)
            stopped.set()

    worker = threading.Thread(target=physics, name="vision-physics", daemon=True)
    worker.start()
    next_image = 0.0
    try:
        while not stopped.is_set():
            now = time.perf_counter()
            if now >= next_image:
                for body in world.bodies:
                    if body.camera is not None:
                        body.camera.render(world)
                next_image = time.perf_counter() + 0.2
            if viewer is not None:
                if not viewer.is_running():
                    break
                with world.lock:
                    viewer.sync()
            stopped.wait(0.02)
        if errors:
            raise RuntimeError("physics worker failed") from errors[0]
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        worker.join(timeout=2)
        if viewer is not None:
            viewer.close()
