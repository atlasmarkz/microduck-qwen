import math
from types import SimpleNamespace
import unittest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from microduck_qwen.agent_data import tasks
from microduck_qwen.bypass import plan_bypass, run_bypass, verify_passes, yellow_detections
from microduck_qwen.navigation import clearance
from microduck_qwen.skills import NeedsUserInput, validate_action
from microduck_qwen.vision_scene import make_scene


class BypassTests(unittest.TestCase):
    def test_missing_visual_confirmation_refuses_before_motion(self):
        executor = SimpleNamespace(scene=make_scene(0, yellow_course=True), info={"port": 1},
            healthy=lambda: {"trunk": [0, 0, 0.2], "yaw": 0})
        with patch("microduck_qwen.vision.capture"), patch("microduck_qwen.bypass.yellow_detections", return_value=[]), patch("microduck_qwen.bypass.execute") as navigation:
            with self.assertRaises(NeedsUserInput):
                run_bypass(executor, {"action": "bypass_return", "side": "auto"}, Path("unused.json"))
            navigation.assert_not_called()

    def test_cancel_between_legs_retains_task_start_timestamp(self):
        from microduck_qwen.navigation import execute
        robot, observer = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cancel = directory / "cancel"
            cancel.write_text("100")
            with patch("microduck_qwen.navigation.time.time_ns", return_value=200):
                result = execute({"status": "ok", "goals": [{"target": "point", "x": 0.2, "y": 0,
                    "speed": "slow", "stop_distance_m": 0.1}]}, robot, observer, directory / "trace.jsonl",
                    scene={"bounds": [-1, 1, -1, 1], "objects": {}, "obstacles": []},
                    cancel_file=cancel, cancel_started_ns=50)
            self.assertTrue(result["cancelled"])
            observer.read.assert_not_called()
            robot.move.assert_not_called()
            robot.stop.assert_called()

    def test_plans_have_frozen_distance_order_and_safe_waypoints(self):
        start = {"x": 0, "y": 0, "yaw": 0}
        for seed in range(4):
            scene = make_scene(seed, yellow_course=True)
            for side in ("auto", "left", "right"):
                legs = plan_bypass(scene, start, side)
                self.assertEqual(len(legs), 2 + seed % 2)
                self.assertEqual([leg["obstacle"]["name"] for leg in legs], [f"yellow_{i}" for i in range(len(legs))])
                self.assertTrue(all(clearance(p, scene) >= 0.04 for leg in legs for p in leg["points"]))
                if side != "auto":
                    self.assertTrue(all(leg["side"] == side for leg in legs))

    def test_measured_trace_requires_requested_side_order_and_real_crossing(self):
        start = {"x": 0, "y": 0, "yaw": 0}
        scene = make_scene(0, yellow_course=True)
        legs = plan_bypass(scene, start, "left")
        positions = [[0, 0]]
        for leg in legs:
            for target in leg["points"]:
                origin = positions[-1]
                steps = max(1, math.ceil(math.dist(origin, target) / 0.02))
                positions.extend([[a + (b - a) * i / steps for a, b in zip(origin, target)] for i in range(1, steps + 1)])
        self.assertTrue(verify_passes(positions, legs, start)["success"])
        self.assertFalse(verify_passes(positions, list(reversed(legs)), start)["success"])
        self.assertFalse(verify_passes([[0, 0]] * 5, legs, start)["success"])
        wrong = [dict(leg, side="right") for leg in legs]
        self.assertFalse(verify_passes(positions, wrong, start)["success"])
        self.assertFalse(verify_passes(positions[::20], legs, start)["success"])

    def test_yellow_confirmation_uses_pixels_and_depth(self):
        frame = SimpleNamespace(rgb=np.full((64, 64, 3), [230, 166, 15], dtype="uint8"),
            depth=np.ones((64, 64), dtype="float32"), metadata={"fx": 64, "fy": 64, "cx": 32, "cy": 32,
                "camera_rotation": np.eye(3).tolist(), "camera_position": [0, 0, 1]})
        self.assertEqual(len(yellow_detections(frame)), 1)
        frame.rgb[:] = [0, 0, 255]
        self.assertFalse(yellow_detections(frame))
        frame.rgb[:] = [230, 166, 15]
        frame.depth[:] = float("nan")
        self.assertFalse(yellow_detections(frame))

    def test_bypass_contract_and_teacher_tasks_preserve_return_requirement(self):
        for side in ("auto", "left", "right"):
            self.assertEqual(validate_action({"action": "bypass_return", "side": side})["side"], side)
        for bad in ({"action": "bypass_return", "side": "behind"}, {"action": "bypass_return", "side": "auto", "return": False}):
            with self.assertRaises(ValueError):
                validate_action(bad)
        self.assertEqual(tasks("train")[9][1], [{"action": "bypass_return", "side": "auto"}])
        for variant in range(3):
            self.assertFalse({t[0] for t in tasks("train", variant)} & {t[0] for t in tasks("test")})
        with self.assertRaises(ValueError):
            make_scene(0, "full", yellow_course=True)
