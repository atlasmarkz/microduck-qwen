import base64
import math
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import cv2  # noqa: F401 -- verifies the optional runtime dependency is importable
    import PIL  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("optional vision dependencies: run tests in .venv-vlm")

from microduck_qwen.navigation import face_target_heading, pathfind, segment_clear
from microduck_qwen.vision import (
    apply_grace,
    approachable,
    decode_array,
    project,
    unproject,
    Frame,
    OUT_OF_VIEW_GRACE,
    visibility_check,
    VisualGuard,
    propose_targets,
)
from microduck_qwen.vision_scene import make_scene


class VisionTests(unittest.TestCase):
    def test_yellow_pillar_gets_a_numbered_localized_target_not_a_ball(self):
        metadata = self.metadata()
        metadata.update(camera_position=[0, 0, 0.15], camera_rotation=[[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        frame = Frame(np.zeros((80, 100, 3), np.uint8), np.ones((80, 100), np.float32), metadata)
        frame.rgb[25:55, 40:60] = [230, 166, 15]
        candidates = propose_targets(frame)
        self.assertEqual(len(candidates), 1)
        target = candidates[0]
        self.assertEqual((target["color"], target["kind"], target["id"]), ("yellow", "pillar", 1))
        self.assertTrue(approachable(target))
        self.assertAlmostEqual(target["world_position"][0], 1.1)
        self.assertTrue(visibility_check(frame, np.asarray(target["world_position"]), "yellow")["safe"])
        frame.depth[:] = float("nan")
        self.assertEqual(propose_targets(frame), [])

    def metadata(self):
        return {
            "width": 100,
            "height": 80,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "camera_position": [0.0, 0.0, 0.0],
            "camera_rotation": np.eye(3).tolist(),
        }

    def test_calibrated_projection_round_trip(self):
        meta = self.metadata()
        point = unproject(20.0, 30.0, 1.4, meta)
        np.testing.assert_allclose(project(point, meta), [20.0, 30.0, 1.4])

    def test_camera_decoder_rejects_wrong_payload_size(self):
        encoded = base64.b64encode(zlib.compress(b"abcd")).decode()
        with self.assertRaises(ValueError):
            decode_array(encoded, np.uint8, (3,))
        np.testing.assert_array_equal(
            decode_array(encoded, np.uint8, (4,)), [97, 98, 99, 100]
        )

    def test_random_layout_is_reproducible_but_not_fixed(self):
        self.assertEqual(make_scene(42), make_scene(42))
        self.assertNotEqual(make_scene(42)["objects"], make_scene(43)["objects"])

    def test_fully_absent_target_not_secretly_added(self):
        self.assertNotIn("ball_0", make_scene(42, "absent")["objects"])

    def test_previously_seen_target_with_foreground_depth_is_occluded(self):
        frame = Frame(
            np.zeros((80, 100, 3), np.uint8),
            np.full((80, 100), 0.5, np.float32),
            self.metadata(),
        )
        result = visibility_check(frame, np.array([0.0, 0.0, -1.0]), "red")
        self.assertFalse(result["safe"])
        self.assertEqual(result["status"], "occluded")

    def test_missing_target_is_not_always_called_occlusion(self):
        frame = Frame(
            np.zeros((80, 100, 3), np.uint8),
            np.full((80, 100), 2.0, np.float32),
            self.metadata(),
        )
        self.assertEqual(
            visibility_check(frame, np.array([0.0, 0.0, -1.0]), "red")["status"],
            "missing",
        )
        self.assertEqual(
            visibility_check(frame, np.array([0.0, 0.0, 1.0]), "red")["status"],
            "out_of_view",
        )

class ApproachGuardTests(unittest.TestCase):
    """Partial occlusion, transient loss of sight and facing the bound object."""

    def scene(self, obstacles=()):
        return {"bounds": (-1.0, 2.0, -1.0, 1.0), "obstacles": list(obstacles), "objects": {}}

    def candidate(self, visibility, fraction):
        return {"visibility": visibility, "visible_fraction_estimate": fraction}

    def test_partial_target_is_approachable_above_the_floor(self):
        self.assertTrue(approachable(self.candidate("visible", 1.0)))
        self.assertTrue(approachable(self.candidate("partial", 0.35)))
        self.assertFalse(approachable(self.candidate("partial", 0.34)))

    def test_out_of_frame_reads_are_tolerated_then_refused(self):
        verdicts, lost = [], 0
        for _ in range(OUT_OF_VIEW_GRACE + 2):
            check, lost = apply_grace({"safe": False, "status": "out_of_view"}, lost)
            verdicts.append(check["safe"])
        self.assertEqual(verdicts, [True] * OUT_OF_VIEW_GRACE + [False, False])

    def test_occlusion_and_missing_never_get_the_grace(self):
        for status in ("occluded", "missing", "partial"):
            check, lost = apply_grace({"safe": False, "status": status}, 2)
            self.assertFalse(check["safe"], status)
            self.assertEqual(lost, 0, status)

    def test_a_visible_read_resets_the_tolerated_run(self):
        _, lost = apply_grace({"safe": False, "status": "out_of_view"}, 0)
        check, lost = apply_grace({"safe": True, "status": "visible"}, lost)
        self.assertEqual(lost, 0)

    def test_suspend_clears_the_cached_verdict_and_the_loss_run(self):
        guard = VisualGuard(1, np.array([0.0, 0.0, 0.0]), "red", Path(tempfile.mkdtemp()))
        calls = []

        def fake_check(frame, target, color):
            calls.append(1)
            return {"safe": False, "status": "out_of_view"}

        with patch("microduck_qwen.vision.capture", return_value=object()), patch(
            "microduck_qwen.vision.visibility_check", side_effect=fake_check
        ), patch("microduck_qwen.vision.propose_targets", return_value=[]), patch(
            "microduck_qwen.vision.save_frame", return_value=None
        ):
            first = guard(None, None)
            guard(None, None)  # inside the 0.6s throttle: served from cache
            guard.suspend()
            self.assertEqual(guard.last["status"], "suspended")
            third = guard(None, None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(first["status"], "out_of_view_tolerated")
        self.assertEqual(third["status"], "out_of_view_tolerated")

    def test_face_target_heading_requires_a_clear_line(self):
        pillar = {"name": "pillar", "x": 0.5, "y": 0.0, "radius": 0.065}
        self.assertAlmostEqual(
            face_target_heading((0.0, 0.0), (1.0, 0.0), 0.3, self.scene()), 0.0
        )
        self.assertIsNone(
            face_target_heading((0.0, 0.0), (1.0, 0.0), 0.3, self.scene([pillar]))
        )

    def test_face_target_heading_aims_inside_the_stopping_region(self):
        self.assertAlmostEqual(
            face_target_heading((0.0, 0.0), (0.2, 0.0), 0.3, self.scene()), 0.0
        )

    def test_default_segment_margin_is_still_the_path_margin(self):
        beside = {"name": "pillar", "x": 0.5, "y": 0.25, "radius": 0.065}
        self.assertFalse(segment_clear((0.0, 0.0), (1.0, 0.0), self.scene([beside])))
        self.assertTrue(
            segment_clear((0.0, 0.0), (1.0, 0.0), self.scene([beside]), margin=0.0)
        )


class SceneClutterTests(unittest.TestCase):
    """The harder opt-in scene must stay physically consistent and navigable."""

    def items(self, scene):
        return [(name, obj) for name, obj in scene["objects"].items()] + [
            (obj["name"], obj) for obj in scene["obstacles"]
        ]

    def test_default_scene_is_unchanged_by_the_clutter_option(self):
        scene = make_scene(42)
        self.assertFalse(scene["clutter"])
        self.assertEqual(len(scene["objects"]), 3)
        self.assertEqual(len(scene["obstacles"]), 1)

    def test_clutter_adds_duplicate_colors_without_overlapping_bodies(self):
        scene = make_scene(7, "full", True)
        self.assertTrue(scene["clutter"])
        self.assertEqual(len(scene["objects"]), 5)
        colors = [obj["color"] for obj in scene["objects"].values()]
        self.assertGreater(len(colors), len(set(colors)))
        items = self.items(scene)
        for index, (name_a, a) in enumerate(items):
            for name_b, b in items[index + 1:]:
                gap = math.dist((a["x"], a["y"]), (b["x"], b["y"])) - a["radius"] - b["radius"]
                self.assertGreater(gap, 0.0, f"{name_a} overlaps {name_b}")

    def test_clutter_keeps_every_visible_target_reachable(self):
        scene = make_scene(7, "full", True)
        for name in ("ball_1", "ball_2", "ball_3", "ball_4"):
            obj = scene["objects"][name]
            self.assertTrue(
                pathfind((0.0, 0.0), (obj["x"], obj["y"]), 0.30, scene), name
            )

    def test_fully_occluded_target_has_no_reachable_stop_region(self):
        # Documents the conservative design: the ball behind the occluder cannot be
        # physically approached at the 0.30 m stopping distance either.
        scene = make_scene(7, "full", True)
        obj = scene["objects"]["ball_0"]
        with self.assertRaises(ValueError):
            pathfind((0.0, 0.0), (obj["x"], obj["y"]), 0.30, scene)


if __name__ == "__main__":
    unittest.main()
