import base64
import math
import unittest
import zlib

try:
    import numpy as np
    import cv2  # noqa: F401 -- verifies the optional runtime dependency is importable
    import PIL  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("optional vision dependencies: run tests in .venv-vlm")

from microduck_qwen.navigation import pathfind
from microduck_qwen.vision import (
    decode_array,
    project,
    unproject,
    Frame,
    visibility_check,
)
from microduck_qwen.vision_scene import make_scene


class VisionTests(unittest.TestCase):
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
