import base64
import unittest
import zlib

try:
    import numpy as np
    import cv2  # noqa: F401 -- verifies the optional runtime dependency is importable
    import PIL  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("optional vision dependencies: run tests in .venv-vlm")

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

if __name__ == "__main__":
    unittest.main()
