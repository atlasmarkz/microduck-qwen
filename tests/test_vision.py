import base64
import json
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
    validate_answer,
    Frame,
    visibility_check,
    semantic_guard,
    parse_visual_choice,
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

    def test_vlm_cannot_execute_nonexistent_or_nonvisible_target(self):
        base = {
            "status": "ok",
            "target_id": 1,
            "visibility": "visible",
            "reason": "test",
        }
        self.assertEqual(validate_answer(json.dumps(base), [{"id": 1}]), base)
        with self.assertRaises(ValueError):
            validate_answer(json.dumps(base), [])
        with self.assertRaises(ValueError):
            validate_answer(json.dumps(dict(base, visibility="partial")), [{"id": 1}])
        with self.assertRaises(ValueError):
            validate_answer(json.dumps(dict(base, target_id=True)), [{"id": 1}])

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

    def test_spatial_guard_rejects_without_replacing_wrong_vlm_choice(self):
        candidates = [
            {"id": 8, "color": "blue", "pixel": [10.0, 20.0]},
            {"id": 1, "color": "green", "pixel": [60.0, 20.0]},
        ]
        result = semantic_guard("走到画面左边的球旁边", candidates[1], candidates)
        self.assertFalse(result["safe"])
        self.assertEqual(candidates[1]["id"], 1)
        self.assertTrue(
            semantic_guard("走到画面左边的球旁边", candidates[0], candidates)["safe"]
        )

    def test_visual_color_label_binding_and_duplicates_fail_closed(self):
        candidates = [{"id": 2, "color": "blue"}]
        answer, _ = parse_visual_choice(
            '{"status":"visible","color":"blue","id":2}', candidates
        )
        self.assertEqual(answer["target_id"], 2)
        with self.assertRaises(ValueError):
            parse_visual_choice('{"status":"visible","color":"red","id":2}', candidates)
        with self.assertRaises(ValueError):
            parse_visual_choice(
                '{"status":"visible","color":"blue","id":1,"id":2}', candidates
            )

    def test_unknown_color_and_hidden_reference_do_not_trigger_guessing(self):
        candidates = [{"id": 2, "color": "red", "pixel": [10.0, 20.0]}]
        self.assertFalse(
            semantic_guard("走到紫色球旁边", candidates[0], candidates)["safe"]
        )
        self.assertFalse(
            semantic_guard("走到挡住的球旁边", candidates[0], candidates)["safe"]
        )


if __name__ == "__main__":
    unittest.main()
