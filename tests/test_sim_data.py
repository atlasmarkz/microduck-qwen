import threading
import unittest

try:
    import numpy as np
    import PIL  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("optional vision dependencies required")

from microduck_qwen.sim_data import bbox_iou, verify_objects
from microduck_qwen.vision import Frame
from microduck_qwen.vision_sensor import RGBDCamera


class SimulationDataTests(unittest.TestCase):
    def inputs(self):
        scene = {"objects": {"ball_0": {"color": "red", "x": 0.0, "y": 0.0, "radius": 0.07}}}
        frame = Frame(np.zeros((80, 100, 3), np.uint8), np.ones((80, 100)), {
            "width": 100, "height": 80, "fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0,
            "camera_position": [0, 0, 1.07], "camera_rotation": np.eye(3).tolist()})
        gt = {"objects": {"ball_0": {"visible_pixels": 154, "pixel": [49.5, 39.5], "bbox": [43, 33, 57, 47]}}}
        candidates = [{"color": "red", "id": 9, "bbox": [43, 33, 57, 47]}]
        return scene, frame, gt, candidates

    def test_visible_target_binds_to_the_single_matching_candidate(self):
        verified = verify_objects(*self.inputs())
        red = next(obj for obj in verified if obj["color"] == "red")
        self.assertEqual(red["status"], "visible")
        self.assertEqual(red["candidate"]["id"], 9)

    def test_detector_failure_is_discarded_not_called_absence(self):
        scene, frame, gt, _ = self.inputs()
        verified = verify_objects(scene, frame, gt, [])
        self.assertEqual(verified[0]["status"], "discard")

    def test_partial_target_has_no_candidate_binding(self):
        scene, frame, gt, candidates = self.inputs()
        gt["objects"]["ball_0"]["visible_pixels"] = 50
        verified = verify_objects(scene, frame, gt, candidates)
        self.assertEqual(verified[0]["status"], "partial")
        self.assertIsNone(verified[0]["candidate"])

    def test_iou_does_not_match_separate_objects(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_default_camera_packet_never_exposes_collection_truth(self):
        camera = RGBDCamera.__new__(RGBDCamera)
        camera.lock = threading.Lock()
        camera.latest = {"sim_time": 1.0, "rgb_zlib": "pixels"}
        camera.latest_gt = {"objects": {"hidden_ball": {"visible_pixels": 0}}}
        self.assertEqual(set(camera.packet()), {"sim_time", "rgb_zlib"})
        self.assertIn("ground_truth", camera.collection_packet())


if __name__ == "__main__":
    unittest.main()
