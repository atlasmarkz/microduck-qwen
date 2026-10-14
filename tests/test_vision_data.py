import json
from pathlib import Path
import tempfile
import unittest
import threading

try:
    import numpy as np
    import PIL
except ModuleNotFoundError:
    raise unittest.SkipTest("optional vision dependencies required")

from microduck_qwen.vision import Frame, MODEL, REVISION, adapter_metadata, visual_prompt, explicit_stop_instruction
from microduck_qwen.vision_data import bbox_iou, digest, load_splits, verify_objects, visual_examples
from microduck_qwen.vision_sft import model_rows
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

    def test_labels_are_verified_against_segmentation(self):
        verified = verify_objects(*self.inputs())
        expected = dict((instruction, answer) for instruction, answer, _ in visual_examples(verified))
        self.assertEqual(expected["走到红色球旁边"], {"status": "visible", "color": "red", "id": 9})
        self.assertEqual(expected["走到绿色球旁边"], {"status": "not_visible", "color": None, "id": None})

    def test_detector_failure_is_discarded_not_called_absence(self):
        scene, frame, gt, _ = self.inputs()
        verified = verify_objects(scene, frame, gt, [])
        self.assertEqual(verified[0]["status"], "discard")
        instructions = [row[0] for row in visual_examples(verified)]
        self.assertNotIn("走到红色球旁边", instructions)
        self.assertNotIn("走到画面左边的球旁边", instructions)

    def test_partial_target_has_null_identity(self):
        scene, frame, gt, candidates = self.inputs()
        gt["objects"]["ball_0"]["visible_pixels"] = 50
        verified = verify_objects(scene, frame, gt, candidates)
        self.assertEqual(verified[0]["status"], "partial")
        expected = next(answer for instruction, answer, _ in visual_examples(verified) if instruction == "走到红色球旁边")
        self.assertEqual(expected, {"status": "partial", "color": None, "id": None})

    def test_iou_does_not_match_separate_objects(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_privileged_metadata_never_becomes_a_model_input(self):
        raw = [{"image": "rgb.png", "instruction": "走到画面左边的球旁边",
                "expected": {"status": "visible", "color": "blue", "id": 3},
                "scene_seed": 987654, "ground_truth": {"secret_position": [1.0, 2.0]}}]
        rows = model_rows(raw, Path("/tmp"))
        self.assertEqual(set(rows[0]), {"image", "messages"})
        self.assertNotIn("secret_position", json.dumps(rows))
        self.assertNotIn("987654", json.dumps(rows))
        self.assertEqual(rows[0]["messages"][0]["content"], visual_prompt(raw[0]["instruction"]))

    def make_dataset(self, directory, duplicate_seed=False):
        manifest = {"format": "simulation_visual_v1", "status": "completed", "splits": {}}
        for index, split in enumerate(("train", "valid", "test")):
            image = directory / f"{split}.png"
            image.write_bytes(split.encode())
            row = {"image": image.name, "image_sha256": digest(image), "split": split,
                   "scene_seed": 1 if duplicate_seed else index,
                   "expected": {"status": "stop", "color": None, "id": None}}
            path = directory / f"{split}.jsonl"
            path.write_text(json.dumps(row) + "\n")
            manifest["splits"][split] = {"sha256": digest(path)}
        (directory / "manifest.json").write_text(json.dumps(manifest))

    def test_group_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_dataset(directory, duplicate_seed=True)
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_splits(directory)

    def test_modified_images_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_dataset(directory)
            load_splits(directory)
            (directory / "valid.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_splits(directory)

    def test_memory_probe_cannot_be_loaded_as_task_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            weights = directory / "adapters.safetensors"
            weights.write_bytes(b"weights")
            (directory / "training_report.json").write_text(json.dumps({
                "format": "memory_probe", "status": "completed", "model": MODEL, "revision": REVISION,
                "adapter_sha256": digest(weights)}))
            with self.assertRaisesRegex(ValueError, "compatible vision SFT"):
                adapter_metadata(directory)

    def test_default_camera_packet_never_exposes_collection_truth(self):
        camera = RGBDCamera.__new__(RGBDCamera)
        camera.lock = threading.Lock()
        camera.latest = {"sim_time": 1.0, "rgb_zlib": "pixels"}
        camera.latest_gt = {"objects": {"hidden_ball": {"visible_pixels": 0}}}
        self.assertEqual(set(camera.packet()), {"sim_time", "rgb_zlib"})
        self.assertIn("ground_truth", camera.collection_packet())

    def test_hallucinated_stop_is_not_navigation_success(self):
        for instruction in ("停下来", "请停止移动", "stop", "Please stop now!"):
            self.assertTrue(explicit_stop_instruction(instruction))
        for instruction in ("走到红球旁边", "走到球旁停下来", "不要停", "go to the ball then stop"):
            self.assertFalse(explicit_stop_instruction(instruction))


if __name__ == "__main__":
    unittest.main()
