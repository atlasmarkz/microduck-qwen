import json
from pathlib import Path
import tempfile
import unittest

from microduck_qwen.vision_lora_probe import image_token_id, read_fixture


class VisionLoRAProbeTests(unittest.TestCase):
    def test_both_upstream_image_token_config_names(self):
        self.assertEqual(image_token_id({"image_token_id": 151655}), 151655)
        self.assertEqual(image_token_id({"image_token_index": 151655}), 151655)
        self.assertEqual(image_token_id({"image_token_index": None, "image_token_id": 151655}), 151655)
        for config in ({}, {"image_token_id": True}, {"image_token_id": "151655"}):
            with self.assertRaises(ValueError):
                image_token_id(config)

    def test_reviewed_fixture_is_present_and_probe_only(self):
        fixture, image = read_fixture(Path(__file__).parent / "fixtures/vision_lora_probe.json")
        self.assertTrue(image.is_file())
        self.assertEqual(fixture["purpose"], "memory_probe_only")
        self.assertEqual(fixture["answer"]["id"], 3)

    def test_regular_dataset_cannot_be_mislabeled_as_memory_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps({"purpose": "training"}))
            with self.assertRaises(ValueError):
                read_fixture(path)

    def test_replaced_image_cannot_keep_reviewed_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            (Path(directory) / "image.png").write_bytes(b"different image")
            path.write_text(json.dumps({"purpose": "memory_probe_only", "image": "image.png", "image_sha256": "incorrect"}))
            with self.assertRaises(ValueError):
                read_fixture(path)


if __name__ == "__main__":
    unittest.main()
