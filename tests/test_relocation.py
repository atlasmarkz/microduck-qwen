import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microduck_qwen import DEFAULT_MODEL
from microduck_qwen.download import model_path


class RelocationTests(unittest.TestCase):
    def test_relative_manifest_is_resolved_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "models/local"
            snapshot.mkdir(parents=True)
            (root / "runs").mkdir()
            (root / "runs/model.json").write_text(json.dumps({"snapshot": "models/local"}))
            with patch("microduck_qwen.download.ROOT", root):
                self.assertEqual(model_path(), snapshot.resolve())

    def test_relocated_cache_takes_precedence_over_historical_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / ".cache/hub/models--Qwen--Qwen3-0.6B/snapshots/test-revision"
            snapshot.mkdir(parents=True)
            (root / "runs").mkdir()
            (root / "runs/model.json").write_text(json.dumps({
                "model": DEFAULT_MODEL, "revision": "test-revision",
                "snapshot": "/nonexistent-historical-project/snapshot",
            }))
            with patch("microduck_qwen.download.ROOT", root):
                self.assertEqual(model_path(), snapshot.resolve())

    def test_missing_model_still_requires_an_explicit_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("microduck_qwen.download.ROOT", Path(directory)):
                with self.assertRaises(FileNotFoundError):
                    model_path()


if __name__ == "__main__":
    unittest.main()
