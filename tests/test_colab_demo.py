import ast
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from PIL import Image

from microduck_qwen import ROOT, colab_demo, demo_gif, explore


class ColabDemoTests(unittest.TestCase):
    def test_notebooks_are_valid_json_and_python_cells_parse(self):
        for name in ("microduck-live-demo.ipynb", "microduck-baseline.ipynb"):
            path = ROOT / "notebooks" / name
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            self.assertGreaterEqual(len(code_cells), 3)
            combined = ""
            for cell in code_cells:
                source = "".join(cell["source"])
                ast.parse(source, filename=f"{name}:{cell['id']}")
                combined += source + "\n"
            self.assertIn("prepare_rendering", combined)

    def test_free_port_pair_returns_two_distinct_ports(self):
        port, control_port = colab_demo._free_port_pair()
        self.assertIsInstance(port, int)
        self.assertIsInstance(control_port, int)
        self.assertNotEqual(port, control_port)
        self.assertGreater(port, 0)
        self.assertGreater(control_port, 0)

    def test_format_action_includes_parameters(self):
        self.assertEqual(colab_demo._format_action({"action": "explore", "color": "blue"}), "explore(color=blue)")
        self.assertEqual(colab_demo._format_action({"action": "return_home"}), "return_home")
        self.assertEqual(colab_demo._format_action(None), "无法解析动作")

    def test_explore_scan_tolerates_mild_overshoot_but_rejects_stalled_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = MagicMock()
            executor.info = {"port": 0}
            explorer = explore.Explorer(executor, Path(tmp) / "explore.json")
            explorer.look = MagicMock(return_value=[])

            executor._bounded.return_value = {"success": False, "achieved": math.radians(105)}
            self.assertEqual(explorer.scan("blue"), [])
            self.assertEqual(executor._bounded.call_count, 3)

            executor._bounded.return_value = {"success": False, "achieved": math.radians(10)}
            with self.assertRaises(RuntimeError) as ctx:
                explorer.scan("blue")
            self.assertIn("10.0 deg", str(ctx.exception))

    def test_demo_gif_composer_and_encoder_produce_valid_animated_gif(self):
        spec = Image.new("RGB", (640, 360), "#1e3a5f")
        cam = Image.new("RGB", (640, 360), "#0f172a")
        mini = Image.new("RGB", (420, 420), "#0f172a")
        frames = [
            demo_gif.quantize_demo_frame(
                demo_gif.compose_demo_frame(
                    spec, cam, mini, "去别的房间找蓝球，再回到起点",
                    [{"action": "explore", "color": "blue"}], elapsed=t, distance=0.8, speed=5, done=(t == 2.0),
                )
            )
            for t in (0.0, 1.0, 2.0)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo.gif"
            info = demo_gif.save_demo_gif(frames, [0.0, 1.0, 2.0], out, speed=5)
            self.assertEqual(info["frames"], 3)
            self.assertTrue(out.is_file())
            with Image.open(out) as gif:
                self.assertTrue(gif.is_animated)
                self.assertEqual(gif.size, (1220, 650))
                self.assertEqual(gif.n_frames, 3)


if __name__ == "__main__":
    unittest.main()
