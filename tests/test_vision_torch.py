import json
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from microduck_qwen.vision_torch import TorchAgentPlanner, agent_reward_examples, encoded, reward_prompt


@unittest.skipUnless(find_spec("torch"), "optional PyTorch backend")
class VisionTorchTests(unittest.TestCase):
    def test_agent_adapter_metadata_remains_serializable_for_evaluators(self):
        model = Mock()
        model.eval.return_value = model
        peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=Mock(return_value=model)))
        with patch.dict(sys.modules, {"peft": peft}), patch("microduck_qwen.vision_torch.load_model", return_value=(model, Mock())), patch("microduck_qwen.vision_torch.checked_adapter", return_value={"adapter_sha256": "verified"}) as checked:
            planner = TorchAgentPlanner(Path("adapters/example"))
        checked.assert_called_once_with(Path("adapters/example"), "agent")
        self.assertEqual(json.loads(json.dumps(planner.adapter))["sha256"], "verified")

    def test_image_and_prompt_are_excluded_from_completion_loss(self):
        import torch
        from PIL import Image

        class Processor:
            image_token_id = 7
            tokenizer = SimpleNamespace(encode=lambda *a, **k: [4, 5], eos_token_id=9)
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "prefix"
            def __call__(self, **kwargs):
                return {"input_ids": torch.tensor([[1, 7, 2]]), "attention_mask": torch.ones((1, 3), dtype=torch.long),
                    "mm_token_type_ids": torch.tensor([[0, 1, 0]]), "pixel_values": torch.ones((4, 6))}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (32, 32)).save(path)
            processor = Processor()
            inputs = encoded(processor, path, "instruction only", "yes")
            self.assertEqual(inputs["labels"].tolist(), [[-100, -100, -100, 4, 5, 9]])
            self.assertEqual(inputs["mm_token_type_ids"].tolist(), [[0, 1, 0, 0, 0, 0]])
            self.assertEqual(tuple(inputs["pixel_values"].shape), (4, 6))
            encoded(processor, [path, path], "before and after", "no")
            self.assertEqual([part["type"] for part in processor.messages[0]["content"]], ["image", "image", "text"])
            with self.assertRaises(ValueError):
                encoded(processor, path, "instruction", "yes", limit=5)

    def test_reward_negatives_never_duplicate_the_correct_skill(self):
        expected = {"action": "move", "direction": "forward", "distance_m": 0.2}
        row = {"id": "train-row", "image": "step-000-candidates.png", "instruction": "前进0.2米", "expected": expected}
        examples = agent_reward_examples([row])
        self.assertEqual([r["reward_label"] for r in examples], [1, 0, 0])
        for negative in examples[1:]:
            self.assertNotEqual(negative["proposed_action"], expected)
        self.assertNotIn("scene_seed", reward_prompt("前进0.2米", expected))
