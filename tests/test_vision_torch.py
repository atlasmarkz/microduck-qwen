import json
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from microduck_qwen.vision_torch import MODEL, REVISION, RewardGuidedPlanner, TorchAgentPlanner, actor_rows_with_reward_replay, agent_reward_examples, checked_adapter, encoded, reward_prompt
from microduck_qwen.sim_data import digest


@unittest.skipUnless(find_spec("torch"), "optional PyTorch backend")
class VisionTorchTests(unittest.TestCase):
    def test_2b_actor_and_reward_adapters_cannot_enter_4b_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for task in ("agent", "reward"):
                (path / "training_report.json").write_text(json.dumps({"status": "completed", "task": task,
                    "model": "Qwen/Qwen3-VL-2B-Instruct", "revision": "89644892e4d85e24eaac8bacfd4f463576704203"}))
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    checked_adapter(path, task)

    def test_actor_replay_requires_reward_acceptance_and_independent_verification(self):
        valid = {"id": "ok", "split": "train", "verified": True, "reward_verified_accepted": True}
        rows = [valid, dict(valid, id="rejected", reward_verified_accepted=False),
                dict(valid, id="failed", verified=False), dict(valid, id="held-out", split="valid")]
        replay = actor_rows_with_reward_replay(rows)
        self.assertEqual([r["id"] for r in replay], ["ok", "rejected", "failed", "held-out", "ok"])
        self.assertEqual(len(rows), 4)

    def test_legacy_reward_adapter_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            weights = path / "adapter_model.safetensors"
            weights.write_bytes(b"old reward")
            (path / "training_report.json").write_text(json.dumps({"status": "completed", "task": "reward",
                "model": MODEL, "revision": REVISION, "adapter_sha256": digest(weights)}))
            with self.assertRaisesRegex(ValueError, "memory"):
                checked_adapter(path, "reward")

    def test_reward_uses_history_and_keeps_actual_premature_done_as_negative(self):
        memory = {"home_recorded": True, "odometry_relative_home": {"x": 0.2, "y": 0, "yaw_deg": 0},
                  "completed": [{"action": "move", "direction": "forward", "distance_m": 0.2}]}
        row = {"id": "corrected", "instruction": "前进后回家", "memory": memory,
               "expected": {"action": "return_home"}, "reward_proposals": [{"action": "done"}] * 2}
        examples = agent_reward_examples([row])
        done = [r for r in examples if r.get("proposed_action") == {"action": "done"}]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["reward_label"], 0)
        self.assertIn('"completed":[{"action":"move"', done[0]["prompt"])

    def test_reward_ranks_unique_actions_and_rejects_low_scores(self):
        planner = object.__new__(RewardGuidedPlanner)
        planner.model, planner.processor = Mock(), Mock()
        planner.samples, planner.threshold = 3, 0.5
        planner.adapter, planner.reward_adapter = {}, {}
        memory = {"home_recorded": False, "odometry_relative_home": None, "completed": []}
        outputs = [('{"action":"done"}', 1), ('{"action":"stop"}', 1), ('{"action":"done"}', 1)]
        with patch("microduck_qwen.vision_torch.generate_text", side_effect=outputs), patch("microduck_qwen.vision_torch.reward_score", side_effect=[0.1, 0.8]) as scorer:
            result = planner.predict("停下来", "image.png", memory)
        self.assertEqual(result["action"], {"action": "stop"})
        self.assertEqual(result["distinct_proposals"], 2)
        self.assertEqual(len(result["proposals"]), 3)
        self.assertEqual(scorer.call_count, 2)
        self.assertEqual(planner.model.set_adapter.call_args.args, ("default",))
        with patch("microduck_qwen.vision_torch.generate_text", side_effect=outputs), patch("microduck_qwen.vision_torch.reward_score", return_value=0.1):
            rejected = planner.predict("停下来", "image.png", memory)
        self.assertIsNone(rejected["action"])
        self.assertFalse(rejected["reward_accepted"])
        with patch("microduck_qwen.vision_torch.generate_text", return_value=("invalid", 1)), patch("microduck_qwen.vision_torch.reward_score") as scorer:
            invalid = planner.predict("停下来", "image.png", memory)
        self.assertEqual(invalid["distinct_proposals"], 0)
        self.assertIsNone(invalid["action"])
        scorer.assert_not_called()
        with patch("microduck_qwen.vision_torch.generate_text", side_effect=outputs), patch("microduck_qwen.vision_torch.reward_score", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                planner.predict("停下来", "image.png", memory)
        self.assertEqual(planner.model.set_adapter.call_args.args, ("default",))

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
