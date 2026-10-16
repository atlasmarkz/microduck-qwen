import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from importlib.util import find_spec

from microduck_qwen.locking import lock_exclusive
from microduck_qwen.train import Dataset
from microduck_qwen.torch_backend import collate, load, train
from types import SimpleNamespace


@unittest.skipUnless(find_spec("torch") and find_spec("peft"), "optional PyTorch/PEFT dependencies: install .[torch]")
class TorchBackendTests(unittest.TestCase):
    def test_completion_mask_and_padding(self):
        batch = collate([([10, 11, 12, 13], 2), ([20, 21, 22], 1)], 99)
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 12, 13], [-100, 21, 22, -100]])
        self.assertEqual(batch["input_ids"].tolist()[1], [20, 21, 22, 99])
        self.assertEqual(batch["attention_mask"].tolist()[1], [1, 1, 1, 0])

    def test_long_sample_is_rejected(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2],
            encode=lambda *a, **k: [3, 4], eos_token_id=5)
        with self.assertRaises(ValueError):
            Dataset([{"instruction": "stop", "expected": {"status": "stop", "goals": []}}], tokenizer, 4)

    def test_missing_adapter_does_not_load_base_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load(Path(directory))

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "keep.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                train(SimpleNamespace(iters=1, learning_rate=1e-4, resume_adapter_file=None, adapter=directory))
            self.assertEqual(marker.read_text(), "keep")

    def test_tiny_qwen_trains_and_saves_reloadable_adapter(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import PeftModel

        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=17, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        model = Qwen3ForCausalLM(config)
        frozen = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2],
            encode=lambda *a, **k: [3, 4], eos_token_id=5, save_pretrained=lambda *a: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "runs").mkdir()
            row = {"instruction": "stop", "expected": {"status": "stop", "goals": []}}
            for split in ("train", "valid"):
                (root / f"data/{split}.jsonl").write_text(json.dumps(row) + "\n")
            (root / "data/manifest.json").write_text('{}')
            (root / "runs/model.json").write_text('{"model":"tiny-test","revision":"test"}')
            with patch("microduck_qwen.torch_backend.ROOT", root), patch("microduck_qwen.torch_backend.load", return_value=(model, tokenizer)):
                train(SimpleNamespace(iters=2, learning_rate=1e-4, resume_adapter_file=None,
                    adapter="adapters/test", data="data"))
            adapter = root / "adapters/test"
            report = json.loads((adapter / "provenance.json").read_text())
            self.assertGreater(report["changed_tensors"], 0)
            self.assertEqual(len(report["train_losses"]), 2)
            self.assertTrue(torch.equal(frozen, model.model.layers[0].self_attn.q_proj.weight))
            reloaded = PeftModel.from_pretrained(Qwen3ForCausalLM(config), adapter)
            self.assertTrue(torch.isfinite(reloaded(input_ids=torch.tensor([[1, 2, 3]])).logits).all())

    def test_controller_lock_excludes_second_handle_and_releases_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "navigation.lock"
            with path.open("a") as first:
                lock_exclusive(first)
                with path.open("a") as second:
                    with self.assertRaises(BlockingIOError):
                        lock_exclusive(second)
            with path.open("a") as third:
                lock_exclusive(third)
