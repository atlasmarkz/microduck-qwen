"""Actual Qwen inference. Invalid output is reported, never replaced by a rules-based answer."""

import time
import platform
from pathlib import Path

from . import ROOT
from .contract import messages, validate
from .download import model_path


class Planner:
    def __init__(self, *, base=False, adapter=None):
        self.backend = "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "torch"
        if self.backend == "torch":
            from .torch_backend import load

            self.adapter = None if base else Path(adapter or ROOT / "adapters/nav-torch")
            self.model, self.tokenizer = load(self.adapter)
            return
        from mlx_lm import load

        self.adapter = None if base else Path(adapter or ROOT / "adapters/nav-v2")
        if self.adapter is not None and not (self.adapter / "adapters.safetensors").exists():
            raise FileNotFoundError(f"没有已训练的 LoRA 权重: {self.adapter}。先运行训练，或显式使用 --base 比较原模型。")
        self.model, self.tokenizer = load(str(model_path()), adapter_path=str(self.adapter) if self.adapter else None)

    def predict(self, instruction):
        if self.backend == "torch":
            from .torch_backend import predict

            return predict(self.model, self.tokenizer, instruction, self.adapter)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages(instruction), tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        started = time.perf_counter()
        raw = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=256,
                       sampler=make_sampler(temp=0.0), verbose=False)
        elapsed = time.perf_counter() - started
        plan = validate(raw)
        return {"instruction": instruction, "plan": plan, "raw": raw, "latency_s": elapsed,
                "adapter": str(self.adapter) if self.adapter else None}
