"""LoRA SFT with exactly the same non-thinking chat prefix as inference."""

import argparse
import copy
import importlib.metadata
import json
import platform
from types import SimpleNamespace

from . import ROOT
from .contract import compact, messages
from .download import model_path


class Dataset:
    def __init__(self, rows, tokenizer, limit):
        self.samples = []
        self.max_length = 0
        for row in rows:
            prefix = tokenizer.apply_chat_template(messages(row["instruction"]), tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            suffix = tokenizer.encode(compact(row["expected"]), add_special_tokens=False)
            tokens = list(prefix) + list(suffix) + [tokenizer.eos_token_id]
            if len(tokens) > limit:
                raise ValueError(f"样本长 {len(tokens)} > {limit}；拒绝静默截断目标")
            self.samples.append((tokens, len(prefix)))
            self.max_length = max(self.max_length, len(tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def process(self, sample):
        return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--adapter")
    parser.add_argument("--data", default="data")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-adapter-file")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        from .torch_backend import train

        args.adapter = args.adapter or "adapters/nav-torch"
        return train(args)
    args.adapter = args.adapter or "adapters/nav"
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model

    adapter = ROOT / args.adapter
    if (adapter / "adapters.safetensors").exists():
        raise FileExistsError(f"已有训练结果 {adapter}；请选择新的 --adapter 保存本次训练。")
    config = copy.deepcopy(CONFIG_DEFAULTS)
    data_dir = ROOT / args.data
    config.update(model=str(model_path()), data=str(data_dir), train=True,
        adapter_path=str(adapter), seed=42, iters=args.iters, batch_size=2,
        num_layers=16, learning_rate=args.learning_rate, max_seq_length=768,
        mask_prompt=True, val_batches=4, steps_per_eval=50, steps_per_report=10,
        save_every=50, lora_parameters={"rank": 16, "dropout": 0.0, "scale": 32.0})
    if args.resume_adapter_file:
        resume = ROOT / args.resume_adapter_file
        if not resume.exists():
            raise FileNotFoundError(resume)
        config["resume_adapter_file"] = str(resume)
    np.random.seed(42)
    mx.random.seed(42)
    model, tokenizer = load(config["model"])
    datasets = {}
    for split in ("train", "valid"):
        rows = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        datasets[split] = Dataset(rows, tokenizer, config["max_seq_length"])
    print(f"Training examples: {len(datasets['train'])}, max tokens: {datasets['train'].max_length}", flush=True)
    train_model(SimpleNamespace(**config), model, datasets["train"], datasets["valid"])
    versions = {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers")}
    (adapter / "provenance.json").write_text(json.dumps({"versions": versions,
        "dataset": json.loads((data_dir / "manifest.json").read_text()),
        "model": json.loads((ROOT / "runs/model.json").read_text()),
        "chat_template": "enable_thinking=False; assistant completion only loss", "iterations": args.iters,
        "resume_adapter_file": args.resume_adapter_file, "optimizer_state_resumed": False,
        "training_config": {k: config[k] for k in ("seed", "batch_size", "num_layers", "learning_rate", "max_seq_length", "lora_parameters")},
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved trained LoRA: {adapter}", flush=True)


if __name__ == "__main__":
    main()
