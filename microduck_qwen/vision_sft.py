"""Image-conditioned skill-agent LoRA SFT on simulator-collected, scene-disjoint data."""

import argparse
from collections import OrderedDict
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

from . import ROOT
from .agent_data import load_splits, model_rows
from .sim_data import digest
from .vision import MODEL, REVISION
from .vision_lora_probe import image_token_id, swap_snapshot


def train(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm import load
    from mlx_vlm.trainer.peft.utils import get_peft_model
    from mlx_vlm.trainer.common.utils import enable_model_gradient_checkpointing, save_trainable_weights
    from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset
    from mlx_vlm.trainer.vlm.sft.runtime import collate_vlm_batch, vlm_sft_loss
    from mlx_vlm.utils import load_config

    manifest, splits = load_splits(args.data)
    output = (args.output or ROOT / "adapters" / ("agent-sim-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"format": "agent_sft_v1", "status": "started", "model": MODEL, "revision": REVISION,
              "dataset_manifest_sha256": digest(args.data / "manifest.json"), "dataset": str(args.data),
              "split_counts": {k: len(v) for k, v in splits.items()}, "seed": args.seed,
              "config": {"rank": args.rank, "last_layers": args.layers, "alpha": 2 * args.rank,
                         "targets": ["q_proj", "v_proj"], "optimizer_steps": args.steps,
                         "gradient_accumulation": args.accumulation, "microbatch": 1,
                         "learning_rate": args.learning_rate, "max_seq_length": args.max_seq_length,
                         "vision_frozen": True, "gradient_checkpointing": True,
                         "loss": "assistant completion only", "base": "MLX 4-bit"},
              "swap_before": swap_snapshot(), "steps": []}
    report["source_sha256"] = {name: digest(ROOT / "microduck_qwen" / name)
                               for name in ("vision_sft.py", "agent.py", "agent_data.py", "skills.py")}
    started = time.perf_counter()
    mx.random.seed(args.seed)
    mx.set_memory_limit(8 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    mx.reset_peak_memory()
    try:
        path = ROOT / ".cache/vlm-hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
        if not path.is_dir():
            raise FileNotFoundError("Run vision download first")
        model, processor = load(str(path), trust_remote_code=False)
        config = load_config(str(path))
        selected = {id(layer) for layer in model.language_model.layers[-args.layers:]}
        if len(selected) != args.layers:
            raise ValueError("too many LoRA layers")
        prefixes = [name for name, module in model.named_modules() if id(module) in selected]
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, (nn.Linear, nn.QuantizedLinear))
                   and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
                   and any(name.startswith(prefix + ".") for prefix in prefixes)]
        if len(targets) != 2 * args.layers:
            raise ValueError("unexpected attention module layout")
        if getattr(args, "resume_adapter", None):
            from .agent import adapter_metadata
            from mlx_vlm.trainer.peft.utils import load_adapters
            parent = adapter_metadata(args.resume_adapter)
            parent_report = json.loads((Path(parent["path"]) / "training_report.json").read_text())
            if parent_report["config"]["rank"] != args.rank or parent_report["config"]["last_layers"] != args.layers:
                raise ValueError("resume LoRA config mismatch")
            model = load_adapters(model, parent["path"])
            report.update(parent_adapter=parent, optimizer_resume=False)
        else:
            model = get_peft_model(model, targets, rank=args.rank, alpha=2 * args.rank, dropout=0.0)
        parameters = dict(tree_flatten(model.trainable_parameters()))
        if not parameters or any(not name.endswith(("lora_a", "lora_b")) for name in parameters):
            raise ValueError("only selected LoRA parameters may be trained")
        mx.eval(parameters)
        before = {name: np.array(value).copy() for name, value in parameters.items()}
        report["trainable_parameters"] = sum(value.size for value in parameters.values())
        report["versions"] = {p: importlib.metadata.version(p) for p in ("mlx", "mlx-vlm", "transformers")}
        datasets = {split: VisionDataset(model_rows(splits[split], args.data), config, processor,
                                        {"train_on_completions": True}) for split in ("train", "valid")}
        image_id = image_token_id(config)
        cache = OrderedDict()
        token_lengths = []

        def batch(split, index):
            key = (split, index)
            if key not in cache:
                sample = datasets[split][index]
                length = int(sample["input_ids"].size)
                if length > args.max_seq_length:
                    raise ValueError(f"{split} row {index}: {length} tokens exceed limit; no truncation allowed")
                result = collate_vlm_batch([sample], args.max_seq_length,
                                          pad_token_id=processor.tokenizer.pad_token_id, image_token_id=image_id)
                if int(result["completion_mask"][:, 1:].sum().item()) == 0 or int((result["input_ids"] == image_id).sum().item()) == 0:
                    raise ValueError("image and supervised answer tokens required")
                mx.eval(result)
                cache[key] = result
                token_lengths.append(length)
                if len(cache) > 8:
                    cache.popitem(last=False)
            cache.move_to_end(key)
            return cache[key]

        def valid_loss():
            model.eval()
            values = []
            for index in range(min(3, len(datasets["valid"]))):
                value, _ = vlm_sft_loss(model, batch("valid", index), train_on_completions=True)
                values.append(float(value.item()))
            model.train()
            return sum(values) / len(values)

        enable_model_gradient_checkpointing(model)
        report["valid_loss_first_3_before"] = valid_loss()
        optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
        gradient = nn.value_and_grad(model, lambda m, b: vlm_sft_loss(m, b, train_on_completions=True)[0])
        rng, order, position = random.Random(args.seed), [], 0
        for step in range(1, args.steps + 1):
            tick, total, losses, row_ids = time.perf_counter(), None, [], []
            for _ in range(args.accumulation):
                if position >= len(order):
                    order = list(range(len(datasets["train"])))
                    rng.shuffle(order)
                    position = 0
                index = order[position]
                position += 1
                row_ids.append(splits["train"][index]["id"])
                value, grads = gradient(model, batch("train", index))
                mx.eval(value, grads)
                loss = float(value.item())
                if not math.isfinite(loss):
                    raise FloatingPointError("nonfinite loss; refuse update")
                losses.append(loss)
                total = grads if total is None else tree_map(lambda a, b: a + b, total, grads)
                mx.eval(total)
                del grads
            total = tree_map(lambda x: x / args.accumulation, total)
            optimizer.update(model, total)
            mx.eval(model.trainable_parameters(), optimizer.state)
            del total
            info = {"step": step, "loss": sum(losses) / len(losses), "row_ids": row_ids,
                    "elapsed_s": time.perf_counter() - tick, "mlx_peak_bytes": mx.get_peak_memory()}
            report["steps"].append(info)
            print(json.dumps(info), flush=True)
        report["valid_loss_first_3_after"] = valid_loss()
        changed = [name for name, value in tree_flatten(model.trainable_parameters())
                   if not np.array_equal(before[name], np.array(value))]
        if not changed:
            raise ValueError("no parameter changed")
        report["changed_adapter_tensors"] = len(changed)
        save_trainable_weights(model, output / "adapters.safetensors")
        report.update(status="completed", adapter_sha256=digest(output / "adapters.safetensors"),
                      adapter_config_sha256=digest(output / "adapter_config.json"),
                      observed_token_lengths={"min": min(token_lengths), "max": max(token_lengths)})
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(elapsed_s=time.perf_counter() - started, mlx_peak_bytes=mx.get_peak_memory(), swap_after=swap_snapshot())
        (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"TRAINING REPORT: {output / 'training_report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train")
    trainer.add_argument("--data", type=Path, required=True)
    trainer.add_argument("--output", type=Path)
    trainer.add_argument("--seed", type=int, default=42)
    trainer.add_argument("--resume-adapter", type=Path)
    trainer.add_argument("--steps", type=int, default=100)
    trainer.add_argument("--rank", type=int, default=8)
    trainer.add_argument("--layers", type=int, default=8)
    trainer.add_argument("--accumulation", type=int, default=4)
    trainer.add_argument("--learning-rate", type=float, default=5e-5)
    trainer.add_argument("--max-seq-length", type=int, default=768)
    args = parser.parse_args()
    args.data = args.data.resolve()
    if os.name == "nt":
        parser.error("Windows skill-agent training uses microduck_qwen.vision_torch train; see WINDOWS_VISION.md")
    if not (1 <= args.steps <= 1000 and 1 <= args.rank <= 16 and 1 <= args.layers <= 28
            and 1 <= args.accumulation <= 16 and 2 <= args.max_seq_length <= 2048
            and math.isfinite(args.learning_rate) and args.learning_rate > 0):
        parser.error("invalid or excessive small-Mac training configuration")
    train(args)


if __name__ == "__main__":
    main()
