"""Bounded, real image-conditioned LoRA updates; a memory probe, not a policy.

The single manually reviewed fixture is intentionally NOT a train/val benchmark.
Never deploy this adapter or report its loss as generalization performance.
"""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from . import ROOT


def read_fixture(path):
    path = Path(path).resolve()
    row = json.loads(path.read_text())
    if row.get("purpose") != "memory_probe_only":
        raise ValueError("This entry point accepts only memory_probe_only fixtures")
    image = (path.parent / row["image"]).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    if hashlib.sha256(image.read_bytes()).hexdigest() != row.get("image_sha256"):
        raise ValueError("Probe image differs from the manually reviewed fixture")
    answer = row["answer"]
    if answer != {"status": "visible", "color": "blue", "id": 3}:
        raise ValueError("Unexpected label for the manually reviewed probe fixture")
    return row, image


def swap_snapshot():
    result = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def image_token_id(config):
    value = config.get("image_token_index")
    if value is None:
        value = config.get("image_token_id")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Missing integer image placeholder token ID")
    return value


def main():
    import resource  # Only the macOS MLX memory probe needs Unix process statistics.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path,
                        default=ROOT / "tests/fixtures/vision_lora_probe.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20 or not 1 <= args.rank <= 16 or not 1 <= args.layers <= 28:
        parser.error("probe bounds: steps 1..20, rank 1..16, layers 1..28")
    if args.max_seq_length < 2 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("invalid sequence length or learning rate")
    fixture, image = read_fixture(args.fixture)
    output = (args.output or ROOT / "adapters" /
              ("vl-memory-probe-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"purpose": "memory_probe_only_not_deployable", "status": "started",
              "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
              "fixture_provenance": fixture["provenance"],
              "swap_before": swap_snapshot(), "steps": [], "snapshots": []}
    started = time.perf_counter()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_vlm import load
    from mlx_vlm.trainer.peft.utils import get_peft_model
    from mlx_vlm.trainer.common.utils import enable_model_gradient_checkpointing, save_trainable_weights
    from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset
    from mlx_vlm.trainer.vlm.sft.runtime import collate_vlm_batch, vlm_sft_loss
    from mlx_vlm.utils import load_config
    from .vision import MODEL, REVISION

    def snapshot(stage):
        entry = {"stage": stage, "elapsed_s": time.perf_counter() - started,
                 "mlx_active_bytes": mx.get_active_memory(),
                 "mlx_cache_bytes": mx.get_cache_memory(),
                 "mlx_peak_active_bytes": mx.get_peak_memory(),
                 "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        report["snapshots"].append(entry)
        print(json.dumps(entry), flush=True)

    mx.random.seed(42)
    mx.set_memory_limit(8 * 1024**3)  # MLX guideline, NOT a hard RSS/OS limit.
    mx.set_cache_limit(256 * 1024**2)
    mx.reset_peak_memory()
    device = mx.device_info()
    report.update(model=MODEL, revision=REVISION, device=device,
                  versions={p: importlib.metadata.version(p) for p in ("mlx", "mlx-vlm", "transformers")},
                  config={"batch_size": 1, "rank": args.rank, "last_layers": args.layers,
                          "target_modules": ["q_proj", "v_proj"], "alpha": 2 * args.rank,
                          "gradient_checkpointing": True, "learning_rate": args.learning_rate,
                          "max_seq_length": args.max_seq_length, "seed": 42,
                          "base_quantization": "pre-existing MLX 4-bit, not bitsandbytes NF4",
                          "loss": "assistant completion only", "vision_tower_frozen": True})
    try:
        model_path = ROOT / ".cache/vlm-hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
        if not model_path.is_dir():
            raise FileNotFoundError("Run microduck_qwen.vision download first")
        model, processor = load(str(model_path), trust_remote_code=False)
        config = load_config(str(model_path))
        image_id = image_token_id(config)
        mx.eval(model.parameters())
        snapshot("base_loaded")
        layer_count = len(model.language_model.layers)
        if args.layers > layer_count:
            raise ValueError("Requested more layers than model has")
        selected = {id(layer) for layer in model.language_model.layers[-args.layers:]}
        prefixes = [name for name, module in model.named_modules() if id(module) in selected]
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, (nn.Linear, nn.QuantizedLinear))
                   and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
                   and any(name.startswith(prefix + ".") for prefix in prefixes)]
        if len(targets) != 2 * args.layers:
            raise ValueError("Unexpected Qwen3-VL attention module layout")
        model = get_peft_model(model, targets, rank=args.rank, alpha=2 * args.rank, dropout=0.0)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        if not trainable or any(not name.endswith(("lora_a", "lora_b")) for name in trainable):
            raise ValueError("Only LoRA parameters may be trainable")
        mx.eval(trainable)
        before = {name: np.array(value).copy() for name, value in trainable.items()}
        report["trainable_parameters"] = sum(value.size for value in trainable.values())
        dataset = VisionDataset(
            [{"image": str(image), "messages": [
                {"role": "user", "content": fixture["prompt"]},
                {"role": "assistant", "content": json.dumps(fixture["answer"], separators=(",", ":"))}]}],
            config, processor, {"train_on_completions": True})
        sample = dataset[0]
        length = int(sample["input_ids"].size)
        if length > args.max_seq_length:
            raise ValueError(f"{length} tokens exceed limit; refuse truncation")
        batch = collate_vlm_batch([sample], args.max_seq_length,
                                 pad_token_id=processor.tokenizer.pad_token_id,
                                 image_token_id=image_id)
        mx.eval(batch)
        report["batch_shapes"] = {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
        report["input_tokens"] = length
        report["supervised_tokens"] = int(batch["completion_mask"][:, 1:].sum().item())
        report["image_tokens"] = int((batch["input_ids"] == image_id).sum().item())
        if report["supervised_tokens"] == 0 or report["image_tokens"] == 0:
            raise ValueError("Must contain both image tokens and supervised answer tokens")
        enable_model_gradient_checkpointing(model)
        model.train()
        optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
        value_and_grad = nn.value_and_grad(model, lambda m: vlm_sft_loss(m, batch, train_on_completions=True)[0])
        snapshot("prepared")
        for step in range(1, args.steps + 1):
            tick = time.perf_counter()
            loss, gradients = value_and_grad(model)
            optimizer.update(model, gradients)
            mx.eval(loss, model.trainable_parameters(), optimizer.state)
            value = float(loss.item())
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite training loss")
            report["steps"].append({"step": step, "loss": value, "elapsed_s": time.perf_counter() - tick})
            del gradients
            snapshot(f"update_{step}")
        changed = [name for name, value in tree_flatten(model.trainable_parameters())
                   if not np.array_equal(before[name], np.array(value))]
        report["changed_adapter_tensors"] = len(changed)
        if not changed:
            raise ValueError("No adapter parameter was updated")
        save_trainable_weights(model, output / "adapters.safetensors")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        snapshot("final")
        report.update(elapsed_s=time.perf_counter() - started, swap_after=swap_snapshot())
        (output / "memory_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Memory report: {output / 'memory_report.json'}", flush=True)


if __name__ == "__main__":
    if sys.platform != "darwin":
        raise SystemExit("This probe reports Darwin RSS bytes and targets Apple Silicon MLX")
    main()
