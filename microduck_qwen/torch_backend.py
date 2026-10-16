"""Text navigation SFT and inference with PyTorch/PEFT on Windows or Linux."""

import importlib.metadata
import json
import math
import time
from pathlib import Path

from . import ROOT
from .contract import messages, validate
from .download import model_path
from .train import Dataset


def load(adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if adapter is not None and not (Path(adapter) / "adapter_config.json").is_file():
        raise FileNotFoundError(f"没有 PEFT LoRA 权重: {adapter}；先训练，或使用 --base。")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path()), dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval(), AutoTokenizer.from_pretrained(str(model_path()))


def predict(model, tokenizer, instruction, adapter):
    import torch

    prompt = tokenizer.apply_chat_template(messages(instruction), tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    elapsed = time.perf_counter() - started
    return {"instruction": instruction, "plan": validate(raw), "raw": raw,
        "latency_s": elapsed, "adapter": str(adapter) if adapter else None}


def collate(samples, pad_token_id):
    import torch

    length = max(len(tokens) for tokens, _ in samples)
    inputs, labels, masks = [], [], []
    for tokens, prefix in samples:
        padding = length - len(tokens)
        inputs.append(tokens + [pad_token_id] * padding)
        labels.append([-100] * prefix + tokens[prefix:] + [-100] * padding)
        masks.append([1] * len(tokens) + [0] * padding)
    return {key: torch.tensor(value) for key, value in
        (("input_ids", inputs), ("labels", labels), ("attention_mask", masks))}


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model

    if args.iters < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("iters 和 learning-rate 必须为正数且有限。")
    if args.resume_adapter_file:
        raise ValueError("PyTorch 不接受 MLX --resume-adapter-file；适配器格式不同。")
    adapter = ROOT / args.adapter
    if adapter.exists() and any(adapter.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有输出: {adapter}")
    torch.manual_seed(42)
    model, tokenizer = load()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16,
        lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=list(range(model.config.num_hidden_layers - 16, model.config.num_hidden_layers))))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    data_dir = ROOT / args.data
    datasets = {split: Dataset([json.loads(line) for line in
        (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()], tokenizer, 768)
        for split in ("train", "valid")}
    if not all(len(dataset) for dataset in datasets.values()):
        raise ValueError("训练与验证集不能为空。")
    from functools import partial
    loader = torch.utils.data.DataLoader(datasets["train"], batch_size=2, shuffle=True,
        collate_fn=partial(collate, pad_token_id=tokenizer.eos_token_id))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate, weight_decay=0.0)
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    losses = []
    started = time.perf_counter()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    while len(losses) < args.iters:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise RuntimeError("训练 loss 非有限值；未保存权重。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            if len(losses) == 1 or len(losses) % 10 == 0:
                print(f"step={len(losses)}/{args.iters} loss={losses[-1]:.4f}", flush=True)
            if len(losses) == args.iters:
                break
    elapsed = time.perf_counter() - started
    model.eval()
    valid_loss, valid_tokens = 0.0, 0
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(datasets["valid"], batch_size=2,
                collate_fn=partial(collate, pad_token_id=tokenizer.eos_token_id)):
            count = (batch["labels"][:, 1:] != -100).sum().item()
            value = model(**{k: v.to(model.device) for k, v in batch.items()}).loss.item()
            valid_loss += value * count
            valid_tokens += count
    changed = sum(not torch.equal(initial[name], p.detach().cpu())
        for name, p in model.named_parameters() if p.requires_grad)
    if not changed:
        raise RuntimeError("LoRA 参数未更新；未保存权重。")
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    report = {"backend": "torch", "versions": {p: importlib.metadata.version(p)
        for p in ("torch", "transformers", "peft")}, "device": model.device.type,
        "iterations": args.iters, "seed": 42, "batch_size": 2, "learning_rate": args.learning_rate,
        "lora": {"rank": 16, "alpha": 32, "last_layers": 16, "target_modules": "all-linear"},
        "max_seq_length": 768, "completion_only_loss": True, "enable_thinking": False,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "changed_tensors": changed, "train_losses": losses, "valid_loss": valid_loss / valid_tokens,
        "elapsed_s": elapsed, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if model.device.type == "cuda" else None,
        "dataset": json.loads((data_dir / "manifest.json").read_text(encoding="utf-8")),
        "model": {k: v for k, v in json.loads((ROOT / "runs/model.json").read_text(encoding="utf-8")).items()
            if k != "snapshot"},
        "note": "PEFT adapters are separate from historical MLX weights; no physical navigation evaluation."}
    (adapter / "provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("train_losses", "dataset")}, ensure_ascii=False), flush=True)
