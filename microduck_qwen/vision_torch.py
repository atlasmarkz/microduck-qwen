"""Windows CUDA Qwen3-VL grounding and binary action rewards with 4-bit LoRA."""

import argparse
from contextlib import ExitStack
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from . import ROOT
from .vision import VisualOutputError, parse_visual_choice, visual_prompt
from .vision_data import digest, load_splits

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_PATH = ROOT / ".cache/vlm-torch-hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots" / REVISION


def reward_prompt(instruction, action):
    return ("Judge whether this proposed visual action correctly follows the instruction and visible image. "
        "Partial, absent or ambiguous targets must not be approached. Stop and unsupported instructions need their corresponding status. "
        f"Instruction: {instruction}\nProposed action: {json.dumps(action, separators=(',', ':'))}\n"
        "Answer exactly yes or no.")


def outcome_prompt(instruction):
    return ("First image: before navigation. Second image: current state. "
        "Judge whether the robot has completed approaching the ball specified by the instruction, "
        "rather than stopping early or approaching a different ball. "
        f"Instruction: {instruction}\nAnswer exactly yes or no.")


def checked_adapter(path, task):
    report = json.loads((Path(path) / "training_report.json").read_text(encoding="utf-8"))
    if (report.get("status"), report.get("task"), report.get("model"), report.get("revision")) != ("completed", task, MODEL, REVISION):
        raise ValueError("Incompatible completed visual adapter")
    if digest(Path(path) / "adapter_model.safetensors") != report["adapter_sha256"]:
        raise ValueError("Adapter weights hash mismatch")
    return report


def candidates_for(directory, row):
    image = directory / row["image"]
    path = image.with_name(image.name.replace("-candidates.png", "-observation.json"))
    return json.loads(path.read_text(encoding="utf-8"))["candidates"]


def reward_examples(rows, directory):
    """Counterfactual proposals are labeled independently, never used as agent answers."""
    examples = []
    for row in rows:
        actions = [row["expected"]]
        actions += [{"status": "visible", "color": c["color"], "id": c["id"]}
                    for c in candidates_for(directory, row)]
        actions += [{"status": s, "color": None, "id": None}
                    for s in ("partial", "not_visible", "stop", "unsupported")]
        unique = {json.dumps(action, sort_keys=True): action for action in actions}
        positive = dict(row, prompt=reward_prompt(row["instruction"], row["expected"]), completion="yes", reward_label=1)
        examples.append(positive)
        negatives = [a for a in unique.values() if a != row["expected"]]
        random.Random(row["id"]).shuffle(negatives)
        for index, action in enumerate(negatives[:2]):
            examples.append(dict(row, id=f"{row['id']}-negative-{index}", prompt=reward_prompt(row["instruction"], action),
                completion="no", reward_label=0, proposed_action=action))
    return examples


def trajectory_reward_examples(directory, manifest):
    """Physical success positives and measured early-truncation negatives, train only."""
    examples = []
    for episode in manifest["episodes"]:
        if episode["split"] != "train" or not episode.get("expert", {}).get("verified_success"):
            continue
        base = directory / "episodes" / f"train-{episode['seed']}-{episode['scenario']}"
        scene = json.loads((base / "scene.json").read_text(encoding="utf-8"))
        target = scene["objects"][episode["expert"]["target"]]
        cn = {"red": "红", "blue": "蓝", "green": "绿"}[target["color"]]
        before = base / "frame-000-candidates.png"
        final = base / f"frame-{len(episode['frames']) - 1:03d}-candidates.png"
        row = {"id": f"task-{episode['seed']}", "scene_seed": episode["seed"], "split": "train",
            "image": str(final.relative_to(directory)), "image_sha256": digest(final),
            "history_image": str(before.relative_to(directory)), "history_sha256": digest(before),
            "instruction": f"走到{cn}色球旁边", "prompt": outcome_prompt(f"走到{cn}色球旁边"),
            "completion": "yes", "reward_label": 1, "reward_scope": "task", "expected": {}}
        examples.append(row)
        trace = [json.loads(line) for line in (base / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(episode["frames"]) > 2:
            mid_time = episode["frames"][1]["sim_time"]
            nearest = min(trace, key=lambda state: abs(state["sim_time"] - mid_time))
            if nearest["target_distance_m"] > 0.35:
                middle = base / "frame-001-candidates.png"
                examples.append(dict(row, id=row["id"] + "-early", image=str(middle.relative_to(directory)),
                    image_sha256=digest(middle), completion="no", reward_label=0))
    return examples


def load_model():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError("Download the fixed Qwen3-VL model first; see Windows vision instructions")
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded QLoRA experiment requires CUDA")
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL_PATH),
        quantization_config=config, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), size={"shortest_edge": 65536, "longest_edge": 131072})
    return model.eval(), processor


def encoded(processor, image, prompt, completion=None, limit=768):
    import torch
    from PIL import Image

    paths = image if isinstance(image, list) else [image]
    messages = [{"role": "user", "content": [{"type": "image"} for _ in paths] + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(path)).convert("RGB") for path in paths]
        inputs = processor(text=[text], images=images, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    prefix = inputs["input_ids"].shape[1]
    if completion is not None:
        suffix = processor.tokenizer.encode(completion, add_special_tokens=False) + [processor.tokenizer.eos_token_id]
        tail = torch.tensor([suffix])
        inputs["input_ids"] = torch.cat((inputs["input_ids"], tail), dim=1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(tail)), dim=1)
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(tail)), dim=1)
        labels = inputs["input_ids"].clone()
        labels[:, :prefix] = -100
        inputs["labels"] = labels
        assert (labels[:, 1:] != -100).sum().item() == len(suffix)
    if inputs["input_ids"].shape[1] > limit:
        raise ValueError("Image/text sequence exceeds limit; refusing silent truncation")
    if not (inputs["input_ids"] == processor.image_token_id).any():
        raise ValueError("Missing real image tokens")
    return inputs


class TorchVisionPlanner:
    def __init__(self, adapter=None):
        self.adapter = None
        self.model, self.processor = load_model()
        if adapter:
            from peft import PeftModel

            report = checked_adapter(adapter, "actor")
            self.adapter = {"path": str(adapter), "sha256": report["adapter_sha256"], "backend": "torch"}
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).eval()

    def generate(self, image, prompt, temperature=0.0):
        import torch

        inputs = encoded(self.processor, image, prompt).to(self.model.device)
        options = {"temperature": temperature} if temperature else {}
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=96, do_sample=bool(temperature), use_cache=True, **options)
        raw = self.processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return raw, time.perf_counter() - started

    def predict(self, instruction, image, candidates, temperature=0.0):
        raw, elapsed = self.generate(image, visual_prompt(instruction), temperature)
        result = {"instruction": instruction, "raw": raw, "latency_s": elapsed, "model": MODEL, "revision": REVISION,
            "adapter": self.adapter}
        try:
            answer, choice = parse_visual_choice(raw, candidates)
        except (ValueError, TypeError, KeyError) as error:
            raise VisualOutputError(str(error), result) from error
        result.update(answer=answer, visual_choice=choice)
        return result


def reward_score(model, processor, images, prompt):
    import torch

    ids = [processor.tokenizer.encode(word, add_special_tokens=False) for word in ("no", "yes")]
    if any(len(value) != 1 for value in ids):
        raise ValueError("Reward labels must be single next tokens")
    inputs = encoded(processor, images, prompt).to(model.device)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
        return torch.softmax(logits[[value[0] for value in ids]].float(), dim=0)[1].item()


def train(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Positive finite training budget and learning rate required")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, splits = load_splits(args.data)
    if args.task == "reward":
        splits = {split: reward_examples([r for r in rows if r["frame"] == 0], args.data) for split, rows in splits.items()}
        splits["train"] += trajectory_reward_examples(args.data, manifest) * 8
    replay_rows, replay_repeat = 0, 0
    if args.extra:
        extra = [json.loads(line) for line in args.extra.read_text(encoding="utf-8").splitlines()]
        held_out = {r["scene_seed"] for split in ("valid", "test") for r in splits[split]}
        if any(r["split"] != "train" or r["scene_seed"] in held_out for r in extra):
            raise ValueError("Only independent training scenes may be replayed")
        replay_rows = len(extra)
        replay_repeat = math.ceil(len(splits["train"]) / len(extra)) if extra else 0
        splits["train"] += extra * replay_repeat
    torch.manual_seed(42)
    model, processor = load_model()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    # Frozen tied embeddings need no FP32 optimizer state; keep room for image activations.
    model.get_input_embeddings().to(dtype=torch.bfloat16)
    if args.resume:
        checked_adapter(args.resume, args.task)
        model = PeftModel.from_pretrained(model, str(args.resume), is_trainable=True)
    else:
        last = range(model.config.text_config.num_hidden_layers - 8, model.config.text_config.num_hidden_layers)
        targets = [name for name, _ in model.named_modules() if any(f"language_model.layers.{i}.self_attn." in name for i in last)
                   and name.endswith(("q_proj", "v_proj"))]
        if len(targets) != 16:
            raise ValueError("Unexpected Qwen3-VL attention layout")
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
            lora_dropout=0.0, target_modules=targets))
    model.config.text_config.use_cache = False
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not parameters or any("lora_" not in name for name in parameters):
        raise ValueError("Only LoRA parameters may be updated")
    before = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.learning_rate, weight_decay=0.0)
    cache = {}

    def batch(row):
        key = row["id"]
        if key not in cache:
            image = (args.data / row["image"]).resolve()
            if not image.is_relative_to(args.data.resolve()):
                raise ValueError("Image must remain inside dataset directory")
            if digest(image) != row["image_sha256"]:
                raise ValueError("Replay image hash mismatch")
            if row.get("history_image"):
                history = (args.data / row["history_image"]).resolve()
                if not history.is_relative_to(args.data.resolve()) or digest(history) != row["history_sha256"]:
                    raise ValueError("History image path or hash invalid")
                image = [history, image]
            prompt = row.get("prompt", visual_prompt(row["instruction"]))
            completion = row.get("completion", json.dumps(row["expected"], separators=(",", ":")))
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[key] = encoded(processor, image, prompt, completion)
        return {k: value.to(model.device) for k, value in cache[key].items()}

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    rng, order, history = random.Random(42), [], []
    model.train()
    for step in range(1, args.steps + 1):
        if not order:
            order = list(range(len(splits["train"])))
            rng.shuffle(order)
        row = splits["train"][order.pop()]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch(row), use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss; refusing to save adapter")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "row_id": row["id"], "loss": loss.item()})
        if step == 1 or step % 10 == 0:
            print(json.dumps(history[-1]), flush=True)
    elapsed = time.perf_counter() - started
    model.eval()
    valid_loss = 0.0
    with torch.inference_mode():
        for row in splits["valid"]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                valid_loss += model(**batch(row), use_cache=False).loss.item()
    changed = sum(not torch.equal(before[name], p.detach().cpu()) for name, p in parameters.items())
    if not changed:
        raise RuntimeError("No adapter tensor changed")
    model.save_pretrained(args.output)
    config_path = args.output / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report = {"status": "completed", "task": args.task, "model": MODEL, "revision": REVISION,
        "steps": args.steps, "seed": 42, "batch_size": 1, "learning_rate": args.learning_rate,
        "last_layers": 8, "rank": 8, "alpha": 16, "quantization": "NF4 double-quant; BF16 compute",
        "image_pixels": {"min": 65536, "max": 131072}, "max_seq_length": 768,
        "completion_only_loss": True, "trainable_parameters": sum(p.numel() for p in parameters.values()),
        "changed_tensors": changed, "optimizer_state_resumed": False,
        "elapsed_s": elapsed, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "valid_loss": valid_loss / len(splits["valid"]), "history": history,
        "train_rows": len(splits["train"]), "dataset_manifest_sha256": digest(args.data / "manifest.json"),
        "training_rows_sha256": hashlib.sha256(json.dumps(splits["train"], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "replay_rows": replay_rows, "replay_repeat": replay_repeat,
        "replay_sha256": digest(args.extra) if args.extra else None,
        "parent_adapter_sha256": digest(args.resume / "adapter_model.safetensors") if args.resume else None,
        "adapter_sha256": digest(args.output / "adapter_model.safetensors"),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes")}}
    (args.output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, ensure_ascii=False), flush=True)


def evaluate(args):
    _, splits = load_splits(args.data)
    planner = TorchVisionPlanner(args.adapter)
    rows = splits[args.split]
    if args.limit:
        rows = random.Random(42).sample(rows, min(args.limit, len(rows)))
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    for row in rows:
        item = {k: row[k] for k in ("id", "scene_seed", "scenario", "task_type", "expected", "instruction")}
        try:
            prediction = planner.predict(row["instruction"], args.data / row["image"], candidates_for(args.data, row))
            item.update(prediction)
            item.update(valid=True, correct=prediction["visual_choice"] == row["expected"])
        except VisualOutputError as error:
            item.update(error.prediction)
            item.update(valid=False, correct=False, error=str(error))
        results.append(item)
        print(f"{len(results)}/{len(rows)} correct={item['correct']} {row['instruction']}", flush=True)
    summary = {"count": len(results), "scene_count": len({r["scene_seed"] for r in results}),
        "correct": sum(r["correct"] for r in results), "accuracy": statistics.mean(r["correct"] for r in results),
        "valid_rate": statistics.mean(r["valid"] for r in results),
        "median_latency_s": statistics.median(r["latency_s"] for r in results),
        "split": args.split, "dataset_manifest_sha256": digest(args.data / "manifest.json"), "model": MODEL, "revision": REVISION,
        "adapter_sha256": digest(args.adapter / "adapter_model.safetensors") if args.adapter else None,
        "by_task": {kind: {"correct": sum(r["correct"] for r in results if r["task_type"] == kind),
            "total": sum(r["task_type"] == kind for r in results)} for kind in sorted({r["task_type"] for r in results})},
        "note": "Synthetic scene grounding; independent seeds; no physical success inferred from JSON accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def evaluate_reward(args):
    from peft import PeftModel

    if args.output.exists():
        raise FileExistsError(args.output)
    _, splits = load_splits(args.data)
    checked_adapter(args.adapter, "reward")
    model, processor = load_model()
    model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    rows = reward_examples([r for r in splits[args.split] if r["frame"] == 0], args.data)
    results = []
    for row in rows:
        score = reward_score(model, processor, args.data / row["image"], row["prompt"])
        results.append({"id": row["id"], "label": row["reward_label"], "score": score,
            "correct": (score >= 0.5) == bool(row["reward_label"])})
    negatives = [r for r in results if not r["label"]]
    positives = [r for r in results if r["label"]]
    summary = {"split": args.split, "count": len(results),
        "accuracy": statistics.mean(r["correct"] for r in results),
        "positive_recall": statistics.mean(r["correct"] for r in positives),
        "false_accept_rate": statistics.mean(not r["correct"] for r in negatives),
        "adapter_sha256": digest(args.adapter / "adapter_model.safetensors"),
        "note": "Action reward only; two counterfactual negatives per instruction. Not a held-out task-outcome reward evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--task", choices=["actor", "reward"], default="actor")
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--learning-rate", type=float, default=5e-5)
    training.add_argument("--resume", type=Path)
    training.add_argument("--extra", type=Path)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--adapter", type=Path)
    evaluation.add_argument("--split", choices=["valid", "test"], default="valid")
    evaluation.add_argument("--limit", type=int)
    reward_evaluation = subparsers.add_parser("evaluate-reward")
    reward_evaluation.add_argument("--data", type=Path, required=True)
    reward_evaluation.add_argument("--output", type=Path, required=True)
    reward_evaluation.add_argument("--adapter", type=Path, required=True)
    reward_evaluation.add_argument("--split", choices=["valid", "test"], default="valid")
    args = parser.parse_args()
    return {"train": train, "evaluate": evaluate, "evaluate-reward": evaluate_reward}[args.command](args)


if __name__ == "__main__":
    main()
