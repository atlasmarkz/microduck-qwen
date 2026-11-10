"""Windows CUDA Qwen3-VL skill-agent SFT and binary action rewards with 4-bit LoRA."""

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
from .sim_data import digest

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_PATH = ROOT / ".cache/vlm-torch-hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION

ALTERNATIVE_SKILLS = [
    {"action": "move", "direction": "forward", "distance_m": 0.2},
    {"action": "move", "direction": "backward", "distance_m": 0.2},
    {"action": "turn", "direction": "left", "degrees": 45},
    {"action": "turn", "direction": "right", "degrees": 45},
    {"action": "search", "color": "red"},
    {"action": "search", "color": "blue"},
    {"action": "search", "color": "green"},
    {"action": "return_home"},
    {"action": "stop"},
    {"action": "done"},
    {"action": "clarify"},
    {"action": "unsupported"},
    {"action": "bypass_return", "side": "auto"},
]


REWARD_PROMPT_VERSION = 2


def reward_prompt(instruction, action, memory=None):
    from .agent import checked_memory
    observable = json.dumps(checked_memory(memory), ensure_ascii=False, separators=(",", ":")) if memory is not None else "not supplied"
    return ("Judge whether this proposed robot skill correctly follows the instruction and visible image. "
        "Partial, absent or ambiguous targets must not be approached. Stop and unsupported instructions need their corresponding status. "
        "Use completed actions to judge the NEXT step; done is correct only when all requested steps are completed. "
        f"Observable memory: {observable}\n"
        f"Instruction: {instruction}\nProposed skill: {json.dumps(action, separators=(',', ':'))}\n"
        "Answer exactly yes or no.")


def row_prompt(row):
    """Single-image agent prompt from a row's instruction and observable memory."""
    if row.get("prompt"):
        return row["prompt"]
    from .agent import agent_prompt
    return agent_prompt(row["instruction"], row["memory"])


def agent_reward_examples(rows):
    """Verified answers plus counterfactual/actual wrong proposals for action rewards."""
    from .skills import validate_action
    examples = []
    for row in rows:
        expected = row["expected"]
        examples.append(dict(row, prompt=reward_prompt(row["instruction"], expected, row.get("memory")), completion="yes", reward_label=1))
        negatives = [action for action in ALTERNATIVE_SKILLS if action != expected]
        random.Random(row["id"]).shuffle(negatives)
        actual = [action for action in row.get("reward_proposals", []) if action != expected]
        seen = set()
        for index, action in enumerate(actual + negatives[:2]):
            validate_action(action)
            key = json.dumps(action, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            examples.append(dict(row, id=f"{row['id']}-negative-{index}", prompt=reward_prompt(row["instruction"], action, row.get("memory")),
                completion="no", reward_label=0, proposed_action=action))
    return examples


def checked_adapter(path, task):
    report = json.loads((Path(path) / "training_report.json").read_text(encoding="utf-8"))
    if (report.get("status"), report.get("task"), report.get("model"), report.get("revision")) != ("completed", task, MODEL, REVISION):
        raise ValueError(f"Incompatible completed {task} adapter: expected {MODEL}@{REVISION}; "
                         f"got {report.get('model')}@{report.get('revision')}")
    if digest(Path(path) / "adapter_model.safetensors") != report["adapter_sha256"]:
        raise ValueError("Adapter weights hash mismatch")
    if task == "reward" and report.get("reward_prompt_version") != REWARD_PROMPT_VERSION:
        raise ValueError("Reward adapter must be retrained with observable memory (prompt v2)")
    return report


def actor_rows_with_reward_replay(rows):
    """One extra copy of reward-accepted AND expert/physics-verified training rows."""
    return rows + [row for row in rows if row.get("reward_verified_accepted") is True
                   and row.get("verified") is True and row.get("split") == "train"]


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


def encoded(processor, image, prompt, completion=None, limit=1024):
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


def generate_text(model, processor, image, prompt, max_new_tokens=110, temperature=0.0):
    import torch

    inputs = encoded(processor, image, prompt).to(model.device)
    options = {"temperature": temperature} if temperature else {}
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=bool(temperature), use_cache=True, **options)
    raw = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return raw, time.perf_counter() - started


def with_prefix(inputs, processor, text):
    """Prompt inputs with `text` appended after the generation prompt, as if the model had already written it."""
    import torch

    tail = torch.tensor([processor.tokenizer.encode(text, add_special_tokens=False)])
    out = dict(inputs)
    out["input_ids"] = torch.cat((inputs["input_ids"], tail), dim=1)
    out["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(tail)), dim=1)
    if "mm_token_type_ids" in inputs:
        out["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(tail)), dim=1)
    return out, tail.shape[1]


def prefix_logprobs(model, processor, image, prompt, prefixes):
    """{prefix: summed log-probability the model assigns to writing that prefix first}; one forward pass per prefix."""
    import torch

    base = encoded(processor, image, prompt)
    scores = {}
    for text in prefixes:
        inputs, length = with_prefix(base, processor, text)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=length + 1).logits[0, :-1]
        targets = inputs["input_ids"][0, -length:]
        scores[text] = torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).sum().item()
    return scores


def generate_with_prefix(model, processor, image, prompt, text, max_new_tokens=110):
    """Greedy continuation after a forced output prefix; returns the full raw output including the prefix."""
    import torch

    inputs, _ = with_prefix(encoded(processor, image, prompt), processor, text)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return text + processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class TorchAgentPlanner:
    """Windows counterpart of agent.AgentPlanner: one bounded skill JSON per observation."""

    def __init__(self, adapter=None):
        self.adapter = None
        self.model, self.processor = load_model()
        if adapter:
            from peft import PeftModel

            report = checked_adapter(adapter, "agent")
            self.adapter = {"path": str(adapter), "sha256": report["adapter_sha256"], "backend": "torch"}
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).eval()

    def predict(self, instruction, image, memory, temperature=0.0):
        from .skills import parse_action

        prompt = row_prompt({"instruction": instruction, "memory": memory})
        raw, elapsed = generate_text(self.model, self.processor, image, prompt, temperature=temperature)
        result = {"instruction": instruction, "raw": raw, "latency_s": elapsed, "model": MODEL,
            "revision": REVISION, "adapter": self.adapter}
        try:
            result["action"] = parse_action(raw)
        except (ValueError, TypeError, KeyError) as error:
            result["error"] = str(error)
        from . import legality
        if legality.ENABLED and not temperature:
            def score(names):
                values = prefix_logprobs(self.model, self.processor, image, prompt, [legality.prefix(n) for n in names])
                return {name: values[legality.prefix(name)] for name in names}

            def regenerate(name):
                return generate_with_prefix(self.model, self.processor, image, prompt, legality.prefix(name))

            started = time.perf_counter()
            result = legality.enforce(result, instruction, memory, score, regenerate)
            if "legality" in result:
                result["legality"]["latency_s"] = time.perf_counter() - started
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


class RewardGuidedPlanner(TorchAgentPlanner):
    """Bounded proposals ranked by reward; the collector still verifies each action."""

    def __init__(self, adapter, reward_adapter, samples=3, threshold=0.5):
        import torch
        if not 1 <= samples <= 5 or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("invalid reward sampling budget/threshold")
        report = checked_adapter(reward_adapter, "reward")
        super().__init__(adapter)
        self.model.load_adapter(str(reward_adapter), adapter_name="reward")
        self.reward_adapter = {"path": str(reward_adapter), "sha256": report["adapter_sha256"]}
        self.samples, self.threshold = samples, threshold
        torch.manual_seed(42)

    def predict(self, instruction, image, memory):
        from .skills import parse_action
        proposals, seen = [], {}
        try:
            self.model.set_adapter("default")
            for index in range(self.samples):
                raw, elapsed = generate_text(self.model, self.processor, image,
                    row_prompt({"instruction": instruction, "memory": memory}), temperature=0.7 if index else 0.0)
                entry = {"raw": raw, "latency_s": elapsed}
                try:
                    action = parse_action(raw)
                    entry["action"] = action
                    seen.setdefault(json.dumps(action, sort_keys=True), entry)
                except (ValueError, TypeError, KeyError) as error:
                    entry["error"] = str(error)
                proposals.append(entry)
            self.model.set_adapter("reward")
            for entry in seen.values():
                entry["reward_score"] = reward_score(self.model, self.processor, image,
                    reward_prompt(instruction, entry["action"], memory))
            best = max(seen.values(), key=lambda item: item["reward_score"], default=None)
            accepted = best is not None and math.isfinite(best["reward_score"]) and best["reward_score"] >= self.threshold
            return {"raw": best["raw"] if best else "", "action": best["action"] if accepted else None,
                    "reward_accepted": accepted, "reward_threshold": self.threshold,
                    "proposals": proposals, "distinct_proposals": len(seen),
                    "reward_adapter": self.reward_adapter, "adapter": self.adapter}
        finally:
            self.model.set_adapter("default")


def train(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Positive finite training budget and learning rate required")
    if args.output.exists():
        raise FileExistsError(args.output)
    from .agent_data import load_splits

    manifest, splits = load_splits(args.data)
    reward_selected_replay_rows = 0
    if args.task == "reward":
        splits = {split: agent_reward_examples(rows) for split, rows in splits.items()}
    else:
        original_count = len(splits["train"])
        splits["train"] = actor_rows_with_reward_replay(splits["train"])
        reward_selected_replay_rows = len(splits["train"]) - original_count
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
            prompt = row_prompt(row)
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
        "image_pixels": {"min": 65536, "max": 131072}, "max_seq_length": 1024,
        "completion_only_loss": True, "trainable_parameters": sum(p.numel() for p in parameters.values()),
        "changed_tensors": changed, "optimizer_state_resumed": False,
        "elapsed_s": elapsed, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "valid_loss": valid_loss / len(splits["valid"]), "history": history,
        "train_rows": len(splits["train"]), "dataset_manifest_sha256": digest(args.data / "manifest.json"),
        "training_rows_sha256": hashlib.sha256(json.dumps(splits["train"], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "replay_rows": replay_rows, "replay_repeat": replay_repeat,
        "reward_selected_replay_rows": reward_selected_replay_rows,
        "replay_sha256": digest(args.extra) if args.extra else None,
        "parent_adapter_sha256": digest(args.resume / "adapter_model.safetensors") if args.resume else None,
        "adapter_sha256": digest(args.output / "adapter_model.safetensors"),
        "reward_prompt_version": REWARD_PROMPT_VERSION if args.task == "reward" else None,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes")}}
    (args.output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, ensure_ascii=False), flush=True)


def evaluate_reward(args):
    from peft import PeftModel

    if args.output.exists():
        raise FileExistsError(args.output)
    from .agent_data import load_splits

    _, splits = load_splits(args.data)
    checked_adapter(args.adapter, "reward")
    model, processor = load_model()
    model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    rows = agent_reward_examples(splits[args.split])
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
        "note": "Skill-action reward only; two counterfactual negatives per instruction. Not a held-out task-outcome reward evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--task", choices=["agent", "reward"], default="agent")
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--learning-rate", type=float, default=5e-5)
    training.add_argument("--resume", type=Path)
    training.add_argument("--extra", type=Path)
    reward_evaluation = subparsers.add_parser("evaluate-reward")
    reward_evaluation.add_argument("--data", type=Path, required=True)
    reward_evaluation.add_argument("--output", type=Path, required=True)
    reward_evaluation.add_argument("--adapter", type=Path, required=True)
    reward_evaluation.add_argument("--split", choices=["valid", "test"], default="valid")
    args = parser.parse_args()
    return {"train": train, "evaluate-reward": evaluate_reward}[args.command](args)


if __name__ == "__main__":
    main()
