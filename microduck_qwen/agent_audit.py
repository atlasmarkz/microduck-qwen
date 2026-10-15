"""Export small honest pilot evidence; excludes weights, images and raw sensor datasets."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from .agent_eval import semantic_match
from .vision_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cycle = json.loads((args.cycle / "cycle.json").read_text())
    if cycle["status"] != "completed":
        raise ValueError("cycle not completed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "completed", "cycle_sha256": digest(args.cycle / "cycle.json"),
              "selected_adapter": cycle["selected_adapter"], "models": {}, "datasets": {},
              "limitations": ["finite task DSL and few scenes", "development-set pilot, no final test score",
                 "privileged own pose and known safety map", "expert intervention is not unrestricted DAgger",
                 "row counts can include template paraphrases; these are not independent physical trajectories",
                 "no learned reward model or online high-level RL", "MLX peak is not total system RAM; swap recorded separately"]}
    shutil.copyfile(args.cycle / "cycle.json", output / "cycle.json")
    for name in ["base", *[f"r{r['round']}" for r in cycle["rounds"]]]:
        evaluation = json.loads((args.cycle / f"{name}-valid/report.json").read_text())
        semantic = sum(semantic_match(r["prediction"]["raw"], r["expected"]) for r in evaluation["results"])
        groups = defaultdict(lambda: {"correct": 0, "semantic_correct": 0, "total": 0})
        for row in evaluation["results"]:
            item = groups[row["task_type"]]
            item["total"] += 1
            item["correct"] += row["correct"]
            item["semantic_correct"] += semantic_match(row["prediction"]["raw"], row["expected"])
        entry = {"strict_correct": evaluation["correct"], "semantic_correct_after_fence_removal": semantic,
                 "total": evaluation["total"], "per_action": dict(groups)}
        shutil.copyfile(args.cycle / f"{name}-valid/report.json", output / f"{name}-valid.json")
        if name != "base":
            round_info = next(r for r in cycle["rounds"] if f"r{r['round']}" == name)
            adapter = Path(round_info["adapter"])
            training = json.loads((adapter / "training_report.json").read_text())
            entry.update(adapter_sha256=training["adapter_sha256"], optimizer_steps=len(training["steps"]),
                         unique_training_rows=len({row for step in training["steps"] for row in step["row_ids"]}),
                         mlx_peak_bytes=training["mlx_peak_bytes"], elapsed_s=training["elapsed_s"],
                         swap_before=training["swap_before"], swap_after=training["swap_after"],
                         changed_adapter_tensors=training["changed_adapter_tensors"],
                         parent_adapter=training.get("parent_adapter"), token_lengths=training["observed_token_lengths"])
            shutil.copyfile(adapter / "training_report.json", output / f"{name}-training.json")
            rollout_path = args.cycle / f"{name}-rollout/report.json"
            if rollout_path.exists():
                rollout = json.loads(rollout_path.read_text())
                entry["rollout"] = {"correct": rollout["correct"], "total": rollout["total"], "scene_seed": rollout["scene_seed"]}
                shutil.copyfile(rollout_path, output / f"{name}-rollout.json")
                for result in rollout["results"]:
                    shutil.copyfile(result["episode"], output / f"{name}-task-{result['task_id']:02d}.json")
            data = Path(round_info["data"])
            dataset = json.loads((data / "manifest.json").read_text())
            training_rows = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
            report["datasets"][name] = {"sha256": digest(data / "manifest.json"), "splits": dataset["splits"],
                "template_paraphrase_rows": sum("language_source" in row for row in training_rows),
                "unique_training_images": len({row["image_sha256"] for row in training_rows}),
                "episodes": len(dataset["episodes"]), "failed_episodes": sum(not e["verified"] for e in dataset["episodes"]),
                "expert_interventions": sum(e["interventions"] for e in dataset["episodes"])}
            shutil.copyfile(data / "manifest.json", output / f"{name}-dataset.json")
        report["models"][name] = entry
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
