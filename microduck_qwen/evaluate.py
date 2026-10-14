"""Held-out semantic evaluation of the baseline or actual trained adapter."""

import argparse
import collections
import hashlib
import json
import random
import statistics
import time

from . import ROOT
from .contract import compact
from .model import Planner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output")
    args = parser.parse_args()
    path = ROOT / f"data/{args.split}.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    by_category = collections.defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    rng = random.Random(42)
    for values in by_category.values():
        rng.shuffle(values)
    selected = []
    while any(by_category.values()) and len(selected) < args.limit:
        for values in by_category.values():
            if values and len(selected) < args.limit:
                selected.append(values.pop())
    planner = Planner(base=args.base, adapter=args.adapter)
    results = []
    for index, row in enumerate(selected):
        start = time.perf_counter()
        try:
            result = planner.predict(row["instruction"])
            result.update(valid=True, exact=result["plan"] == row["expected"])
        except (ValueError, TypeError, KeyError) as error:
            result = {"instruction": row["instruction"], "valid": False, "exact": False,
                      "error": str(error), "latency_s": time.perf_counter() - start}
        result.update(expected=row["expected"], category=row["category"])
        results.append(result)
        print(f"{index+1}/{len(selected)} exact={result['exact']} {row['instruction']}", flush=True)
    summary = {"count": len(results), "valid_rate": statistics.mean(r["valid"] for r in results),
        "exact_plan_rate": statistics.mean(r["exact"] for r in results),
        "median_latency_s": statistics.median(r["latency_s"] for r in results),
        "by_category": {c: {"count": sum(r["category"] == c for r in results),
            "exact_rate": statistics.mean(r["exact"] for r in results if r["category"] == c)}
            for c in sorted({r["category"] for r in results})},
        "split": args.split, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection_seed": 42, "adapter": str(planner.adapter) if planner.adapter else None,
        "note": "Synthetic held-out paraphrases; exact goal plans, not physical task success."}
    out = ROOT / (args.output or f"runs/{args.split}-{'base' if args.base else 'adapter'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n")
    print(compact(summary))


if __name__ == "__main__":
    main()
