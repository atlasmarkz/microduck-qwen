"""Small user-phrase diagnostic on a saved DEVELOPMENT RGB frame; never execute."""

import argparse
import json
from pathlib import Path

from .agent import AgentPlanner
from .agent_data import load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, splits = load_splits(args.data)
    row = next(r for r in splits["valid"] if not r["memory"]["completed"])
    planner = AgentPlanner(args.adapter)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    cases = [("后退", {"action": "move", "direction": "backward", "distance_m": 0.2}),
             ("后退20厘米", {"action": "move", "direction": "backward", "distance_m": 0.2}),
             ("前进", {"action": "move", "direction": "forward", "distance_m": 0.2}),
             ("左转45度", {"action": "turn", "direction": "left", "degrees": 45}),
             ("停下来", {"action": "stop"}), ("跳舞", {"action": "unsupported"})]
    report = {"adapter": planner.adapter, "frame": str(args.data / row["image"]), "results": [],
              "note": "No execution. Six development phrases, not a benchmark or final test."}
    for instruction, expected in cases:
        prediction = planner.predict(instruction, args.data / row["image"], row["memory"])
        result = {"instruction": instruction, "expected": expected, "prediction": prediction,
                  "correct": prediction.get("action") == expected}
        report["results"].append(result)
        report.update(correct=sum(r["correct"] for r in report["results"]), total=len(report["results"]))
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
