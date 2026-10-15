"""Physical qualification of simulation skills, separate from language-model accuracy."""

import argparse
from datetime import datetime
import json

from . import ROOT
from .skills import SkillExecutor
from .vision_eval import start_scene, stop_process
from .vision_data import request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--port", type=int, default=7852)
    parser.add_argument("--backward-command", type=float, default=0.40)
    args = parser.parse_args()
    output = ROOT / "runs/skill-probe" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    process = log = executor = None
    try:
        process, log, info, failures = start_scene(output, args.seed, "none", args.port, collect_gt=True)
        executor = SkillExecutor(info, output, backward_command=args.backward_command)
        actions = [{"action": "move", "direction": "forward", "distance_m": 0.2},
                   {"action": "move", "direction": "backward", "distance_m": 0.2},
                   {"action": "turn", "direction": "left", "degrees": 45},
                   {"action": "turn", "direction": "right", "degrees": 45},
                   {"action": "move", "direction": "forward", "distance_m": 0.3},
                   {"action": "return_home"}]
        results = []
        for action in actions:
            result = executor.run(action)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            stats = request(args.port, "collection_stats")
            (output / "report.json").write_text(json.dumps({"info": info, "startup_failures": failures, "results": results,
                "backward_command": args.backward_command, "contacts": stats,
                "passed": len(results) == len(actions) and all(r["success"] for r in results) and stats["steps_with_scene_contact"] == 0}, ensure_ascii=False, indent=2) + "\n")
            if not result["success"]:
                break
    finally:
        try:
            if executor:
                executor.close()
        finally:
            if process:
                stop_process(process)
            if log:
                log.close()
    print(f"REPORT: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
