"""Follow-up screens after B2 met section 6 (calibration seeds 12000-12499 only).

A. In-place turning from standstill (no lead-in), |wz| U(0.3, 1.0): yaw column x1 / x2.
B. Forward start from standstill, vx in {0.1, 0.15, 0.2, 0.25}: vx column (obs index 48) x1 / x1.5 / x2 / x3.
Writes results/phase0/screen_followups.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_lab import BRANCH_ROOT  # noqa: E402
from gait_lab.policy import load_params  # noqa: E402
from gait_lab.runner import evaluate  # noqa: E402
from gait_lab.sim import episode_spec, new_success  # noqa: E402

SEEDS = range(12000, 12500)


def scaled(base, index, k):
    p = dict(base)
    p["mlp.0.weight"] = base["mlp.0.weight"].copy()
    p["mlp.0.weight"][:, index] *= k
    return p


def main():
    base = load_params("alpha_walking")
    out = {"seeds": [SEEDS.start, SEEDS.stop], "standstill_turn": [], "forward_start": []}
    # A: same targets as the new condition, but no lead-in. rollout(lead=None) is not exposed by
    # evaluate(), so pass the drawn target explicitly and reuse the old-condition path (no lead).
    items = [(s, "old", episode_spec(s, "new")[1]["command"]) for s in SEEDS]
    for k in (1.0, 2.0):
        res = evaluate(scaled(base, 50, k), items)
        ok = [bool(new_success(r)) for r in res]
        ratio = [r["yaw_rate"] / r["command"][2] for r in res if not r["fell"]]
        row = {"k": k, "success": sum(ok) / len(ok), "falls": sum(r["fell"] for r in res),
               "ratio_p5_p50_p95": [float(x) for x in np.percentile(ratio, (5, 50, 95))]}
        out["standstill_turn"].append(row)
        print("standstill turn", row)
    for k in (1.0, 1.5, 2.0, 3.0):
        p = scaled(base, 48, k)
        for vx in (0.1, 0.15, 0.2, 0.25):
            res = evaluate(p, [(s, "old", (vx, 0.0, 0.0)) for s in SEEDS[:100]])
            up = [r for r in res if not r["fell"]]
            row = {"k": k, "vx_cmd": vx, "falls": len(res) - len(up),
                   "vx_p50": float(np.median([r["vx"] for r in up])) if up else None,
                   "moving_frac": sum(r["vx"] > 0.02 for r in up) / len(res)}
            out["forward_start"].append(row)
            print("forward start", row)
    (BRANCH_ROOT / "results/phase0/screen_followups.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
