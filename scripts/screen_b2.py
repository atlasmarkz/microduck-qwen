"""Go/no-go screen B2 (PROTOCOL.md "第 0 阶段"): scale the first-layer weight column of the yaw
command input (obs index 50) by k. Calibration seeds 11000-11999 only, never check or test seeds.

If any k meets the success criterion of section 6 (new success +50 points, old change >= -1 point)
the editor sessions are not run. Writes results/phase0/screen_b2.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_lab import BRANCH_ROOT  # noqa: E402
from gait_lab.policy import load_params  # noqa: E402
from gait_lab.runner import evaluate  # noqa: E402
from gait_lab.sim import new_success  # noqa: E402
from calibrate import old_success  # noqa: E402

WZ_INDEX = 50  # 3 ang_vel + 3 gravity + 14 pos + 14 vel + 14 last action = 48; twist = [48, 49, 50]
FACTORS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
NEW_SEEDS = range(11000, 11500)
OLD_SEEDS = range(11500, 12000)


def main():
    base = load_params("alpha_walking")
    th = json.loads((BRANCH_ROOT / "results/phase0/old_thresholds.json").read_text(encoding="utf-8"))
    items = [(s, "new", None) for s in NEW_SEEDS] + [(s, "old", None) for s in OLD_SEEDS]
    n_new = len(NEW_SEEDS)
    rows, ref = [], None
    for k in FACTORS:
        params = dict(base)
        params["mlp.0.weight"] = base["mlp.0.weight"].copy()
        params["mlp.0.weight"][:, WZ_INDEX] *= k
        res = evaluate(params, items)
        new, old = res[:n_new], res[n_new:]
        ok_new = [bool(new_success(r)) for r in new]
        ok_old = [bool(old_success(r, th)) for r in old]
        if ref is None:
            ref = (ok_new, ok_old)
        row = {"k": k, "new_success": sum(ok_new) / n_new, "old_success": sum(ok_old) / len(old),
               "new_falls": sum(r["fell"] for r in new), "old_falls": sum(r["fell"] for r in old),
               "S_new": sum(b and not a for a, b in zip(ref[0], ok_new)) - sum(a and not b for a, b in zip(ref[0], ok_new)),
               "S_old": sum(b and not a for a, b in zip(ref[1], ok_old)) - sum(a and not b for a, b in zip(ref[1], ok_old)),
               "old_broken_by_kind": {kind: sum(1 for r, a, b in zip(old, ref[1], ok_old) if r["kind"] == kind and a and not b)
                                      for kind in ("stand", "walk", "spin")}}
        # Tracking ratio achieved/commanded (the success criteria are one-sided: overshoot passes).
        ratio_new = [r["yaw_rate"] / r["command"][2] for r in new if not r["fell"]]
        ratio_spin = [r["yaw_rate"] / r["command"][2] for r in old if not r["fell"] and r["kind"] == "spin"]
        walk_vx = [r["vx"] for r in old if not r["fell"] and r["kind"] == "walk"]
        row["new_ratio_p5_p50_p95"] = [float(x) for x in np.percentile(ratio_new, (5, 50, 95))]
        row["spin_ratio_p5_p50_p95"] = [float(x) for x in np.percentile(ratio_spin, (5, 50, 95))]
        row["walk_vx_p50"] = float(np.median(walk_vx))
        rows.append(row)
        print(json.dumps(row))
    base_new, base_old = rows[0]["new_success"], rows[0]["old_success"]
    passed = [r["k"] for r in rows if r["new_success"] - base_new >= 0.5 and r["old_success"] - base_old >= -0.01]
    out = {"seeds": {"new": [NEW_SEEDS.start, NEW_SEEDS.stop], "old": [OLD_SEEDS.start, OLD_SEEDS.stop]},
           "rows": rows, "meets_section6": passed}
    (BRANCH_ROOT / "results/phase0/screen_b2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("meets section 6:", passed or "none")


if __name__ == "__main__":
    main()
