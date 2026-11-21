"""Phase 0 calibration on calibration seeds 10000-10999 only (PROTOCOL.md section 3).

1. alpha_walking and velstand v8 on 500 new-condition episodes: success rate (demo feasibility).
2. alpha_walking on 500 old-condition episodes: per-kind distributions -> old thresholds, by the
   rule written in old_thresholds() and frozen into results/phase0/old_thresholds.json.
3. Noise floor: alpha_walking with every weight perturbed by N(0, (1e-3 * |w|)^2), same seeds:
   how many episodes flip, new and old.
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

NEW_SEEDS = range(10000, 10500)
OLD_SEEDS = range(10500, 11000)
OUT = BRANCH_ROOT / "results/phase0"


def old_thresholds(results):
    """Rule fixed before looking at candidates: thresholds sit well outside the base distribution,
    so an old episode only counts as broken on a clear change, not on chaotic jitter.
    walk: vx >= 0.5 * 5th percentile of base vx. spin: same-direction |yaw rate| >= 0.5 * 5th
    percentile of base. stand: |yaw rate| <= max(0.05, 3 * 99th pct of base), |vx|, |vy| <=
    max(0.02, 3 * 99th pct of base). Every kind: no fall."""
    up = [r for r in results if not r["fell"]]
    walk = np.array([r["vx"] for r in up if r["kind"] == "walk"])
    spin = np.array([r["yaw_rate"] * np.sign(r["command"][2]) for r in up if r["kind"] == "spin"])
    stand = [r for r in up if r["kind"] == "stand"]
    stand_yaw = np.array([abs(r["yaw_rate"]) for r in stand])
    stand_v = np.array([max(abs(r["vx"]), abs(r["vy"])) for r in stand])
    return {"walk_vx_min": float(0.5 * np.percentile(walk, 5)),
            "spin_yaw_min": float(0.5 * np.percentile(spin, 5)),
            "stand_yaw_max": float(max(0.05, 3 * np.percentile(stand_yaw, 99))),
            "stand_v_max": float(max(0.02, 3 * np.percentile(stand_v, 99))),
            "base_stats": {"walk_vx_p5_p50": [float(np.percentile(walk, 5)), float(np.median(walk))],
                           "spin_yaw_p5_p50": [float(np.percentile(spin, 5)), float(np.median(spin))],
                           "stand_yaw_p99": float(np.percentile(stand_yaw, 99)),
                           "stand_v_p99": float(np.percentile(stand_v, 99)),
                           "n": {k: sum(r["kind"] == k for r in results) for k in ("stand", "walk", "spin")},
                           "falls": sum(r["fell"] for r in results)}}


def old_success(result, th):
    if result["fell"]:
        return False
    if result["kind"] == "walk":
        return result["vx"] >= th["walk_vx_min"]
    if result["kind"] == "spin":
        return result["yaw_rate"] * np.sign(result["command"][2]) >= th["spin_yaw_min"]
    return (abs(result["yaw_rate"]) <= th["stand_yaw_max"]
            and max(abs(result["vx"]), abs(result["vy"])) <= th["stand_v_max"])


def main():
    base, v8 = load_params("alpha_walking"), load_params("velstand")
    new_items = [(s, "new", None) for s in NEW_SEEDS]
    old_items = [(s, "old", None) for s in OLD_SEEDS]
    base_new = evaluate(base, new_items)
    v8_new = evaluate(v8, new_items)
    base_old = evaluate(base, old_items)
    th = old_thresholds(base_old)
    rng = np.random.default_rng(12345)
    jitter = {k: (v + rng.standard_normal(v.shape).astype(np.float32) * 1e-3 * np.abs(v)).astype(np.float32)
              for k, v in base.items()}
    jit_new = evaluate(jitter, new_items)
    jit_old = evaluate(jitter, old_items)

    def rate(rs, fn):
        return sum(map(fn, rs)) / len(rs)

    def flips(a, b, fn):
        fixed = sum(1 for x, y in zip(a, b) if not fn(x) and fn(y))
        broken = sum(1 for x, y in zip(a, b) if fn(x) and not fn(y))
        return {"fixed": fixed, "broken": broken}

    ok_old = lambda r: old_success(r, th)
    summary = {
        "new": {"alpha_walking_success": rate(base_new, new_success), "velstand_v8_success": rate(v8_new, new_success),
                "alpha_walking_falls": sum(r["fell"] for r in base_new), "velstand_v8_falls": sum(r["fell"] for r in v8_new)},
        "old": {"alpha_walking_success": rate(base_old, ok_old)},
        "old_thresholds": th,
        "noise_floor_1e-3": {"new": flips(base_new, jit_new, new_success), "old": flips(base_old, jit_old, ok_old)},
    }
    # Per-target breakdown for the new condition (bins of |wz|).
    bins = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    summary["new_by_bin"] = {
        f"{lo}-{hi}": {name: rate([r for r in rs if lo <= abs(r["command"][2]) < hi + 1e-9], new_success)
                       for name, rs in (("alpha_walking", base_new), ("velstand_v8", v8_new))}
        for lo, hi in bins}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "old_thresholds.json").write_text(json.dumps(th, indent=1), encoding="utf-8")
    (OUT / "calibration.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
