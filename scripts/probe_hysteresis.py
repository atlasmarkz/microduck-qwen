"""Phase 0 diagnostic: is the dead zone a start-up threshold? Calibration seeds 10000-10004.

Each case runs the command from standstill, or after a 1.5 s lead-in with a strong command of the
same kind, or without observation noise. Writes results/phase0/hysteresis.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_lab import BRANCH_ROOT  # noqa: E402
from gait_lab.policy import forward, load_params  # noqa: E402
from gait_lab.sim import rollout  # noqa: E402

CASES = [
    ("vx0.2 from standstill", (0.2, 0, 0), None, True),
    ("vx0.2 after 1.5 s of vx0.4", (0.2, 0, 0), ((0.4, 0, 0), 1.5), True),
    ("vx0.2 no obs noise", (0.2, 0, 0), None, False),
    ("wz+0.6 from standstill", (0, 0, 0.6), None, True),
    ("wz+0.6 after 1.5 s of wz+1.5", (0, 0, 0.6), ((0, 0, 1.5), 1.5), True),
    ("wz-0.6 after 1.5 s of wz-1.5", (0, 0, -0.6), ((0, 0, -1.5), 1.5), True),
    ("wz+0.3 after 1.5 s of wz+1.5", (0, 0, 0.3), ((0, 0, 1.5), 1.5), True),
    ("wz+1.0 after 1.5 s of wz+1.5", (0, 0, 1.0), ((0, 0, 1.5), 1.5), True),
    ("wz+0.6 no obs noise", (0, 0, 0.6), None, False),
]
SEEDS = range(10000, 10005)


def main():
    out = {}
    for name in ("alpha_walking", "velstand"):
        params = load_params(name)
        rows = []
        print("==", name)
        for label, cmd, lead, noise in CASES:
            rs = [rollout(lambda o: forward(params, o), s, "old", command=cmd, lead=lead, noise=noise) for s in SEEDS]
            up = [r for r in rs if not r["fell"]]
            row = {"case": label, "falls": len(rs) - len(up), "n": len(rs),
                   "yaw_rate_mean": float(np.mean([r["yaw_rate"] for r in up])),
                   "vx_mean": float(np.mean([r["vx"] for r in up]))}
            rows.append(row)
            print(f"  {label:32s} falls {row['falls']}/{row['n']}  yaw {row['yaw_rate_mean']:+.3f}  vx {row['vx_mean']:+.3f}")
        out[name] = rows
    path = BRANCH_ROOT / "results/phase0/hysteresis.json"
    path.write_text(json.dumps({"seeds": [SEEDS.start, SEEDS.stop], "policies": out}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
