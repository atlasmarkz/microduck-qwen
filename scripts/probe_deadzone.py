"""Phase 0 probe: achieved yaw rate and forward speed vs command, alpha_walking and velstand v8.

Calibration seeds only (10000-19999, PROTOCOL.md section 3). Writes results/phase0/deadzone.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_lab import BRANCH_ROOT  # noqa: E402
from gait_lab.policy import load_params  # noqa: E402
from gait_lab.runner import evaluate  # noqa: E402

SEEDS = range(10000, 10010)
COMMANDS = ([(0.0, 0.0, s * w) for w in (0.3, 0.6, 1.0, 1.2, 1.5, 2.0) for s in (1, -1)]
            + [(0.2, 0.0, s * w) for w in (0.3, 0.6, 1.0) for s in (1, -1)]
            + [(v, 0.0, 0.0) for v in (0.0, 0.1, 0.2, 0.3, 0.4)])


def main():
    out = {"seeds": [SEEDS.start, SEEDS.stop], "policies": {}}
    for name in ("alpha_walking", "velstand"):
        params = load_params(name)
        items = [(seed, "old", cmd) for cmd in COMMANDS for seed in SEEDS]
        start = time.time()
        results = evaluate(params, items)
        rows = []
        for cmd in COMMANDS:
            rs = [r for r in results if tuple(r["command"]) == cmd]
            up = [r for r in rs if not r["fell"]]
            rows.append({"command": cmd, "n": len(rs), "falls": len(rs) - len(up),
                         "yaw_rate_mean": float(np.mean([r["yaw_rate"] for r in up])) if up else None,
                         "yaw_rate_min": float(np.min([r["yaw_rate"] for r in up])) if up else None,
                         "yaw_rate_max": float(np.max([r["yaw_rate"] for r in up])) if up else None,
                         "vx_mean": float(np.mean([r["vx"] for r in up])) if up else None})
        out["policies"][name] = {"wall_s": round(time.time() - start, 1), "rows": rows}
        print(f"== {name} ({len(items)} episodes, {time.time() - start:.1f} s)")
        for row in rows:
            fmt = lambda v: "  -  " if v is None else f"{v:+.3f}"
            print(f"  cmd vx={row['command'][0]:.1f} wz={row['command'][2]:+.1f}  falls {row['falls']}/{row['n']}"
                  f"  yaw {fmt(row['yaw_rate_mean'])} [{fmt(row['yaw_rate_min'])},{fmt(row['yaw_rate_max'])}]"
                  f"  vx {fmt(row['vx_mean'])}")
    path = BRANCH_ROOT / "results/phase0/deadzone.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
