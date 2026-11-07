"""Paired comparison of two rollout evaluations over the same tasks.

Two runs of the same 80 held-out tasks are not two independent samples: most tasks are solved (or failed) by
both, and only the *discordant* tasks carry information about the difference. Comparing totals (69 vs 66)
ignores that and, with run-to-run noise of about +/-4 per 40 tasks, cannot tell a real change from luck.
This reports the discordant counts, an exact two-sided McNemar (sign) test on them, and a paired bootstrap
interval for the difference in success rate, optionally split by task family.

Reading list for the evaluation side (abstract-level): exact tests on paired binary outcomes are standard
(McNemar, 1947); the project's Wilson intervals (`agent_eval.wilson_interval`) describe one run, not a difference.
"""

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path
import random


def is_invalid(result):
    """A task the harness could not label (the teacher's "leftmost" target was ambiguous or not confidently
    visible): not a model outcome."""
    error = str(result.get("error"))
    return bool(result.get("invalid_task")) or ("teacher" in error and ("ambiguous" in error or "not confidently visible" in error))


def load(run_dir):
    """{(seed, task_id): correct} from one run directory's report.json, without invalid tasks."""
    report = json.loads((Path(run_dir) / "report.json").read_text(encoding="utf-8"))
    return {(r["seed"], r["task_id"]): bool(r["correct"]) for r in report["results"] if not is_invalid(r)}


def family(task_id):
    return "explore" if task_id >= 100 else f"task{task_id}"


def mcnemar_exact(only_a, only_b):
    """Two-sided exact p-value for discordant counts (a solved, b failed) vs (b solved, a failed)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a, b, resamples=5000, seed=0):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("the two runs share no (seed, task) pairs")
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    diffs = [int(b[k]) - int(a[k]) for k in keys]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples))
    by_family = defaultdict(lambda: [0, 0, 0])
    for k in keys:
        row = by_family[family(k[1])]
        row[0] += a[k]
        row[1] += b[k]
        row[2] += 1
    return {"pairs": len(keys), "a_correct": sum(a[k] for k in keys), "b_correct": sum(b[k] for k in keys),
            "only_a": only_a, "only_b": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b),
            "diff_rate": sum(diffs) / len(diffs), "diff_ci95": [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
            "by_family": {k: {"a": v[0], "b": v[1], "n": v[2]} for k, v in sorted(by_family.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, nargs="+", help="run directory(ies) of the baseline, e.g. runs/lang/eff-98000 runs/lang/eff-98100")
    parser.add_argument("--b", type=Path, nargs="+", required=True, help="run directory(ies) of the candidate")
    args = parser.parse_args()
    a, b = {}, {}
    for directory in args.a:
        a.update(load(directory))
    for directory in args.b:
        b.update(load(directory))
    result = compare(a, b)
    print(json.dumps(result, indent=2))
    print(f"\nA {result['a_correct']}/{result['pairs']}  B {result['b_correct']}/{result['pairs']}  "
          f"B-only {result['only_b']}  A-only {result['only_a']}  McNemar p={result['mcnemar_p']:.3f}  "
          f"diff {result['diff_rate']:+.3f} (95% {result['diff_ci95'][0]:+.3f}..{result['diff_ci95'][1]:+.3f})")


if __name__ == "__main__":
    main()
