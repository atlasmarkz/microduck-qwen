"""Off/on comparison of the action-legality check from one replay canary run made with MICRODUCK_LEGALITY=1.

A legal decision passes through `legality.enforce` unchanged, and every intervention keeps the model's original output,
so one run gives both arms: "off" is the original decision, "on" the final one. Correctness is the canary's replay rule
(action type equals the teacher's).

    python tools/legality_report.py runs/canary/legality/lang-v2-all.json [--subset eff-98000 ...]
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def arms(result):
    """(off_correct, on_correct) for one canary result row."""
    expected = result["expected"]["action"]
    on = bool(result["correct"])
    record = result.get("legality")
    if record is None:
        return on, on
    original = record.get("original_action")
    return bool(original) and original["action"] == expected, on


def report(results):
    classes = defaultdict(lambda: [0, 0, 0])
    changes = Counter()
    off_total = on_total = only_on = only_off = 0
    for result in results:
        off, on = arms(result)
        row = classes[result["state"]]
        row[0] += off
        row[1] += on
        row[2] += 1
        off_total += off
        on_total += on
        only_on += on and not off
        only_off += off and not on
        record = result.get("legality")
        if record:
            before = (record.get("original_action") or {}).get("action", "unparseable")
            after = (result.get("legality") or {}).get("chosen", "none")
            changes[(result["state"], before, after, "fixed" if on and not off else "broken" if off and not on else
                     "still right" if on else "still wrong")] += 1
    return {"total": len(results), "off": off_total, "on": on_total, "only_on": only_on, "only_off": only_off,
            "classes": {key: value for key, value in sorted(classes.items())}, "changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canary", type=Path)
    parser.add_argument("--subset", nargs="*", help="only rows whose run directory name is one of these")
    args = parser.parse_args()
    data = json.loads(args.canary.read_text(encoding="utf-8"))
    if not data.get("legality_enabled"):
        raise SystemExit("this canary was not run with MICRODUCK_LEGALITY=1")
    results = data["results"]
    if args.subset:
        results = [r for r in results if r["id"].split("/", 1)[0] in set(args.subset)]
    summary = report(results)
    print(f"decisions {summary['total']}: off {summary['off']}, on {summary['on']} "
          f"(only on {summary['only_on']}, only off {summary['only_off']})")
    for key, (off, on, total) in summary["classes"].items():
        if off != on:
            print(f"  {key:28s} {off}/{total} -> {on}/{total}")
    print("interventions (state, original, chosen, outcome):")
    for key, count in summary["changes"].most_common():
        print(f"  {count:4d} {key}")
    interventions = [r["legality"] for r in results if r.get("legality")]
    agreed = [r["top_scored_is_original"] for r in interventions if "top_scored_is_original" in r]
    if agreed:
        print(f"prefix score ranks the model's own illegal choice first in {sum(agreed)}/{len(agreed)} interventions")
    latency = sorted(r.get("latency_s", 0.0) for r in interventions)
    if latency:
        print(f"intervention latency: median {latency[len(latency) // 2]:.1f} s, max {latency[-1]:.1f} s")


if __name__ == "__main__":
    main()
