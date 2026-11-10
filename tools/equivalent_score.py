"""Strict action-label score next to the outcome-equivalent score (`done` for an expected `stop` after clean motion counts).

The strict score stays the headline; this only shows how many misses are the terminal stop/done confusion.

    PYTHONPATH=. python tools/equivalent_score.py runs/lang/merged-v2-99000 runs/lang/merged-v2-99100
"""

import json
from pathlib import Path
import sys

from microduck_qwen.compare_runs import is_invalid
from tools.failure_taxonomy import done_for_stop


def score(directories):
    valid = strict = equivalent = 0
    for directory in directories:
        for result in json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))["results"]:
            if is_invalid(result):
                continue
            valid += 1
            strict += bool(result["correct"])
            equivalent += bool(result["correct"]) or done_for_stop(result)
    return strict, equivalent, valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    strict, equivalent, valid = score(sys.argv[1:])
    print(f"strict {strict}/{valid} ({strict / valid:.1%}); outcome-equivalent {equivalent}/{valid} ({equivalent / valid:.1%})")
