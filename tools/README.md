# tools

Diagnostics, not part of the package. Run from the repository root (`pip install -e .` makes `microduck_qwen` importable).

| File | What it does |
|---|---|
| `failure_taxonomy.py RUN_DIR ...` | counts the failures of evaluation runs by category and, for explore tasks, by the last skill and its error |
| `render_eval_map.py TASK_DIR SCENE_JSON OUT.png` | draws the depth-built map, frontiers and the true walls/balls of one evaluated task |
| `diag_explore.py SEED OUTDIR` | reproduces `explore` per hidden colour in one layout and dumps the map and true scene |
| `legality_report.py CANARY.json [--subset RUN ...]` | off/on comparison of the action-legality check from one replay canary run made with `MICRODUCK_LEGALITY=1` |
| `diag_pos.py`, `diag_stop.py`, `render_map.py` | earlier one-off diagnostics for stop positions and `diag_explore` output |
