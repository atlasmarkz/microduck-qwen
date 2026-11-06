# scripts

Run everything from the repository root (paths such as `.venv/Scripts/python` and `runs/` are relative to it).

| File | What it does |
|---|---|
| `eval_world.sh ADAPTER DATA NAME [SEED ...]` | held-out rollout evaluation in the 2x2 multi-room world (80 tasks with the default two seed groups) |
| `record_demo_world.sh [ADAPTER] [SEED]` | records the exploration demo GIF, retrying until the run is a complete success |
| `pipelines/lang-v1.sh` | corrected-data aggregation + retrain from `self-v4-r2` -> `adapters/lang-v1`, then base vs lang-v1 evaluation |
| `pipelines/lang-v2.sh` | corrected-data aggregation with the `lang-v1` actor -> `adapters/lang-v2` (the current default adapter) |
| `pipelines/lang-v3.sh` | move-only phrasing data on top of `lang-v2` -> `adapters/lang-v3` (not adopted, see the experiment notes) |
| `pipelines/explore-noise-diag.sh` | repeats the same adapter on the same seeds to measure run-to-run noise, plus a control retrain |

Results and the reasoning behind each step are in [experiments/lang-v1/summary.md](../experiments/lang-v1/summary.md).
