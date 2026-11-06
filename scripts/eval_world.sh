#!/usr/bin/env bash
# Held-out rollout evaluation in the 2x2 multi-room world (level 2).
# One seed group = 8 independent layouts x (left-ball task, "move forward 20 cm" task, 3 explore-and-approach tasks).
# Two groups (the default) give the 80 tasks the results in experiments/lang-v1/summary.md are reported on.
#
#   bash scripts/eval_world.sh ADAPTER DATA NAME [SEED ...]       # run from the repository root
#   bash scripts/eval_world.sh adapters/lang-v2 data/lang-v2 eval  # -> runs/lang/eval-98000, runs/lang/eval-98100
#
# Compare two runs with:  python -m microduck_qwen.compare_runs runs/lang/A-98000 runs/lang/A-98100 --b runs/lang/B-98000 runs/lang/B-98100
set -e
adapter=$1; data=$2; name=$3; shift 3
seeds=${*:-98000 98100}
for seed in $seeds; do
  .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data "$data" --adapter "$adapter" \
    --output "runs/lang/$name-$seed" --rollout --scene-mix world --world-level 2 --seed "$seed" \
    --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931
done
