#!/usr/bin/env bash
# Re-run the held-out tasks of single layouts (all five task types each), e.g. to check a fix on the layouts it concerns.
#   bash scripts/eval_layouts.sh ADAPTER DATA NAME SEED [SEED ...]   -> runs/lang/NAME-SEED
set -e
adapter=$1; data=$2; name=$3; shift 3
for seed in "$@"; do
  .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data "$data" --adapter "$adapter" \
    --output "runs/lang/$name-$seed" --rollout --scene-mix world --world-level 2 --seed "$seed" \
    --rollout-seeds 1 --task-ids 3 4 --explore-tasks 3 --port 7931
done
echo "LAYOUTS DONE"
