#!/usr/bin/env bash
# Paired simulator check of the executor feedback after move/turn: only the "move forward 20 cm" task (task 4),
# on layouts 98400..98731, with MICRODUCK_MOVE_FEEDBACK=0 (before) and 1 (after).
#   bash scripts/pipelines/eval-move-feedback.sh           # run from the repository root
# Compare: python -m microduck_qwen.compare_runs runs/lang/mf0-98400 ... --b runs/lang/mf1-98400 ...
set -u
adapter=adapters/lang-v2; data=data/lang-v2
for flag in 0 1; do
  for seed in 98400 98500 98600 98700; do
    MICRODUCK_MOVE_FEEDBACK=$flag .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data "$data" --adapter "$adapter" \
      --output "runs/lang/mf$flag-$seed" --rollout --scene-mix world --world-level 2 --seed "$seed" \
      --rollout-seeds 8 --task-ids 4 --explore-tasks 0 --port 7931
  done
done
echo "MOVE FEEDBACK EVALUATION DONE"
