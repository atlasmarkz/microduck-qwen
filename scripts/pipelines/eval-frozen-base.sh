#!/usr/bin/env bash
# Frozen Qwen3-VL-4B (no LoRA) with the full skill harness, on the never-used seeds 99200/99300. Same arguments and flags as
# mapoff-v2 (scripts/eval_world.sh, map escape and legality off); only the adapter differs. Measures what LoRA training adds
# on top of the harness:
#   python -m microduck_qwen.truth_score runs/lang/frozen-99200 runs/lang/frozen-99300 --b runs/lang/mapoff-v2-99200 runs/lang/mapoff-v2-99300
set -u
export MICRODUCK_MAP_ESCAPE=0 MICRODUCK_LEGALITY=0
for seed in 99200 99300; do
  .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v2 \
    --output "runs/lang/frozen-$seed" --rollout --scene-mix world --world-level 2 --seed "$seed" \
    --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931
done
echo "FROZEN BASE EVALUATION DONE"
