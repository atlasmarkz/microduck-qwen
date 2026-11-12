#!/usr/bin/env bash
# Frozen Qwen3-VL-4B (no LoRA) with the termination rules on (microduck_qwen/legality.py, MICRODUCK_TERMINATION=1), same
# arguments as scripts/pipelines/eval-frozen-base.sh. Asks whether the adapter's gain over the frozen base (run `frozen`) is a
# termination behaviour that a contract rule can supply without training. Compare:
#   python -m microduck_qwen.truth_score runs/lang/frozterm-99200 runs/lang/frozterm-99300 --b runs/lang/frozen-99200 runs/lang/frozen-99300
#   python -m microduck_qwen.truth_score runs/lang/frozterm-99200 runs/lang/frozterm-99300 --b runs/lang/mapoff-v2-99200 runs/lang/mapoff-v2-99300
set -u
export MICRODUCK_MAP_ESCAPE=0 MICRODUCK_LEGALITY=0 MICRODUCK_TERMINATION=1
for seed in 99200 99300; do
  .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v2 \
    --output "runs/lang/frozterm-$seed" --rollout --scene-mix world --world-level 2 --seed "$seed" \
    --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931
done
echo "FROZEN TERMINATION EVALUATION DONE"
