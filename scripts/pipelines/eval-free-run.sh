#!/usr/bin/env bash
# Uncut re-run of the three harness arms (microduck_qwen/free_run.py): same 80 tasks and layouts as `frozen`, `frozterm`,
# `mapoff-v2` and `raw` (seeds 99200/99300), but the model runs to its own end instead of being cut at the first deviation
# from the teacher. Same switches as the cut runs. Scored by simulator truth:
#   python -m microduck_qwen.truth_score runs/lang/free-frozen-99200 runs/lang/free-frozen-99300 --b runs/lang/free-lora-99200 runs/lang/free-lora-99300
set -u
export MICRODUCK_MAP_ESCAPE=0 MICRODUCK_LEGALITY=0
for arm in lora frozterm frozen; do
  export MICRODUCK_TERMINATION=0
  adapter=()
  [ "$arm" = frozterm ] && export MICRODUCK_TERMINATION=1
  [ "$arm" = lora ] && adapter=(--adapter adapters/lang-v2)
  for seed in 99200 99300; do
    .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v2 "${adapter[@]}" \
      --output "runs/lang/free-$arm-$seed" --rollout --scene-mix world --world-level 2 --seed "$seed" \
      --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931 --free-run
  done
done
echo "FREE-RUN EVALUATION DONE"
