#!/usr/bin/env bash
# Uncut re-run of all four arms with a full pose log (microduck_qwen/free_run.py, state_log.py): same 80 tasks and layouts
# as `raw`, `frozen`, `frozterm` and `mapoff-v2` (seeds 99200/99300). The three harness arms run the deployed loop to its
# own end instead of being cut at the first deviation from the teacher; `raw` was never cut and is repeated only for the
# pose log. Same switches as the earlier runs. Scored by simulator truth:
#   python -m microduck_qwen.truth_score runs/lang/free-frozen-99200 runs/lang/free-frozen-99300 --b runs/lang/free-lora-99200 runs/lang/free-lora-99300
# A finished seed group is skipped, so an interrupted run continues where it stopped.
set -u
export MICRODUCK_MAP_ESCAPE=0 MICRODUCK_LEGALITY=0
for arm in lora frozterm frozen raw; do
  export MICRODUCK_TERMINATION=0
  options=(--free-run)
  [ "$arm" = frozterm ] && export MICRODUCK_TERMINATION=1
  [ "$arm" = lora ] && options=(--free-run --adapter adapters/lang-v2)
  [ "$arm" = raw ] && options=(--primitive-only --max-steps 40)
  for seed in 99200 99300; do
    out="runs/lang/free-$arm-$seed"
    if [ -f "$out/DONE" ]; then continue; fi
    if [ -d "$out" ]; then mv "$out" "$out-partial-$(date +%s)"; fi
    .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v2 "${options[@]}" \
      --output "$out" --rollout --scene-mix world --world-level 2 --seed "$seed" \
      --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931 && touch "$out/DONE"
  done
done
echo "FREE-RUN EVALUATION DONE"
