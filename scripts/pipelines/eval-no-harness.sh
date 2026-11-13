#!/usr/bin/env bash
# No-harness baseline (microduck_qwen/raw_baseline.py): frozen Qwen3-VL-4B drives the robot with move/turn/stop/done only,
# plain RGB, odometry and last actions, up to 40 decisions per task. Same 80 tasks and layouts as `frozen`, `frozterm` and
# `mapoff-v2` (seeds 99200/99300, tasks 3 4 and 3 explore tasks per hidden colour). Scored by simulator truth:
#   python -m microduck_qwen.truth_score runs/lang/raw-99200 runs/lang/raw-99300 --b runs/lang/frozen-99200 runs/lang/frozen-99300
set -u
export MICRODUCK_MAP_ESCAPE=0 MICRODUCK_LEGALITY=0 MICRODUCK_TERMINATION=0
for seed in 99200 99300; do
  .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v2 \
    --output "runs/lang/raw-$seed" --rollout --scene-mix world --world-level 2 --seed "$seed" \
    --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931 --primitive-only --max-steps 40
done
echo "NO-HARNESS EVALUATION DONE"
