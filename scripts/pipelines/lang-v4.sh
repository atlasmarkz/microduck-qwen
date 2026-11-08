#!/usr/bin/env bash
# lang-v4: one more round of expert-intervened aggregation in the multi-room world with the current best adapter (lang-v2),
# aimed at the failures that remain after the execution-layer fixes: `stop` vs `done` after approach, English phrasings read as
# `clarify`, and the move/turn feedback prompt (agent_prompt now tells the model what the executor verified after move/turn).
# New train layouts only (seed offset 8800); valid/test of data/lang-v2 stay fixed. Then the regression canary decides whether
# any rollout evaluation is worth its two hours.
#
#   bash scripts/pipelines/lang-v4.sh            # run from the repository root, nothing else may use the simulator or the GPU
PY=.venv/Scripts/python
$PY -X utf8 -u -m microduck_qwen.agent_data --output data/lang-v4 --base-data data/lang-v2 --actor-adapter adapters/lang-v2 \
  --scene-mix world --world-level 2 --train-scenes 8 --seed-offset 8800 --task-ids 0 1 2 3 4 6 7 8 --explore-tasks 3 --port 7862 || exit 1
$PY -X utf8 -u -m microduck_qwen.vision_torch train --task agent --data data/lang-v4 --output adapters/lang-v4 --steps 300 --resume adapters/lang-v2 || exit 1
# Replay states of the dev layouts that are not training layouts; exit 1 if lang-v4 regressed against lang-v2 in any state class.
bash scripts/canary_chain.sh v4 "--replay runs/lang/eff-98000 runs/lang/eff-98100 runs/lang/ext-on-98200 runs/lang/ext-on-98300 runs/lang/ext-narrow-98200 runs/lang/ext-narrow-98300" \
  adapters/lang-v2 adapters/lang-v4
echo "LANG-V4 PIPELINE DONE"
