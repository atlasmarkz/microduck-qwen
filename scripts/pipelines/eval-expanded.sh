#!/usr/bin/env bash
# Expanded evaluation of adapters/lang-v2 on layouts never evaluated before (seeds 98200 and 98300, 8 layouts each):
#   1. three alternative held-out wordings of the two non-explore tasks (phrase-v1..v3, seed 98200);
#   2. recovery ablation on the same 80 new tasks: MICRODUCK_RECOVERY=0 (ext-off) vs the default (ext-on).
# Compare the arms with:  python -m microduck_qwen.compare_runs runs/lang/ext-off-98200 runs/lang/ext-off-98300 --b runs/lang/ext-on-98200 runs/lang/ext-on-98300
#
#   bash scripts/pipelines/eval-expanded.sh          # run from the repository root; takes about 4-5 hours
set -u
adapter=adapters/lang-v2; data=data/lang-v2

for variant in 1 2 3; do
  .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_eval --data "$data" --adapter "$adapter" \
    --output "runs/lang/phrase-v$variant-98200" --rollout --scene-mix world --world-level 2 --seed 98200 \
    --rollout-seeds 8 --task-ids 3 4 --phrase-variant "$variant" --port 7931
done

MICRODUCK_RECOVERY=0 bash scripts/eval_world.sh "$adapter" "$data" ext-off 98200 98300
MICRODUCK_RECOVERY=1 bash scripts/eval_world.sh "$adapter" "$data" ext-on 98200 98300
echo "EXPANDED EVALUATION DONE"
