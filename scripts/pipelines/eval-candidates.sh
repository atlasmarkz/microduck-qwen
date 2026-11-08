#!/usr/bin/env bash
# Both adapter candidates with every execution-layer fix and the move/turn feedback, on layouts never used for any decision
# (seeds 99000 and 99100, 8 layouts each, five task types per layout). Same layouts for both adapters so they can be paired:
#   python -m microduck_qwen.compare_runs runs/lang/cand-v2-99000 runs/lang/cand-v2-99100 --b runs/lang/cand-v4-99000 runs/lang/cand-v4-99100
#   bash scripts/pipelines/eval-candidates.sh        # run from the repository root; about 4.5 hours
set -u
for seed in 99000 99100; do
  bash scripts/eval_world.sh adapters/lang-v2 data/lang-v2 cand-v2 "$seed"
  bash scripts/eval_world.sh adapters/lang-v4 data/lang-v4 cand-v4 "$seed"
done
echo "CANDIDATE EVALUATION DONE"
