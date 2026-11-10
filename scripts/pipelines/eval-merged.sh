#!/usr/bin/env bash
# lang-v2 with the merged explore/approach fixes on the same layouts as cand-v2 (paired):
#   python -m microduck_qwen.compare_runs runs/lang/cand-v2-99000 runs/lang/cand-v2-99100 --b runs/lang/merged-v2-99000 runs/lang/merged-v2-99100
set -u
for seed in 99000 99100; do
  bash scripts/eval_world.sh adapters/lang-v2 data/lang-v2 merged-v2 "$seed"
done
echo "MERGED EVALUATION DONE"
