#!/usr/bin/env bash
# lang-v2 with the perceived-map escape in explore, same layouts and code as merged-v2 / legal-v2 (paired, flag is the only difference):
#   python -m microduck_qwen.truth_score runs/lang/merged-v2-99000 runs/lang/merged-v2-99100 --b runs/lang/mapesc-v2-99000 runs/lang/mapesc-v2-99100
set -u
export MICRODUCK_MAP_ESCAPE=1
for seed in 99000 99100; do
  bash scripts/eval_world.sh adapters/lang-v2 data/lang-v2 mapesc-v2 "$seed"
done
echo "MAP ESCAPE EVALUATION DONE"
