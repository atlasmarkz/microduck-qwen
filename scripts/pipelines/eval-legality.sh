#!/usr/bin/env bash
# lang-v2 with the action legality check on, same layouts and code as merged-v2 (paired, flag is the only difference):
#   python -m microduck_qwen.compare_runs runs/lang/merged-v2-99000 runs/lang/merged-v2-99100 --b runs/lang/legal-v2-99000 runs/lang/legal-v2-99100
set -u
export MICRODUCK_LEGALITY=1
for seed in 99000 99100; do
  bash scripts/eval_world.sh adapters/lang-v2 data/lang-v2 legal-v2 "$seed"
done
echo "LEGALITY EVALUATION DONE"
