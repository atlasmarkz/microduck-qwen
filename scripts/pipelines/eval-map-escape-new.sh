#!/usr/bin/env bash
# Confirmation of the perceived-map escape on layouts never used for any decision (seeds 99200/99300; lang-v2 training data
# stops at 97202, earlier evaluations used 98000-98700, 99000, 99100). Both arms run here, interleaved per seed group, so the
# flag is the only difference:
#   python -m microduck_qwen.truth_score runs/lang/mapoff-v2-99200 runs/lang/mapoff-v2-99300 --b runs/lang/mapesc-v2-99200 runs/lang/mapesc-v2-99300
set -u
for seed in 99200 99300; do
  MICRODUCK_MAP_ESCAPE=0 bash scripts/eval_world.sh adapters/lang-v2 data/lang-v2 mapoff-v2 "$seed"
  MICRODUCK_MAP_ESCAPE=1 bash scripts/eval_world.sh adapters/lang-v2 data/lang-v2 mapesc-v2 "$seed"
done
echo "MAP ESCAPE CONFIRMATION DONE"
