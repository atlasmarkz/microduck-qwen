#!/usr/bin/env bash
# A/B of the far re-acquisition on the layouts whose approach lost a ball 3 m away (98302) or refused at the start (98204).
# MICRODUCK_FAR_REACQUIRE=0 (before) and 1 (after), two repeats each because one run of one layout is noisy.
#   bash scripts/pipelines/eval-far-reacquire.sh
set -u
for rep in 1 2; do
  for flag in 0 1; do
    MICRODUCK_FAR_REACQUIRE=$flag bash scripts/eval_layouts.sh adapters/lang-v2 data/lang-v2 far$flag-r$rep 98302 98204
  done
done
echo "FAR REACQUIRE DONE"
