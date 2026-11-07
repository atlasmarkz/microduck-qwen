#!/usr/bin/env bash
# Explore diagnostic on the two layouts whose tasks ended with "explore found the colour, then no fully visible ball".
# Usage: bash scripts/diag_settle.sh 0|1   (MICRODUCK_SETTLE value)
set -u
for seed in 98205 98307; do
  MICRODUCK_SETTLE=$1 .venv/Scripts/python -X utf8 -u tools/diag_explore.py $seed runs/settle/s$1-$seed
done
echo "DIAG DONE"
