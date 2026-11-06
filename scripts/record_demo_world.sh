#!/usr/bin/env bash
# Record the multi-room exploration demo (assets/explore-rooms-4b-v1.gif). A run is only kept when the model really
# explored, approached and returned home (the recorder asserts it), so a failed attempt is simply retried.
#
#   bash scripts/record_demo_world.sh [ADAPTER] [WORLD_SEED]      # run from the repository root
adapter=${1:-adapters/lang-v2}; seed=${2:-98107}
mkdir -p logs
for i in 1 2 3; do
  .venv/Scripts/python -X utf8 -u -m microduck_qwen.record_demo --adapter "$adapter" --world --world-seed "$seed" > "logs/record_demo_$i.log" 2>&1 \
    && { echo "OK attempt $i"; exit 0; }
  echo "attempt $i failed: $(grep -E 'AssertionError|RuntimeError' "logs/record_demo_$i.log" | tail -1 | cut -c1-200)"
done
echo "ALL FAILED"; exit 1
