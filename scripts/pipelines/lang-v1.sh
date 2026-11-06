set -ex
PY=.venv/Scripts/python
$PY -X utf8 -u -m microduck_qwen.agent_data --output data/lang-w-v1 --base-data data/self-v4-r2 --actor-adapter adapters/self-v4-r2 --scene-mix world --world-level 2 --train-scenes 6 --seed-offset 6000 --task-ids 0 1 2 3 4 6 7 8 --explore-tasks 3 --port 7862
$PY -X utf8 -u -m microduck_qwen.agent_data --output data/lang-v1 --base-data data/lang-w-v1 --actor-adapter adapters/self-v4-r2 --scene-mix varied --train-scenes 6 --seed-offset 7000 --language-per-color 3 --task-ids 0 1 2 6 7 8 --port 7862
$PY -X utf8 -u -m microduck_qwen.vision_torch train --task agent --data data/lang-v1 --output adapters/lang-v1 --steps 300 --resume adapters/self-v4-r2
ev() { $PY -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v1 --adapter $1 --output runs/lang/$2 --rollout --scene-mix world --world-level 2 --seed $3 --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931; }
mkdir -p runs/lang
ev adapters/self-v4-r2 base-98000 98000
ev adapters/lang-v1 new-98000 98000
ev adapters/self-v4-r2 base-98100 98100
ev adapters/lang-v1 new-98100 98100
