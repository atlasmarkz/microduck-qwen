PY=.venv/Scripts/python
$PY -X utf8 -u -m microduck_qwen.agent_data --output data/lang-v2 --base-data data/lang-w-v2 --actor-adapter adapters/lang-v1 --scene-mix varied --train-scenes 6 --seed-offset 8500 --language-per-color 3 --task-ids 0 1 2 6 7 8 --port 7862
$PY -X utf8 -u -m microduck_qwen.vision_torch train --task agent --data data/lang-v2 --output adapters/lang-v2 --steps 300 --resume adapters/lang-v1
ev() { $PY -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v2 --adapter $1 --output runs/lang/$2 --rollout --scene-mix world --world-level 2 --seed $3 --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931; }
ev adapters/lang-v2 v2-98000 98000
ev adapters/lang-v2 v2-98100 98100
