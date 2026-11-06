PY=.venv/Scripts/python
$PY -X utf8 -u -m microduck_qwen.agent_data --output data/lang-v3 --base-data data/lang-v2 --actor-adapter adapters/lang-v2 --scene-mix varied --train-scenes 7 --seed-offset 8700 --language-per-color 1 --task-ids 0 1 2 4 5 6 7 8 --port 7862
$PY -X utf8 -u -m microduck_qwen.vision_torch train --task agent --data data/lang-v3 --output adapters/lang-v3 --steps 300 --resume adapters/lang-v2
ev() { $PY -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v3 --adapter adapters/lang-v3 --output runs/lang/$1 --rollout --scene-mix world --world-level 2 --seed $2 --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931; }
ev v3-98000 98000
ev v3-98100 98100
