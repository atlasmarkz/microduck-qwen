PY=.venv/Scripts/python
ev() { $PY -X utf8 -u -m microduck_qwen.agent_eval --data data/lang-v2 --adapter adapters/lang-v2 --output runs/lang/$1 --rollout --scene-mix world --world-level 2 --seed $2 --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3 --port 7931; }
ev recov2-98000 98000
ev recov2-98100 98100
