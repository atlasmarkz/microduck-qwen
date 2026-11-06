set -x
PY=.venv/Scripts/python
ev() { $PY -X utf8 -u -m microduck_qwen.agent_eval --data data/explore-v1 --adapter $1 --output runs/diag/$2 --rollout --scene-mix world --world-level 2 --seed $3 --rollout-seeds 6 --task-ids 3 4 --explore-tasks 1 --port 7931; }
mkdir -p runs/diag
ev adapters/agent-explore-v1 noise-a-98000 98000
ev adapters/agent-explore-v1 noise-b-98000 98000
ev adapters/agent-explore-v1 noise-c-98100 98100
ev adapters/agent-explore-v1 noise-d-98200 98200
$PY -X utf8 -u -m microduck_qwen.vision_torch train --task agent --data data/explore-v1 --output adapters/diag-control --steps 150 --resume adapters/agent-explore-v1
ev adapters/diag-control control-98000 98000
ev adapters/diag-control control-98100 98100
