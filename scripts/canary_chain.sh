#!/usr/bin/env bash
# Regression canary along the adapter lineage: each adapter is compared with its predecessor, state class by state class.
#
#   bash scripts/canary_chain.sh TAG "SOURCE FLAGS" ADAPTER_0 ADAPTER_1 [ADAPTER_2 ...]
#   bash scripts/canary_chain.sh replay "--replay runs/lang/eff-98000 runs/lang/eff-98100" adapters/lang-v1 adapters/lang-v2 adapters/lang-v3
#   bash scripts/canary_chain.sh split "--data data/lang-v3" adapters/lang-v1 adapters/lang-v2
#
# Writes runs/canary/TAG/<adapter name>.json; exits 1 if any adapter regressed against the previous one.
# The replay source re-decides the steps of earlier rollout evaluations (world images); the split source is the
# 21-row held-out split of a dataset (see microduck_qwen/agent_canary.py for why replay is the one that matters).
set -u
tag=$1; source_flags=$2; shift 2
mkdir -p "runs/canary/$tag"
previous=""; status=0
for adapter in "$@"; do
  name=$(basename "$adapter")
  extra=""; [ -n "$previous" ] && extra="--baseline runs/canary/$tag/$previous.json"
  # shellcheck disable=SC2086
  .venv/Scripts/python -X utf8 -u -m microduck_qwen.agent_canary $source_flags --adapter "$adapter" --output "runs/canary/$tag/$name.json" $extra || status=1
  previous=$name
done
exit $status
