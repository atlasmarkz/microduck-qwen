"""WeightCode BabyAI-style transfer to the microduck gait policy (see PROTOCOL.md)."""

import os
from pathlib import Path

BRANCH_ROOT = Path(__file__).resolve().parents[1]
# Main microduck-qwen checkout: simulation code, upstream microduck_rl and the policy cache.
MICRODUCK_ROOT = Path(os.environ.get("MICRODUCK_ROOT", "E:/Code/microduck-qwen"))
POLICY_REVISION = "665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448"
POLICY_DIR = (MICRODUCK_ROOT / ".cache/hub/models--pollen-robotics--microduck-policies/snapshots"
              / POLICY_REVISION)
POLICY_SHA256 = {
    "alpha_walking": "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
    "velstand": "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492",
}
WEIGHTS_DIR = BRANCH_ROOT / "policies"
