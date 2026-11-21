"""Extract ONNX initializers of the gait policies into policies/<name>.npz (run with .venv-wc)."""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_lab import POLICY_SHA256, WEIGHTS_DIR  # noqa: E402
from gait_lab.policy import TENSORS, onnx_path  # noqa: E402


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for name in POLICY_SHA256:
        model = onnx.load(str(onnx_path(name)))
        arrays = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        if set(arrays) != set(TENSORS):
            raise ValueError(f"{name}: unexpected initializers {sorted(arrays)}")
        np.savez(WEIGHTS_DIR / f"{name}.npz", __source_sha256__=np.array(POLICY_SHA256[name]), **arrays)
        print(name, sum(a.size for a in arrays.values()), "parameters ->", WEIGHTS_DIR / f"{name}.npz")


if __name__ == "__main__":
    main()
