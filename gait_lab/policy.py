"""NumPy forward pass of the 61-D -> 14-D gait MLP, identical in function to the ONNX graph.

Graph (verified for alpha_walking and velstand at revision 665ae4eb):
    x = (obs - obs_normalizer._mean) / onnx::Div_24
    x = ELU(x @ mlp.0.weight.T + mlp.0.bias)   # 512
    x = ELU(x @ mlp.2.weight.T + mlp.2.bias)   # 256
    x = ELU(x @ mlp.4.weight.T + mlp.4.bias)   # 128
    actions = x @ mlp.6.weight.T + mlp.6.bias  # 14
"""

import hashlib

import numpy as np

from . import POLICY_DIR, POLICY_SHA256, WEIGHTS_DIR

TENSORS = ("obs_normalizer._mean", "onnx::Div_24", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight",
           "mlp.2.bias", "mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias")


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def forward(params, obs):
    """Batched forward pass; obs [..., 61] -> actions [..., 14], float32 like the ONNX graph."""
    obs = np.asarray(obs, dtype=np.float32)
    x = (obs - params["obs_normalizer._mean"].reshape(-1)) / params["onnx::Div_24"].reshape(-1)
    for layer in ("mlp.0", "mlp.2", "mlp.4"):
        x = elu(x @ params[layer + ".weight"].T + params[layer + ".bias"])
    return x @ params["mlp.6.weight"].T + params["mlp.6.bias"]


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_params(name):
    """Weights extracted by scripts/extract_weights.py; checks the source ONNX hash recorded there."""
    path = WEIGHTS_DIR / f"{name}.npz"
    with np.load(path) as data:
        params = {key: data[key] for key in TENSORS}
        source = str(data["__source_sha256__"])
    if source != POLICY_SHA256[name]:
        raise ValueError(f"{path} was extracted from {source}, expected {POLICY_SHA256[name]}")
    return params


def onnx_path(name):
    path = POLICY_DIR / f"{name}.onnx"
    if file_sha256(path) != POLICY_SHA256[name]:
        raise ValueError(f"unexpected hash for {path}")
    return path
