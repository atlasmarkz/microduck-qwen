"""Parallel episode evaluation: one MuJoCo world per worker process, params sent once per batch."""

import multiprocessing as mp
import os

import numpy as np

from . import policy as pol
from .sim import rollout

_PARAMS = {}


def _work(job):
    key, params, items = job
    if params is not None:
        _PARAMS[key] = params
    p = _PARAMS[key]
    return [rollout(lambda obs: pol.forward(p, obs), seed, condition, command)
            for seed, condition, command in items]


def evaluate(params, items, workers=None, key="p"):
    """items: list of (seed, condition, command-or-None). Returns results in the same order."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    chunks = [items[i::workers] for i in range(workers)]
    jobs = [(key, params, chunk) for chunk in chunks if chunk]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        parts = pool.map(_work, jobs)
    out = [None] * len(items)
    for w, part in enumerate(parts):
        for j, res in enumerate(part):
            out[w + j * workers] = res
    return out


def params_sha(params):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(params):
        h.update(k.encode())
        h.update(np.ascontiguousarray(params[k], dtype=np.float32).tobytes())
    return h.hexdigest()
