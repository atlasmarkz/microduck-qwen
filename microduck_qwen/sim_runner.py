"""Own a child simulator process: start it, wait for its manifest, stop it.

Shared by every collector and evaluator so scene startup/cleanup lives in one
place. Ground truth never enters this module.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import ROOT


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def start_scene(case_directory, seed, occlusion, port, collect_gt=False, clutter=False):
    failures = []
    for attempt in range(2):
        manifest = case_directory / f"simulator-{attempt}.json"
        log_path = case_directory / f"simulator-{attempt}.log"
        log = log_path.open("w")
        process = subprocess.Popen(
            [
                sys.executable if os.name == "nt" else str(ROOT / ".venv/bin/python"),
                "-X", "utf8",
                "-u",
                "-m",
                "microduck_qwen.sim_native" if os.name == "nt" else "microduck_qwen.sim",
                "--vision",
                "--seed",
                str(seed),
                "--occlusion",
                occlusion,
                "--port",
                str(port),
                "--manifest",
                str(manifest),
            ] + (["--collect-gt"] if collect_gt else []) + (["--clutter"] if clutter else []),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 35
        try:
            while (
                time.monotonic() < deadline
                and process.poll() is None
                and not manifest.exists()
            ):
                time.sleep(0.1)
            if manifest.exists() and process.poll() is None:
                return process, log, json.loads(manifest.read_text()), failures
        except BaseException:
            stop_process(process)
            log.close()
            raise
        stop_process(process)
        log.close()
        failures.append(
            {
                "attempt": attempt + 1,
                "log": str(log_path),
                "error": log_path.read_text()[-2000:],
            }
        )
    raise RuntimeError(
        "scene failed both startup attempts: "
        + json.dumps(failures, ensure_ascii=False)
    )
