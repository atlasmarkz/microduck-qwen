"""Small reproducible integration battery, not a publication benchmark.

Owns only its child simulators. Ground truth enters scoring AFTER inference,
never the VLM input. Startup attempts and rejected/failed trials are retained.
"""

import argparse
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

import numpy as np
from PIL import Image

from . import ROOT
from .vision import Frame, VisionPlanner, project, run_instruction, visibility_check
from .vision_scene import make_scene


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def load_saved_frame(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "initial-observation.json").read_text())[
        "sensor"
    ]
    return Frame(
        np.asarray(Image.open(directory / "initial-rgb.png")),
        np.load(directory / "initial-depth.npz")["depth"],
        metadata,
    )


def start_scene(case_directory, seed, occlusion, port, collect_gt=False):
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
            ] + (["--collect-gt"] if collect_gt else []),
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


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--port", type=int, default=7832)
    parser.add_argument("--navigate", action="store_true")
    parser.add_argument("--adapter", type=Path, help="explicit completed task-level vision LoRA")
    args = parser.parse_args()
    output = ROOT / "runs/vision-eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    planner = VisionPlanner(args.adapter)
    cases = [(seed, "none") for seed in args.seeds] + [
        (args.seeds[0], kind) for kind in ("partial", "full", "absent")
    ]
    results = []
    memory = None
    for seed, kind in cases:
        case = output / f"seed-{seed}-{kind}"
        case.mkdir()
        row = {"seed": seed, "scenario": kind, "passed": False}
        process = log = None
        try:
            process, log, info, failures = start_scene(case, seed, kind, args.port)
            row["startup_failures"] = failures
            reference = make_scene(seed, "none")["objects"]["ball_0"]
            color_cn = {"red": "红", "blue": "蓝", "green": "绿"}[reference["color"]]
            instruction = (
                "走到画面左边的球旁边" if kind == "none" else f"走到{color_cn}色球旁边"
            )
            result = run_instruction(
                planner, instruction, info, preview=not args.navigate, timeout=45
            )
            row.update(
                result_file=str(Path(result["directory"]) / "result.json"),
                executed=result["executed"],
                vlm_answer=result.get("answer"),
                latency_s=result.get("latency_s"),
                peak_memory_gb=result.get("peak_memory_gb"),
            )
            frame = load_saved_frame(result["directory"])
            if kind == "none":
                scene = json.loads(Path(info["scene_file"]).read_text())
                projected = []
                for name, obj in scene["objects"].items():
                    p = project([obj["x"], obj["y"], obj["radius"]], frame.metadata)
                    if (
                        p is not None
                        and 0 < p[0] < frame.metadata["width"]
                        and 0 < p[1] < frame.metadata["height"]
                    ):
                        projected.append((p[0], name, obj))
                _, expected_name, expected = min(projected, key=lambda p: p[0])
                selected = result.get("selected")
                row["expected_target"] = expected_name
                row["vlm_grounding_correct"] = bool(
                    selected
                    and selected["color"] == expected["color"]
                    and math.dist(
                        selected["world_position"][:2], [expected["x"], expected["y"]]
                    )
                    < 0.15
                )
                row["localization_error_m"] = (
                    math.dist(
                        selected["world_position"][:2], [expected["x"], expected["y"]]
                    )
                    if selected
                    else None
                )
                row["physical_navigation_success"] = result.get("execution", {}).get(
                    "success", False
                )
                if row["physical_navigation_success"]:
                    goal = result["execution"]["goals"][0]
                    row["true_final_distance_m"] = math.dist(
                        goal["position"], [expected["x"], expected["y"]]
                    )
                    row["navigation_elapsed_s"] = goal["elapsed_s"]
                    row["physical_navigation_success"] &= (
                        row["true_final_distance_m"] <= 0.35
                    )
                row["passed"] = row["vlm_grounding_correct"] and (
                    not args.navigate or row["physical_navigation_success"]
                )
                if seed == args.seeds[0] and selected and row["vlm_grounding_correct"]:
                    memory = (
                        selected["world_position"],
                        selected["color"],
                        result["directory"],
                    )
            else:
                row["vlm_abstained"] = result.get("answer", {}).get("status") in (
                    "observe",
                    "clarify",
                )
                row["safe_nonexecution"] = not result["executed"]
                row["passed"] = row["safe_nonexecution"]
                row["stopped_reason"] = result.get("stopped_reason")
                if kind == "full" and memory:
                    row["paired_reset_scene_memory_check"] = visibility_check(
                        frame, np.asarray(memory[0]), memory[1]
                    )
                    row["previous_visible_frame"] = memory[2]
        except Exception as error:
            row["error"] = str(error)
        finally:
            if process is not None:
                stop_process(process)
            if log is not None:
                log.close()
        results.append(row)
        (output / "report.json").write_text(
            json.dumps(
                {
                    "results": results,
                    "navigation_requested": args.navigate,
                    "adapter": planner.adapter,
                    "versions": {
                        p: importlib.metadata.version(p)
                        for p in (("torch", "peft", "bitsandbytes", "transformers", "opencv-python") if os.name == "nt"
                            else ("mlx-vlm", "mlx", "transformers", "opencv-python"))
                    },
                    "note": "Development smoke battery. Robot pose and safety map are oracle. "
                    + ("Explicit task-level vision LoRA loaded. " if planner.adapter else "Frozen VLM baseline. ")
                    + "Paired memory test resets scenes; not an uninterrupted temporal trial.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print("REPORT: " + str(output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
