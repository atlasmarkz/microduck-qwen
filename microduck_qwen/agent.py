"""Learned image/instruction/history -> bounded skill, distinct from visual referent SFT."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

from . import ROOT
from .locking import lock_exclusive
from .skills import SkillExecutor, memory_state, parse_action, validate_action
from .vision import MODEL, REVISION, capture, propose_balls, save_frame
from .vision_data import digest


def checked_memory(memory):
    if set(memory) != {"home_recorded", "odometry_relative_home", "completed"} or type(memory["home_recorded"]) is not bool:
        raise ValueError("invalid observable memory")
    pose = memory["odometry_relative_home"]
    if memory["home_recorded"] != (pose is not None):
        raise ValueError("home flag/odometry mismatch")
    if pose is not None:
        import math
        if set(pose) != {"x", "y", "yaw_deg"} or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose.values()):
            raise ValueError("invalid own odometry")
    if not isinstance(memory["completed"], list) or len(memory["completed"]) > 4:
        raise ValueError("invalid action history")
    for action in memory["completed"]:
        validate_action(action)
    return memory


def agent_prompt(instruction, memory):
    checked_memory(memory)
    return ("MicroDuck仿真。根据图像、指令和本条指令已完成动作，输出下一步的单个JSON，不要解释。"
            '技能：{"action":"move","direction":"forward或backward","distance_m":0.2}；'
            '{"action":"turn","direction":"left或right","degrees":90}；'
            '{"action":"approach","color":"red或blue或green","id":图中编号}；'
            '{"action":"search","color":"red或blue或green"}；'
            '{"action":"return_home"}；{"action":"stop"}；{"action":"done"}；'
            '{"action":"clarify"}；{"action":"unsupported"}。'
            "移动距离0.05至0.35米，转角5至120度；未指定时移动0.2米或转90度。"
            "靠近仅限清晰可见目标；找不到可搜索，遮挡不冒进。回初始位置用return_home（没有home时clarify）。"
            "按顺序完成，已经完成不要重复；全部完成才done。踢球、跳舞、抓取等未实现技能用unsupported。"
            "记忆=" + json.dumps(memory, ensure_ascii=False, separators=(",", ":")) + "\n指令=" + instruction)


def adapter_metadata(path):
    path = Path(path).resolve()
    report = json.loads((path / "training_report.json").read_text())
    if (report.get("format") != "agent_sft_v1" or report.get("status") != "completed"
            or report.get("model") != MODEL or report.get("revision") != REVISION
            or report.get("adapter_sha256") != digest(path / "adapters.safetensors")
            or report.get("adapter_config_sha256") != digest(path / "adapter_config.json")):
        raise ValueError("requires a completed, hash-verified agent adapter (not vision_sft_v1)")
    return {"path": str(path), "sha256": report["adapter_sha256"]}


class AgentPlanner:
    def __new__(cls, adapter=None):
        if os.name == "nt":
            from .vision_torch import TorchAgentPlanner
            return TorchAgentPlanner(adapter)
        return super().__new__(cls)

    def __init__(self, adapter=None):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        path = ROOT / ".cache/vlm-hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
        self.adapter = adapter_metadata(adapter) if adapter is not None else None
        self.model, self.processor = load(str(path), trust_remote_code=False,
                                         adapter_path=self.adapter["path"] if self.adapter else None)
        self.config = load_config(str(path))

    def predict(self, instruction, image, memory):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = agent_prompt(instruction, memory)
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        tick = time.perf_counter()
        output = generate(self.model, self.processor, formatted, image=[str(image)],
                          max_tokens=110, temperature=0.0, verbose=False)
        result = {"raw": output.text, "latency_s": time.perf_counter() - tick,
                  "prompt_tokens": output.prompt_tokens, "peak_memory_gb": getattr(output, "peak_memory", None),
                  "adapter": self.adapter}
        try:
            result["action"] = parse_action(output.text)
        except (ValueError, TypeError, KeyError) as error:
            result["error"] = str(error)
        return result


def run_instruction(planner, instruction, info, max_steps=5, preview=False):
    directory = ROOT / "runs/agent" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    report = {"instruction": instruction, "steps": [], "directory": str(directory),
              "status": "started", "task_success_verified": False,
              "note": "No free-text semantic oracle. done is a model claim, not verified task success."}
    completed = []
    executor = None
    with (ROOT / "runs/navigation.lock").open("a") as lock:
        try:
            lock_exclusive(lock)
        except BlockingIOError:
            raise RuntimeError("已有控制任务执行中，请先 Ctrl+C 或运行 agent stop") from None
        try:
            executor = SkillExecutor(info, directory, cancel_file=ROOT / "runs/cancel_navigation")
            executor.robot.stop()
            for index in range(max_steps):
                frame = capture(info["port"])
                candidates = propose_balls(frame)
                image = save_frame(directory, frame, candidates, f"step-{index:03d}")
                memory = memory_state(info, executor.healthy(), completed)
                prediction = planner.predict(instruction, image, memory)
                step = {"memory": memory, "prediction": prediction}
                report["steps"].append(step)
                print(json.dumps(prediction, ensure_ascii=False), flush=True)
                action = prediction.get("action")
                executor.healthy()  # Also check cancellation AFTER potentially slow inference.
                if action is None:
                    report["status"] = "invalid_model_output"
                    break
                if preview:
                    report["status"] = "preview_only"
                    break
                step["execution"] = executor.run(action, candidates)
                if action["action"] in ("done", "clarify", "unsupported", "stop"):
                    report["status"] = "model_" + action["action"]
                    break
                if not step["execution"]["success"]:
                    report["status"] = "skill_failed_stopped"
                    break
                completed.append(action)
            else:
                report["status"] = "step_limit_stopped"
        except BaseException as error:
            report.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if executor:
                    executor.close()
            finally:
                (directory / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(f"记录：{directory}; 状态：{report['status']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("chat", "run", "plan", "stop"))
    parser.add_argument("instruction", nargs="?")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/vision-simulator.json")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 8:
        parser.error("max-steps must be 1..8")
    info = json.loads(args.manifest.read_text())
    if args.command == "stop":
        (ROOT / "runs/cancel_navigation").write_text(str(time.time_ns()))
        from .navigation import Robot
        robot = Robot(info["socket"])
        try:
            robot.stop()
        finally:
            robot.close()
        return
    if args.command != "chat" and not args.instruction:
        parser.error("instruction required")
    planner = AgentPlanner(args.adapter)
    if args.command == "chat":
        print(f"输入指令；Ctrl+C 中断并停车，exit退出。每条指令最多{args.max_steps}步；home固定为仿真启动位置。")
        while True:
            try:
                text = input("你 > ").strip()
                if text in ("exit", "quit", "退出"):
                    break
                if text:
                    run_instruction(planner, text, info, args.max_steps)
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as error:
                print(f"未执行：{error}", flush=True)
    else:
        run_instruction(planner, args.instruction, info, args.max_steps, args.command == "plan")


if __name__ == "__main__":
    main()
