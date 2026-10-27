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
from .vision import MODEL, REVISION, capture, propose_targets, save_frame
from .sim_data import digest


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
    feedback = ("\n执行器反馈：接近目标已成功完成，机器人已停稳。不要重复接近；要求停下则stop，要求回家则return_home。"
                if memory["completed"] and memory["completed"][-1]["action"] == "approach" else "")
    if memory["completed"] and memory["completed"][-1]["action"] in ("bypass_return", "return_home"):
        feedback = "\n执行器反馈：绕行与回家等已完成动作已经成功验收，机器人已停稳。仅处理指令剩余要求；没有剩余要求则done，禁止重复已完成的绕行或回家。"
    return ("MicroDuck仿真。根据图像、指令和本条指令已完成动作，输出下一步的单个JSON，不要解释。"
            '技能：{"action":"move","direction":"forward或backward","distance_m":0.2}；'
            '{"action":"turn","direction":"left或right","degrees":90}；'
            '{"action":"approach","color":"red或blue或green或yellow","id":图中编号}；'
            '或{"action":"approach","color":"yellow","selector":"nearest或leftmost或rightmost或only"}；'
            '黄色目标有球和柱子两种，可加"kind":"ball或pillar"区分，如{"action":"approach","color":"yellow","kind":"ball","selector":"only"}；'
            '{"action":"search","color":"red或blue或green或yellow"}；'
            '{"action":"return_home"}；{"action":"stop"}；{"action":"done"}；'
            '{"action":"bypass_return","side":"auto或left或right"}；'
            '{"action":"clarify"}；{"action":"unsupported"}。'
            "移动距离0.05至0.35米，转角5至120度；未指定时移动0.2米或转90度。"
            "靠近仅限清晰可见目标；找不到可搜索，遮挡不冒进。回初始位置用return_home（没有home时clarify）。"
            "黄色柱子禁止猜id，必须用selector：最近nearest、最左leftmost、最右rightmost，未指定用only（多根时执行器会停车询问）；接近柱子不绕行。接近后要求停下用stop，不回家。"
            "按顺序完成，已经完成不要重复；全部完成才done。踢球、跳舞、抓取等未实现技能用unsupported。"
            "依次绕过黄色障碍物并回原位用bypass_return：按任务开始距离由近到远，从一侧经过各柱后方并自动回本条任务起点；未指定侧用auto，指定左/右用left/right；不是绕整圈。此技能成功后直接done，不再重复return_home。"
            "记忆=" + json.dumps(memory, ensure_ascii=False, separators=(",", ":")) + "\n指令=" + instruction + feedback)


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


def run_instruction(planner, instruction, info, max_steps=5, preview=False, ask=None):
    directory = ROOT / "runs/agent" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    report = {"instruction": instruction, "active_instruction": instruction, "steps": [], "directory": str(directory),
              "status": "started", "task_success_verified": False, "interactions": [],
              "note": "No free-text semantic oracle. done is a model claim, not verified task success."}
    completed = []
    executor = None
    planning_instruction, pending_action = instruction, None

    def request_input(reason, retry_action=None):
        nonlocal planning_instruction, pending_action
        if ask is None:
            return False
        executor.robot.stop()
        while len(report["interactions"]) < 3:
            report["status"] = "waiting_for_user"
            prompt = (f"机器人已停车。{reason}\n"
                      + ("可输入“重试”（会重新检查安全条件）、" if retry_action else "可")
                      + "补充具体指令（改目标请输入“改为…”），或输入“取消”。\n你 > ")
            event = {"reason": reason, "question": prompt, "completed": list(completed),
                     "retry_allowed": retry_action is not None}
            report["interactions"].append(event)
            (directory / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            try:
                answer = ask(prompt).strip()
            except EOFError:
                report["status"] = "user_cancelled"
                return False
            event["answer"] = answer
            if answer.lower() in ("取消", "停止", "停下", "退出", "cancel", "stop", "exit"):
                report["status"] = "user_cancelled"
                return False
            executor.healthy()  # Cancellation and state freshness also apply after waiting.
            if not answer or answer in ("等待", "等一下", "等一会"):
                continue
            if answer in ("重试", "继续", "再试一次"):
                if retry_action is None:
                    reason = "当前目标仍不明确，请提供目标颜色、位置或明确的新指令。"
                    continue
                pending_action = dict(retry_action)
            else:
                if len(answer) > 160:
                    reason = "补充内容过长，请用不超过160字说明下一步意图。"
                    continue
                # Keep the original goal and measured successful history; never treat
                # a failed attempt as completed or silently retry partial movement.
                if answer.startswith(("改为", "改成")):
                    replacement = answer[2:].lstrip("：: ")
                    if not replacement:
                        reason = "请在“改为”后提供明确的新目标。"
                        continue
                    planning_instruction = replacement
                    report["active_instruction"] = replacement
                    event["replaced_goal"] = replacement
                    completed.clear()
                    state = executor.healthy()
                    executor.task_start_pose = {"x": state["trunk"][0], "y": state["trunk"][1], "yaw": state["yaw"]}
                    event["task_start_pose"] = executor.task_start_pose
                else:
                    planning_instruction = (planning_instruction + "\n执行反馈=" + reason
                        + "\n用户最新补充=" + answer + "\n结合已完成动作和当前图像重新选择下一步；安全约束始终有效。")
            report["status"] = "resuming"
            return True
        report["status"] = "dialogue_limit_stopped"
        return False

    with (ROOT / "runs/navigation.lock").open("a") as lock:
        try:
            lock_exclusive(lock)
        except BlockingIOError:
            raise RuntimeError("已有控制任务执行中，请先 Ctrl+C 或运行 agent stop") from None
        try:
            executor = SkillExecutor(info, directory, cancel_file=ROOT / "runs/cancel_navigation")
            executor.robot.stop()
            initial = executor.healthy()
            executor.task_start_pose = {"x": initial["trunk"][0], "y": initial["trunk"][1], "yaw": initial["yaw"]}
            report["task_start_pose"] = executor.task_start_pose
            for index in range(max_steps):
                frame = capture(info["port"])
                candidates = propose_targets(frame)
                image = save_frame(directory, frame, candidates, f"step-{index:03d}")
                memory = memory_state(info, executor.healthy(), completed)
                if pending_action is not None:
                    prediction = {"action": pending_action, "source": "user_retry_after_fresh_observation"}
                    pending_action = None
                else:
                    prediction = planner.predict(planning_instruction, image, memory)
                step = {"memory": memory, "prediction": prediction, "planning_instruction": planning_instruction}
                report["steps"].append(step)
                print(json.dumps(prediction, ensure_ascii=False), flush=True)
                action = prediction.get("action")
                executor.healthy()  # Also check cancellation AFTER potentially slow inference.
                if action is None:
                    report["status"] = "invalid_model_output"
                    if not preview and request_input("模型未生成可执行动作，请明确目标或改写指令。校验原因：" + prediction.get("error", "无有效动作")):
                        continue
                    break
                if preview:
                    report["status"] = "preview_only"
                    break
                step["execution"] = executor.run(action, candidates)
                if action["action"] == "clarify" and ask is not None:
                    if request_input("当前指令的目标或参数不明确，请补充说明。"):
                        continue
                    break
                if action["action"] in ("done", "clarify", "unsupported", "stop"):
                    report["status"] = "model_" + action["action"]
                    break
                if not step["execution"]["success"]:
                    report["status"] = "skill_failed_stopped"
                    execution = step["execution"]
                    if execution.get("needs_user_input"):
                        reason = ("当前移动路径在已知安全地图中受阻，动作尚未开始。" if action["action"] == "move"
                                  else "黄色障碍的视觉确认或安全绕行预检查未通过，动作尚未开始。" if action["action"] == "bypass_return"
                                  else "目标未唯一绑定或不可见，动作尚未开始。请明确最近、最左或最右的目标。" + execution.get("error", ""))
                        if request_input(reason, action if execution.get("retry_safe") else None):
                            continue
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
    parser.add_argument("--interactive", action="store_true", help="allow run to ask for clarification while stopped; chat enables this by default")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 8:
        parser.error("max-steps must be 1..8")
    if args.interactive and args.command in ("plan", "stop"):
        parser.error("interactive questions are available only for run/chat")
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
                    run_instruction(planner, text, info, args.max_steps, ask=input)
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as error:
                print(f"未执行：{error}", flush=True)
    else:
        run_instruction(planner, args.instruction, info, args.max_steps, args.command == "plan",
                        ask=input if args.interactive else None)


if __name__ == "__main__":
    main()
