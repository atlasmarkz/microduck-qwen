"""No-harness baseline: the VLM drives the robot directly with primitive moves.

What the model gets: the plain RGB camera frame (no numbered candidates, no colour/depth proposer), the instruction, its own
odometry relative to the start, and its last few actions with their outcome. What it may output: one bounded `move`
(0.05-0.35 m), one bounded `turn` (5-120 deg), `stop` or `done`. Up to `max_steps` decisions per task.

What it does NOT get (the harness under test): the skill vocabulary (`explore`, `approach`, `search`, `return_home`,
`bypass_return`), the depth occupancy map, frontier exploration and A*, target binding and the visual guard of `approach`,
the validator, the executor feedback lines and the 5-skill plan structure.

Kept, because without them the robot cannot move or the comparison would only measure crashes: the frozen official gait
policy (via the same bounded move/turn executor), and the executor's known-map refusal of a translation that would end too
close to an obstacle (reported back to the model as "refused", like a bump sensor). Odometry and the refusal make this a
generous baseline, so the measured harness gain is a lower bound.

Scored only by simulator truth (`truth_score.score_episode`); there is no teacher and no strict score.
"""

import json

from .sim_data import collection_frame, request
from .skills import SkillExecutor, memory_state, parse_action
from .vision import MODEL, REVISION, save_frame

ACTIONS = ("move", "turn", "stop", "done")
TERMINAL = ("stop", "done")
HISTORY = 6
FATAL = ("fallen", "stale physics", "outside map", "cancelled")


def raw_prompt(instruction, odometry, history, step, max_steps):
    lines = [
        "MicroDuck仿真机器人。你直接控制它：根据当前相机图像、指令、里程计和最近动作，输出下一步的单个JSON，不要解释。",
        '动作只有四种：{"action":"move","direction":"forward或backward","distance_m":0.05至0.35}；'
        '{"action":"turn","direction":"left或right","degrees":5至120}；{"action":"stop"}；{"action":"done"}。',
        "每个动作执行完会给你新的图像。走到球旁边指机器人离球不超过约0.35米（球在画面下方、显得很大）。"
        "目标不在画面里时，要自己转向和走动寻找，可以穿过门洞去别的房间。回到起点指里程计x、y都接近0。"
        "不要撞墙和物体，不安全的移动会被拒绝。指令全部完成后输出done；指令要求停下则输出stop。",
        f"里程计（相对起点，x、y单位米，yaw单位度）={json.dumps(odometry, ensure_ascii=False)}",
        f"最近动作={json.dumps(history[-HISTORY:], ensure_ascii=False, separators=(',', ':'))}",
        f"已用步数={step}/{max_steps}",
        f"指令={instruction}",
    ]
    return "\n".join(lines)


def outcome(result):
    """Short text the model sees about its last action."""
    if result.get("needs_user_input"):
        return "被拒绝：不安全"
    if result.get("error"):
        return "中断：" + result["error"].split(":", 1)[0]
    return "完成" if result.get("success") else "未达到要求的距离或角度"


def raw_episode(directory, info, instruction, specs, generate, max_steps=40, meta=None):
    """Run one task with primitive control only; returns the episode report (same shape the truth scorer reads).

    `generate(image_path, prompt) -> (raw_text, seconds)` is the frozen model.
    """
    directory.mkdir(parents=True)
    report = {"instruction": instruction, "teacher_specs": specs, "task_meta": meta, "steps": [], "verified": False,
              "mode": "primitive_only", "max_steps": max_steps}
    executor = SkillExecutor(info, directory)
    history = []
    contact_start = request(info["port"], "collection_stats")["steps_with_scene_contact"]
    try:
        # Restore the task start exactly as the harness evaluation does; not part of the model's episode.
        reset = executor.run({"action": "return_home"})
        report["reset"] = reset
        if not reset["success"]:
            raise RuntimeError("cannot restore task start")
        for index in range(max_steps):
            executor.robot.stop()
            frame, gt = collection_frame(info["port"])
            save_frame(directory, frame, [], f"step-{index:03d}")
            image = directory / f"step-{index:03d}-rgb.png"
            if index == 0:  # the truth scorer resolves "the leftmost ball in the photo" from the first frame
                (directory / "step-000-truth.json").write_text(json.dumps({"ground_truth": gt}, indent=2) + "\n")
            memory = memory_state(info, executor.healthy(), [])
            prompt = raw_prompt(instruction, memory["odometry_relative_home"], history, index, max_steps)
            raw, latency = generate(image, prompt)
            prediction = {"instruction": instruction, "raw": raw, "latency_s": latency, "model": MODEL,
                          "revision": REVISION, "adapter": None}
            step = {"memory": memory, "prediction": prediction}
            report["steps"].append(step)
            try:
                action = parse_action(raw)
                if action["action"] not in ACTIONS:
                    raise ValueError(f"{action['action']} is not a primitive action")
            except (ValueError, TypeError, KeyError) as error:
                prediction["error"] = str(error)
                report["error"] = "invalid model output: " + str(error)
                break
            prediction["action"] = action
            if action["action"] in TERMINAL:
                report["verified"] = True  # ended by the model itself; success is decided by the truth scorer
                break
            result = executor.run(action)
            step["execution"] = result
            history.append({"action": action, "result": outcome(result)})
            if any(word in str(result.get("error", "")) for word in FATAL):
                report["error"] = "robot stopped: " + result["error"]
                break
        else:
            report["error"] = f"step budget of {max_steps} exhausted without done or stop"
        stats = request(info["port"], "collection_stats")
        report["scene_contact_steps_delta"] = stats["steps_with_scene_contact"] - contact_start
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            executor.close()
        finally:
            (directory / "episode.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
