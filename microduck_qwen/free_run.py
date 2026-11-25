"""Uncut evaluation episode: the deployed closed loop (`agent.run_instruction`) with nobody to ask.

`agent_data.episode` walks the teacher's step list and stops at the model's first deviation, so a run
that would have ended well (or badly) on its own is cut there and has no natural outcome. Here there
is no teacher: the model decides until it ends the task itself, a skill fails, its output cannot be
executed, or the deployed step limit is reached. Success is decided only by simulator truth
(`truth_score.score_episode`), so it is not comparable with the teacher-strict score.
"""

import json

from .sim_data import collection_frame, request
from .skills import SkillExecutor, memory_state
from .vision import propose_targets, save_frame

MAX_STEPS = 5  # run_instruction's default, which the Colab console uses
TERMINAL = ("done", "stop", "clarify", "unsupported")


def free_episode(directory, info, instruction, specs, planner, max_steps=MAX_STEPS, meta=None):
    directory.mkdir(parents=True)
    report = {"instruction": instruction, "teacher_specs": specs, "task_meta": meta, "steps": [], "verified": False,
              "mode": "free_run", "max_steps": max_steps}
    executor = SkillExecutor(info, directory)
    completed = []
    contact_start = request(info["port"], "collection_stats")["steps_with_scene_contact"]
    try:
        # Restore the task start exactly as the other evaluations do; not part of the model's episode.
        reset = executor.run({"action": "return_home"})
        report["reset"] = reset
        if not reset["success"]:
            raise RuntimeError("cannot restore task start")
        initial = executor.healthy()
        executor.task_start_pose = {"x": initial["trunk"][0], "y": initial["trunk"][1], "yaw": initial["yaw"]}
        for index in range(max_steps):
            executor.robot.stop()
            frame, gt = collection_frame(info["port"])
            candidates = propose_targets(frame)
            image = save_frame(directory, frame, candidates, f"step-{index:03d}")
            if index == 0:  # the truth scorer resolves "the leftmost ball in the photo" from the first frame
                (directory / "step-000-truth.json").write_text(json.dumps({"ground_truth": gt}, indent=2) + "\n")
            state = executor.healthy()
            memory = memory_state(info, state, completed)
            prediction = planner.predict(instruction, image, memory)
            step = {"memory": memory, "prediction": prediction, "sim_time": state["sim_time"],
                    "decided_sim_time": executor.observer.read()["sim_time"]}
            report["steps"].append(step)
            action = prediction.get("action")
            if action is None:
                report["error"] = "invalid model output: " + str(prediction.get("error"))
                break
            step["execution"] = result = executor.run(action, candidates)
            if action["action"] in TERMINAL:
                report["verified"] = True  # ended by the model itself; success is decided by the truth scorer
                break
            if not result["success"]:
                report["error"] = "skill failed: " + str(result.get("error"))
                break
            completed.append(action)
        else:
            report["error"] = f"step limit of {max_steps} reached without a terminal decision"
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
