import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from microduck_qwen import agent
from microduck_qwen.skills import NeedsUserInput, SkillExecutor


MOVE = {"action": "move", "direction": "forward", "distance_m": 0.2}
BLOCKED = {"success": False, "needs_user_input": True, "no_motion": True, "retry_safe": True}


class AgentDialogueTests(unittest.TestCase):
    def test_invalid_model_output_asks_for_clarification_in_interactive_mode(self):
        report, planner, executor, captures = self.run_case(
            [{"success": True}], [None, {"action": "stop"}], ["最近的一根黄色柱子"])
        self.assertEqual(report["status"], "model_stop")
        self.assertEqual(captures, 2)
        self.assertEqual(executor.run.call_count, 1)
        self.assertIn("最近的一根黄色柱子", planner.predict.call_args.args[0])
        self.assertEqual(len(report["interactions"]), 1)

    def run_case(self, results, predictions, answers=None, max_steps=5):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = Mock()
            executor.healthy.return_value = {"trunk": [0, 0, 0.1], "yaw": 0}
            executor.run.side_effect = results
            planner = Mock()
            planner.predict.side_effect = [{"action": action} for action in predictions]
            iterator = iter(answers or [])
            def ask(prompt):
                self.assertIn("已停车", prompt)
                executor.robot.stop.assert_called()
                saved = json.loads(next((root / "runs/agent").glob("*/result.json")).read_text())
                self.assertEqual(saved["status"], "waiting_for_user")
                return next(iterator)
            with patch.object(agent, "ROOT", root), patch.object(agent, "SkillExecutor", return_value=executor), patch.object(agent, "capture", return_value=None) as capture, patch.object(agent, "propose_targets", return_value=[]), patch.object(agent, "save_frame", return_value=root / "frame.png"):
                report = agent.run_instruction(planner, "前进0.2米，然后回家", {"port": 1}, max_steps,
                                               ask=ask if answers is not None else None)
            return report, planner, executor, capture.call_count

    def test_retry_recaptures_and_rechecks_without_reasking_model(self):
        report, planner, executor, captures = self.run_case(
            [BLOCKED, {"success": True}, {"success": True}], [MOVE, {"action": "stop"}], ["重试"])
        self.assertEqual(captures, 3)
        self.assertEqual(planner.predict.call_count, 2)
        self.assertEqual(executor.run.call_args_list[1].args[0], MOVE)
        self.assertEqual(report["steps"][1]["memory"]["completed"], [])
        self.assertEqual(report["steps"][2]["memory"]["completed"], [MOVE])
        self.assertEqual(report["status"], "model_stop")

    def test_cancel_and_wait_never_issue_more_motion(self):
        report, _, executor, captures = self.run_case([BLOCKED], [MOVE], ["等待", "取消"])
        self.assertEqual(report["status"], "user_cancelled")
        self.assertEqual(executor.run.call_count, 1)
        self.assertEqual(captures, 1)
        self.assertEqual(len(report["interactions"]), 2)

    def test_clarification_passes_reply_with_original_goal_to_fresh_inference(self):
        report, planner, _, _ = self.run_case(
            [{"success": False}, {"success": True}], [{"action": "clarify"}, {"action": "stop"}], ["目标是左边的红球"])
        text = planner.predict.call_args.args[0]
        self.assertIn("前进0.2米，然后回家", text)
        self.assertIn("目标是左边的红球", text)
        self.assertFalse(report["task_success_verified"])

    def test_noninteractive_and_partial_failure_never_auto_retry(self):
        report, _, executor, _ = self.run_case([BLOCKED], [MOVE])
        self.assertEqual(report["status"], "skill_failed_stopped")
        self.assertFalse(report["interactions"])
        partial, _, executor, _ = self.run_case([{"success": False, "error": "partial motion timeout"}], [MOVE], ["重试"])
        self.assertEqual(partial["status"], "skill_failed_stopped")
        self.assertFalse(partial["interactions"])
        self.assertEqual(executor.run.call_count, 1)

    def test_explicit_goal_replacement_starts_new_progress_without_erasing_audit(self):
        report, planner, _, _ = self.run_case(
            [{"success": True}, BLOCKED, {"success": True}], [MOVE, MOVE, {"action": "stop"}], ["改为：停下来"])
        self.assertEqual(planner.predict.call_args.args[0], "停下来")
        self.assertEqual(planner.predict.call_args.args[2]["completed"], [])
        self.assertEqual(report["steps"][1]["memory"]["completed"], [MOVE])
        self.assertEqual(report["instruction"], "前进0.2米，然后回家")
        self.assertEqual(report["active_instruction"], "停下来")

    def test_repeated_blocking_has_a_question_budget(self):
        report, _, executor, _ = self.run_case([BLOCKED] * 4, [MOVE], ["重试"] * 3)
        self.assertEqual(report["status"], "dialogue_limit_stopped")
        self.assertEqual(len(report["interactions"]), 3)
        self.assertEqual(executor.run.call_count, 4)

    def test_only_pre_execution_blocking_is_marked_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = object.__new__(SkillExecutor)
            executor.directory, executor.index = Path(tmp), 0
            executor.robot = Mock()
            executor._bounded = Mock(side_effect=NeedsUserInput("blocked before motion"))
            blocked = executor.run(MOVE)
            self.assertTrue(blocked["retry_safe"])
            executor._bounded.side_effect = TimeoutError("partial movement")
            failed = executor.run(MOVE)
            self.assertNotIn("retry_safe", failed)


if __name__ == "__main__":
    unittest.main()
