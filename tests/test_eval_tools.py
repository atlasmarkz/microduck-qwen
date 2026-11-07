import json
import tempfile
import unittest
from pathlib import Path

from microduck_qwen.agent_canary import decision_correct, regressions, replay_rows, state_class, summarize
from microduck_qwen.compare_runs import compare, family, is_invalid, load, mcnemar_exact


class CanaryTests(unittest.TestCase):
    def row(self, completed, expected):
        return {"memory": {"completed": [{"action": a} for a in completed]}, "expected": {"action": expected}}

    def test_state_class_names_the_last_completed_skill_or_start(self):
        self.assertEqual(state_class(self.row([], "explore")), "start->explore")
        self.assertEqual(state_class(self.row(["approach", "return_home"], "done")), "return_home->done")

    def test_a_class_that_loses_rows_is_flagged_and_gains_are_not(self):
        old = summarize([{"state": "move->done", "correct": True}] * 4 + [{"state": "start->move", "correct": False}])
        new = summarize([{"state": "move->done", "correct": c} for c in (True, True, False, False)]
                        + [{"state": "start->move", "correct": True}])
        flagged = regressions(new, old)
        self.assertEqual([f["state"] for f in flagged], ["move->done"])
        self.assertEqual((flagged[0]["before"], flagged[0]["after"]), (4, 2))

    def test_classes_with_different_totals_are_not_compared(self):
        self.assertEqual(regressions({"a->b": [1, 2]}, {"a->b": [3, 3]}), [])


class ReplayTests(unittest.TestCase):
    def test_rows_come_from_recorded_steps_that_have_an_image_and_a_teacher_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "seed-5" / "task-04"
            task.mkdir(parents=True)
            (task / "step-000-candidates.png").write_bytes(b"x")
            memory = {"home_recorded": True, "completed": []}
            steps = [{"expected": {"action": "move", "direction": "forward", "distance_m": 0.2}, "memory": memory,
                      "prediction": {"instruction": "往前移动20厘米"}},
                     {"expected": {"action": "done"}, "memory": memory, "prediction": {"instruction": "往前移动20厘米"}}]
            (task / "episode.json").write_text(json.dumps({"instruction": "往前移动20厘米", "steps": steps}), encoding="utf-8")
            rows = replay_rows([tmp])
            self.assertEqual(len(rows), 1)  # step 1 has no recorded image
            self.assertEqual(rows[0]["expected"]["action"], "move")
            self.assertTrue(rows[0]["id"].endswith("seed-5/task-04#0"))

    def test_replay_counts_the_action_type_but_the_dataset_split_needs_an_exact_match(self):
        expected = {"action": "approach", "color": "blue", "selector": "nearest"}
        prediction = {"action": {"action": "approach", "color": "blue", "selector": "leftmost"}}
        self.assertTrue(decision_correct(prediction, expected, replay=True))
        self.assertFalse(decision_correct(prediction, expected, replay=False))
        self.assertFalse(decision_correct({"action": None}, expected, replay=True))  # invalid JSON never counts
        self.assertFalse(decision_correct({"action": {"action": "unsupported"}}, {"action": "done"}, replay=True))


class PairedComparisonTests(unittest.TestCase):
    def test_mcnemar_exact_matches_hand_values(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(5, 0), 2 / 32)          # 2 * P(X<=0), n=5
        self.assertAlmostEqual(mcnemar_exact(3, 3), 1.0)             # balanced discordance
        self.assertAlmostEqual(mcnemar_exact(1, 4), 2 * 6 / 32)      # 2 * P(X<=1), n=5

    def test_compare_counts_only_discordant_pairs_and_splits_families(self):
        a = {(1, 3): True, (1, 4): False, (1, 200): False, (1, 201): True}
        b = {(1, 3): True, (1, 4): True, (1, 200): True, (1, 201): True}
        result = compare(a, b, resamples=200)
        self.assertEqual((result["only_a"], result["only_b"], result["pairs"]), (0, 2, 4))
        self.assertAlmostEqual(result["diff_rate"], 0.5)
        self.assertEqual(result["by_family"]["explore"], {"a": 1, "b": 2, "n": 2})
        self.assertEqual(family(4), "task4")

    def test_runs_without_shared_tasks_are_rejected(self):
        with self.assertRaises(ValueError):
            compare({(1, 3): True}, {(2, 3): True})

    def test_tasks_the_harness_could_not_label_are_not_model_outcomes(self):
        invalid = {"seed": 7, "task_id": 3, "correct": False,
                   "error": "ValueError: several yellow balls in view; teacher leftmost target is ambiguous"}
        self.assertTrue(is_invalid(invalid))
        self.assertTrue(is_invalid({"seed": 7, "task_id": 3, "correct": False,
                                    "error": "ValueError: teacher leftmost target not confidently visible"}))
        self.assertTrue(is_invalid({"seed": 7, "task_id": 3, "correct": False, "invalid_task": True}))
        self.assertFalse(is_invalid({"seed": 7, "task_id": 3, "correct": False, "error": "failed physical skill"}))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "report.json").write_text(json.dumps({"results": [invalid, {"seed": 7, "task_id": 4, "correct": True}]}))
            self.assertEqual(load(tmp), {(7, 4): True})

    def test_load_reads_seed_and_task_from_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "report.json").write_text(json.dumps({"results": [{"seed": 7, "task_id": 3, "correct": True}]}))
            self.assertEqual(load(tmp), {(7, 3): True})


if __name__ == "__main__":
    unittest.main()
