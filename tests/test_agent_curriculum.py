import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from microduck_qwen import agent_curriculum
from microduck_qwen.agent_curriculum import decide, free_seed_offset, held_out_seed


class DecisionTests(unittest.TestCase):
    def test_accepts_when_level_holds_and_nothing_is_forgotten(self):
        self.assertEqual(decide((6, 12), (8, 12), {0: (10, 12)}, {0: (9, 12)}, 0.25), (True, []))
        self.assertTrue(decide((6, 12), (6, 12), {}, {}, 0.25)[0])

    def test_rejects_a_drop_at_the_trained_level(self):
        accepted, reasons = decide((8, 12), (5, 12), {}, {}, 0.25)
        self.assertFalse(accepted)
        self.assertIn("fell", reasons[0])

    def test_rejects_forgetting_beyond_tolerance_or_a_missing_recheck(self):
        accepted, reasons = decide((6, 12), (9, 12), {0: (12, 12)}, {0: (6, 12)}, 0.25)
        self.assertFalse(accepted)
        self.assertIn("forgot", reasons[0])
        self.assertFalse(decide((6, 12), (9, 12), {0: (12, 12)}, {}, 0.25)[0])

    def test_correction_offset_skips_seeds_already_in_the_parent_data(self):
        rows = lambda *seeds: [{"scene_seed": seed} for seed in seeds]
        self.assertEqual(free_seed_offset({"train": rows(60000)}, 0, 2), 100)
        self.assertEqual(free_seed_offset({"train": rows(61100, 61101), "valid": rows(71100)}, 0, 2), 200)
        self.assertEqual(free_seed_offset({"train": rows(61700)}, 3, 1), 800)

    def test_held_out_seeds_are_per_level_and_outside_training_ranges(self):
        seeds = [held_out_seed(level) for level in range(4)]
        self.assertEqual(len(set(seeds)), 4)
        self.assertTrue(all(seed >= 95000 for seed in seeds))


class CurriculumRunTests(unittest.TestCase):
    def run_curriculum(self, rates, selected=True):
        """rates: iterator of (correct, total) returned by each agent_eval call, in order."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def process(command, **kwargs):
                module = command[command.index("-m") + 1].split(".")[-1]
                calls.append((module, command))
                if module == "agent_eval":
                    target = Path(command[command.index("--output") + 1])
                    target.mkdir(parents=True)
                    correct, total = next(rates)
                    (target / "report.json").write_text(json.dumps({"correct": correct, "total": total, "layers": {}}))
                elif module == "agent_cycle":
                    name = command[command.index("--name") + 1]
                    folder = root / "runs/agent-cycle" / name
                    folder.mkdir(parents=True)
                    adapter = str(root / "adapters" / (name + "-r0"))
                    (folder / "cycle.json").write_text(json.dumps({
                        "selected_adapter": adapter if selected else None,
                        "rounds": [{"adapter": adapter, "data": str(root / "data" / name)}]}))
                return SimpleNamespace(wait=lambda: 0)

            argv = ["curriculum", "--initial-data", str(root / "data"), "--initial-adapter", str(root / "adapters/base"),
                    "--name", "t", "--levels", "0", "1", "--eval-seeds", "2", "--forget-seeds", "1"]
            with patch.object(agent_curriculum, "ROOT", root), patch.object(agent_curriculum, "load_splits", return_value=({}, {})), \
                    patch.object(agent_curriculum.subprocess, "Popen", side_effect=process), patch("sys.argv", argv):
                agent_curriculum.main()
            return json.loads((root / "runs/agent-curriculum/t/curriculum.json").read_text()), calls

    def test_levels_that_already_pass_are_not_trained(self):
        report, calls = self.run_curriculum(iter([(4, 4), (4, 4)]))
        self.assertEqual([lv["action"] for lv in report["levels"]], ["promoted_without_training"] * 2)
        self.assertFalse(any(name == "agent_cycle" for name, _ in calls))
        self.assertTrue(report["final_adapter"].endswith("base"))

    def test_a_failing_level_trains_with_world_flags_then_is_accepted_if_it_improves(self):
        # L0 passes; L1 before 1/4; after 3/4; L0 re-check 4/4.
        report, calls = self.run_curriculum(iter([(4, 4), (1, 4), (3, 4), (4, 4)]))
        cycle = next(command for name, command in calls if name == "agent_cycle")
        self.assertEqual(cycle[cycle.index("--world-level") + 1], "1")
        self.assertIn("--collect-first", cycle)
        self.assertIn("--no-reward-guided", cycle)
        self.assertEqual(report["levels"][1]["action"], "accepted")
        self.assertTrue(report["final_adapter"].endswith("t-L1-r0"))

    def test_regression_is_rejected_and_the_previous_adapter_kept(self):
        report, _ = self.run_curriculum(iter([(4, 4), (2, 4), (1, 4), (4, 4)]))
        self.assertEqual(report["levels"][1]["action"], "rejected")
        self.assertTrue(report["final_adapter"].endswith("base"))

    def test_unqualified_training_keeps_the_adapter(self):
        report, _ = self.run_curriculum(iter([(4, 4), (1, 4)]), selected=False)
        self.assertEqual(report["levels"][1]["action"], "training_not_qualified")
        self.assertTrue(report["final_adapter"].endswith("base"))


if __name__ == "__main__":
    unittest.main()
