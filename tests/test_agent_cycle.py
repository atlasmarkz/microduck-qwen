import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from microduck_qwen import agent_cycle


class AgentCycleTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "CUDA reward iteration is Windows-only")
    def test_two_rounds_train_reward_before_guided_collection_and_resume_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modules = []

            def process(command, **kwargs):
                module = command[command.index("-m") + 1].split(".")[-1]
                modules.append((module, command))
                target = Path(command[command.index("--output") + 1])
                if module == "agent_eval":
                    target.mkdir(parents=True)
                    (target / "report.json").write_text(json.dumps({"correct": 1, "total": 2}))
                elif "evaluate-reward" in command:
                    target.write_text(json.dumps({"summary": {"accuracy": 0.5}}))
                return SimpleNamespace(wait=lambda: 0)

            with patch.object(agent_cycle, "ROOT", root), patch.object(agent_cycle, "load_splits", return_value=({}, {})), patch.object(agent_cycle.subprocess, "Popen", side_effect=process), patch("sys.argv", ["cycle", "--initial-data", str(root / "data"), "--name", "smoke", "--rounds", "2", "--steps", "1", "--reward-steps", "1", "--reward-guided"]):
                agent_cycle.main()
            report = json.loads((root / "runs/agent-cycle/smoke/cycle.json").read_text())
            self.assertEqual(report["status"], "completed")
            correction_index = next(i for i, (name, _) in enumerate(modules) if name == "agent_data")
            self.assertTrue(any("evaluate-reward" in command for _, command in modules[:correction_index]))
            command = modules[correction_index][1]
            self.assertIn("--reward-adapter", command)
            self.assertTrue(command[command.index("--reward-adapter") + 1].endswith("smoke-reward-r0"))
            reward_updates = [command for name, command in modules if name == "vision_torch" and "reward" in command and "train" in command]
            self.assertEqual(len(reward_updates), 2)
            self.assertIn("--resume", reward_updates[1])
            self.assertFalse(any("--split" in command and command[command.index("--split") + 1] == "test" for _, command in modules))
