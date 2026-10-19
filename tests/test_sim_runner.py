import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microduck_qwen.sim_runner import start_scene


class FakeProcess:
    """Stands in for the child simulator: writes the manifest, never exits."""

    def __init__(self):
        self.returncode = None

    def poll(self):
        return None


def command_for(tmp, **kwargs):
    commands = []

    def fake_popen(command, **ignore):
        commands.append(command)
        Path(command[command.index("--manifest") + 1]).write_text(json.dumps({"port": 1}))
        return FakeProcess()

    with tempfile.TemporaryDirectory() as case, patch("microduck_qwen.sim_runner.subprocess.Popen", side_effect=fake_popen):
        _, log, info, failures = start_scene(Path(case), 7, "full", 7862, **kwargs)
        log.close()
    return commands[0], info, failures


class SimRunnerTests(unittest.TestCase):
    def test_hard_scene_flags_reach_the_simulator_command(self):
        command, info, failures = command_for("hard", collect_gt=True, clutter=True)
        self.assertIn("--clutter", command)
        self.assertIn("--collect-gt", command)
        self.assertEqual(command[command.index("--occlusion") + 1], "full")
        self.assertEqual(info, {"port": 1})
        self.assertEqual(failures, [])

    def test_plain_scene_omits_the_clutter_flag(self):
        command, _, _ = command_for("plain")
        self.assertNotIn("--clutter", command)
        self.assertNotIn("--collect-gt", command)


if __name__ == "__main__":
    unittest.main()
