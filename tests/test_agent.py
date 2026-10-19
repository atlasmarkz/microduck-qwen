import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microduck_qwen.agent import agent_prompt, adapter_metadata, checked_memory
from microduck_qwen.agent_data import (
    APPROACH_TRUE_DISTANCE_M,
    model_rows,
    load_splits,
    scene_profile,
    tasks,
    teacher_action,
    write_splits,
)
from microduck_qwen.skills import memory_state, parse_action, validate_action
from microduck_qwen.skills import SkillExecutor
from microduck_qwen.agent_eval import semantic_match
from microduck_qwen.sim_data import digest
from microduck_qwen.vision import MODEL, REVISION


class AgentTests(unittest.TestCase):
    def memory(self):
        return {"home_recorded": True, "odometry_relative_home": {"x": 0, "y": 0, "yaw_deg": 0}, "completed": []}

    def test_strict_bounded_schema(self):
        for value in [True, float("nan"), 0.8, -0.1]:
            with self.assertRaises(ValueError):
                validate_action({"action": "move", "direction": "backward", "distance_m": value})
        for raw in ['{"action":"stop","action":"done"}', '{"action":"move","direction":"backward","distance_m":0.2,"code":"x"}', '{"action":"kick"}']:
            with self.assertRaises(ValueError):
                parse_action(raw)
        self.assertEqual(parse_action('{"action":"move","direction":"backward","distance_m":0.2}')["direction"], "backward")

    def test_home_is_never_first_chat_position(self):
        state = {"trunk": [1, 2, 3], "yaw": 0.4}
        self.assertIsNone(memory_state({}, state, [])["odometry_relative_home"])
        memory = memory_state({"home_pose": {"x": 0.1, "y": 0.2, "yaw": 0.4}}, state, [])
        self.assertEqual(memory["odometry_relative_home"], {"x": 0.9, "y": 1.8, "yaw_deg": 0.0})

    def test_input_allowlist_excludes_teacher_and_scene_truth(self):
        row = {"image": "rgb.png", "instruction": "后退0.2米", "memory": self.memory(),
               "expected": {"action": "move", "direction": "backward", "distance_m": 0.2},
               "scene_seed": 987654321, "teacher_specs": "SECRET", "ground_truth": "HIDDEN"}
        value = json.dumps(model_rows([row], Path("/tmp")))
        for secret in ("SECRET", "HIDDEN", "987654321"):
            self.assertNotIn(secret, value)
        with self.assertRaises(ValueError):
            checked_memory(dict(self.memory(), objects="HIDDEN"))

    def test_completed_actions_are_in_prompt(self):
        memory = self.memory()
        memory["completed"] = [{"action": "return_home"}]
        self.assertIn('"completed":[{"action":"return_home"}]', agent_prompt("回到起点", memory))

    def test_approach_verification_allows_the_stop_transient(self):
        # Hard-valid seed 71700 measured 0.3211m inside the control loop but
        # 0.3303m once actually at rest, so a bare stop+tolerance bound rejected
        # a sound demonstration. The allowance has to cover that transient and
        # still reject a stop beside the neighbouring ball of the row.
        self.assertGreater(APPROACH_TRUE_DISTANCE_M, 0.3303)
        self.assertLess(APPROACH_TRUE_DISTANCE_M, 0.45)

    def test_teacher_does_not_replace_occluded_leftmost(self):
        objects = [{"pixel": [0, 0], "status": "partial"}, {"pixel": [100, 0], "status": "visible"}]
        with self.assertRaisesRegex(ValueError, "not confidently visible"):
            teacher_action({"teacher_target": "left"}, objects)

    def test_vision_adapter_cannot_enter_skill_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "adapters.safetensors").write_bytes(b"w")
            (path / "adapter_config.json").write_text("{}")
            (path / "training_report.json").write_text(json.dumps({"format": "vision_sft_v1", "status": "completed",
                "model": MODEL, "revision": REVISION, "adapter_sha256": digest(path / "adapters.safetensors")}))
            with self.assertRaisesRegex(ValueError, "agent adapter"):
                adapter_metadata(path)

    def test_splits_integrity_and_leak_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = {"format": "simulation_agent_v1", "status": "completed", "splits": {}}
            splits = {}
            for index, split in enumerate(("train", "valid", "test")):
                image = path / f"{split}.png"
                image.write_bytes(split.encode())
                splits[split] = [{"id": split, "split": split, "scene_seed": index, "image": image.name,
                    "image_sha256": digest(image), "verified": True, "memory": self.memory(),
                    "expected": {"action": "stop"}, "task_type": "stop"}]
            write_splits(path, manifest, splits)
            load_splits(path)
            splits["valid"][0]["scene_seed"] = 0
            write_splits(path, manifest, splits)
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_splits(path)

    def test_validation_instructions_are_held_out(self):
        for variant in range(3):
            self.assertFalse({t[0] for t in tasks("train", variant)} & {t[0] for t in tasks("valid")})
            self.assertFalse({t[0] for t in tasks("train", variant)} & {t[0] for t in tasks("test")})
        self.assertFalse({t[0] for t in tasks("valid")} & {t[0] for t in tasks("test")})

    def test_format_only_gain_is_reported_separately(self):
        raw = '```json\n{"action":"done"}\n```'
        with self.assertRaises(ValueError):
            parse_action(raw)
        self.assertTrue(semantic_match(raw, {"action": "done"}))
        self.assertFalse(semantic_match(raw, {"action": "return_home"}))

    def test_cancel_is_checked_before_reading_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = SkillExecutor.__new__(SkillExecutor)
            executor.cancel_file = Path(tmp) / "cancel"
            executor.cancel_file.write_text("100")
            executor.started_ns = 1
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                executor.healthy()

    def test_missing_home_rejects_without_relabeling_current_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = SkillExecutor.__new__(SkillExecutor)
            executor.info, executor.directory, executor.index = {}, Path(tmp), 0
            class Robot:
                def stop(self):
                    pass
            executor.robot = Robot()
            result = executor.run({"action": "return_home"})
            self.assertFalse(result["success"])
            self.assertIn("not recorded", result["error"])

    def test_free_text_model_done_is_not_verified_success(self):
        from microduck_qwen import agent
        class Executor:
            def __init__(self, *args, **kwargs):
                self.robot = self
            def stop(self):
                pass
            def close(self):
                pass
            def healthy(self):
                return {"trunk": [0, 0, 0.1], "yaw": 0}
            def run(self, action, candidates):
                return {"success": False, "no_motion": True}
        class Planner:
            def predict(self, *args):
                return {"action": {"action": "done"}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(agent, "ROOT", Path(tmp)), patch.object(agent, "SkillExecutor", Executor), patch.object(agent, "capture", return_value=None), patch.object(agent, "propose_balls", return_value=[]), patch.object(agent, "save_frame", return_value=Path(tmp) / "rgb.png"):
            result = agent.run_instruction(Planner(), "走到蓝球旁边", {"port": 1})
            self.assertEqual(result["status"], "model_done")
            self.assertFalse(result["task_success_verified"])


    def test_hard_scene_mix_covers_occlusion_and_clutter_per_split(self):
        for split, index in (("train", 0), ("valid", 2), ("test", 5)):
            self.assertEqual(scene_profile(split, index, "plain"), ("none", False))
        self.assertEqual([scene_profile("train", i, "hard") for i in range(3)],
                         [("none", True), ("full", True), ("absent", True)])
        self.assertEqual(scene_profile("valid", 0, "hard"), ("partial", True))
        self.assertEqual(scene_profile("test", 0, "hard"), ("full", True))
        for split in ("train", "valid", "test"):
            occlusion, clutter = scene_profile(split, 0, "hard")
            self.assertTrue(occlusion != "none" or clutter, split)

    def test_torch_agent_row_prompt_matches_agent_and_hides_truth(self):
        from microduck_qwen.vision_torch import row_prompt
        row = {"instruction": "后退0.2米", "memory": self.memory(),
               "expected": {"action": "move", "direction": "backward", "distance_m": 0.2},
               "scene_seed": 987654321, "teacher_specs": "SECRET"}
        prompt = row_prompt(row)
        self.assertEqual(prompt, agent_prompt("后退0.2米", self.memory()))
        for secret in ("SECRET", "987654321"):
            self.assertNotIn(secret, prompt)

    def test_torch_row_prompt_prefers_an_explicit_prompt(self):
        from microduck_qwen.vision_torch import row_prompt
        self.assertEqual(row_prompt({"prompt": "已经构造好的提示"}), "已经构造好的提示")


if __name__ == "__main__":
    unittest.main()
