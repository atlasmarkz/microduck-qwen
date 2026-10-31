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
from microduck_qwen.skills import SkillExecutor, NeedsUserInput, bind_target
from microduck_qwen.agent_eval import semantic_match
from microduck_qwen.sim_data import digest
from microduck_qwen.vision import MODEL, REVISION


class AgentTests(unittest.TestCase):
    def test_relative_target_binding_does_not_guess_ids_or_ambiguous_objects(self):
        close = {"id": 3, "color": "yellow", "world_position": [0.8, 0.2, 0.15],
                 "pixel": [100, 40], "visibility": "visible"}
        far = {"id": 1, "color": "yellow", "world_position": [1.7, -0.2, 0.15],
               "pixel": [300, 40], "visibility": "visible"}
        for selector, expected in (("nearest", close), ("leftmost", close), ("rightmost", far)):
            action = validate_action({"action": "approach", "color": "yellow", "selector": selector})
            self.assertEqual(bind_target(action, [far, close], [0, 0]), expected)
        action = {"action": "approach", "color": "yellow", "selector": "only"}
        with self.assertRaises(NeedsUserInput):
            bind_target(action, [far, close], [0, 0])
        self.assertEqual(bind_target(action, [close], [0, 0]), close)
        with self.assertRaises(ValueError):
            validate_action({"action": "approach", "color": "yellow", "id": 1})
        with self.assertRaises(ValueError):
            validate_action(dict(action, id=1))

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

    def test_reward_false_accept_is_overridden_and_returned_as_negative_feedback(self):
        from microduck_qwen import agent_data
        executed = []

        class Executor:
            def __init__(self, *args, **kwargs):
                self.robot = self
                self.scene = {}
            def stop(self):
                pass
            def close(self):
                pass
            def healthy(self):
                return {"trunk": [0, 0, 0.1], "yaw": 0}
            def run(self, action, candidates=None):
                executed.append(action)
                return {"success": True}

        class Planner:
            action = "done"
            def predict(self, *args):
                return {"action": {"action": self.action}, "reward_accepted": True,
                        "proposals": [{"action": {"action": self.action}, "reward_score": 0.99}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(directory, *args):
                image = directory / "input.png"
                image.write_bytes(b"image")
                return image
            with patch.object(agent_data, "SkillExecutor", Executor), patch.object(agent_data, "request", return_value={"steps_with_scene_contact": 0}), patch.object(agent_data, "collection_frame", return_value=(None, None)), patch.object(agent_data, "propose_targets", return_value=[]), patch.object(agent_data, "verify_objects", return_value=[]), patch.object(agent_data, "save_frame", side_effect=save):
                rows, report = agent_data.episode(root, root / "episode", {"port": 1}, "train", 99,
                                                 "停下来", [{"action": "stop"}], Planner(), True)
                accepted = Planner()
                accepted.action = "stop"
                accepted_rows, _ = agent_data.episode(root, root / "accepted", {"port": 1}, "train", 99,
                                                     "停下来", [{"action": "stop"}], accepted, True)
            self.assertTrue(report["verified"])
            self.assertTrue(report["steps"][0]["teacher_intervened"])
            self.assertEqual(executed[-1], {"action": "stop"})
            self.assertEqual(rows[0]["expected"], {"action": "stop"})
            self.assertEqual(rows[0]["reward_proposals"], [{"action": "done"}])
            self.assertFalse(rows[0]["reward_verified_accepted"])
            self.assertTrue(accepted_rows[0]["reward_verified_accepted"])

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
        with tempfile.TemporaryDirectory() as tmp, patch.object(agent, "ROOT", Path(tmp)), patch.object(agent, "SkillExecutor", Executor), patch.object(agent, "capture", return_value=None), patch.object(agent, "propose_targets", return_value=[]), patch.object(agent, "save_frame", return_value=Path(tmp) / "rgb.png"):
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

    def test_equivalent_target_forms_are_one_action_and_other_targets_are_not(self):
        from microduck_qwen.agent_data import same_action
        left = {"id": 4, "color": "red", "world_position": [1.0, 0.4, 0.07], "pixel": [50, 40], "visibility": "visible"}
        mid = {"id": 2, "color": "red", "world_position": [1.0, 0.0, 0.07], "pixel": [150, 40], "visibility": "visible"}
        blue = {"id": 1, "color": "blue", "world_position": [1.0, -0.4, 0.07], "pixel": [250, 40], "visibility": "visible"}
        scene = [mid, left, blue]
        teacher = {"action": "approach", "color": "red", "id": 4}
        leftmost = {"action": "approach", "color": "red", "selector": "leftmost"}
        self.assertTrue(same_action(leftmost, teacher, scene, [0, 0]))
        self.assertTrue(same_action(teacher, teacher, scene, [0, 0]))
        # Same colour, different object; wrong colour; unbindable id: all different actions.
        self.assertFalse(same_action({"action": "approach", "color": "red", "selector": "rightmost"}, teacher, scene, [0, 0]))
        self.assertFalse(same_action({"action": "approach", "color": "blue", "id": 1}, teacher, scene, [0, 0]))
        self.assertFalse(same_action({"action": "approach", "color": "red", "id": 9}, teacher, scene, [0, 0]))
        self.assertFalse(same_action({"action": "approach", "color": "red", "selector": "only"}, teacher, scene, [0, 0]))
        # Non-approach actions stay strictly compared.
        move = {"action": "move", "direction": "forward", "distance_m": 0.2}
        self.assertTrue(same_action(dict(move), move, scene, [0, 0]))
        self.assertFalse(same_action(dict(move, distance_m=0.25), move, scene, [0, 0]))
        self.assertFalse(same_action({"action": "done"}, teacher, scene, [0, 0]))

    def test_layered_scoring_separates_format_type_target_and_motion(self):
        from microduck_qwen.agent_data import score_layers
        ok = {"steps": [{"equivalent_to_teacher": True, "action_type_ok": True, "execution": {}, "physical_ok": True}], "verified": True}
        self.assertEqual(score_layers(ok), dict(format_ok=True, action_type_ok=True, target_ok=True, physical_ok=True, task_ok=True))
        wrong_target = {"steps": [{"target_ok": False, "action_type_ok": True, "execution": {}, "physical_ok": True}], "verified": False}
        layers = score_layers(wrong_target)
        self.assertTrue(layers["action_type_ok"] and layers["physical_ok"])
        self.assertFalse(layers["target_ok"] or layers["task_ok"])
        self.assertFalse(score_layers({"steps": [], "verified": False, "error": "invalid actor JSON"})["format_ok"])

    def test_wilson_interval_brackets_the_rate_and_widens_for_small_samples(self):
        from microduck_qwen.agent_eval import wilson_interval
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))
        low, high = wilson_interval(3, 4)
        self.assertTrue(low < 0.75 < high)
        self.assertGreater(high - low, wilson_interval(30, 40)[1] - wilson_interval(30, 40)[0])
        self.assertEqual(wilson_interval(10, 10)[1], 1.0)
        self.assertLess(wilson_interval(10, 10)[0], 1.0)


if __name__ == "__main__":
    unittest.main()
