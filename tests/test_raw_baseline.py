import unittest

from microduck_qwen.raw_baseline import ACTIONS, outcome, raw_prompt
from microduck_qwen.truth_score import score_episode


HOME = {"x": 0.0, "y": 0.0, "yaw": 0.0}
SCENE = {"objects": {"ball_red": {"x": 2.0, "y": 0.0, "color": "red", "shape": "sphere"}}}


def step(x, y, action, yaw=0.0):
    return {"memory": {"home_recorded": True, "odometry_relative_home": {"x": x, "y": y, "yaw_deg": yaw}, "completed": []},
            "prediction": {"raw": "", "action": action}}


MOVE = {"action": "move", "direction": "forward", "distance_m": 0.35}


class PromptTests(unittest.TestCase):
    def test_prompt_offers_only_primitives(self):
        prompt = raw_prompt("去红球旁边", {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}, [], 0, 40)
        for name in ("explore", "approach", "return_home", "search", "bypass_return"):
            self.assertNotIn(name, prompt)
        for name in ACTIONS:
            self.assertIn(name, prompt)
        self.assertIn("0/40", prompt)

    def test_outcome_text(self):
        self.assertEqual(outcome({"success": True}), "完成")
        self.assertEqual(outcome({"needs_user_input": True, "error": "NeedsUserInput: unsafe"}), "被拒绝：不安全")


class TruthOnPrimitiveEpisodes(unittest.TestCase):
    specs = [{"teacher_nearest": "red"}]

    def test_reaching_and_ending_is_success(self):
        episode = {"teacher_specs": self.specs, "steps": [step(0, 0, MOVE), step(1.7, 0, {"action": "done"})],
                   "scene_contact_steps_delta": 0}
        self.assertEqual(score_episode(episode, SCENE, HOME)["verdict"], "success")

    def test_budget_exhausted_is_a_fail_not_a_cut(self):
        episode = {"teacher_specs": self.specs, "steps": [step(0, 0, MOVE), step(0.35, 0, MOVE)],
                   "error": "step budget of 40 exhausted without done or stop", "scene_contact_steps_delta": 0}
        self.assertEqual(score_episode(episode, SCENE, HOME)["verdict"], "fail")

    def test_ending_far_from_the_ball_fails(self):
        episode = {"teacher_specs": self.specs, "steps": [step(0, 0, {"action": "done"})], "scene_contact_steps_delta": 0}
        self.assertEqual(score_episode(episode, SCENE, HOME)["verdict"], "fail")

    def test_touching_the_scene_fails(self):
        episode = {"teacher_specs": self.specs, "steps": [step(1.7, 0, {"action": "done"})], "scene_contact_steps_delta": 3}
        self.assertEqual(score_episode(episode, SCENE, HOME)["verdict"], "fail")
