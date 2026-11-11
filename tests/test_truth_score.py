import unittest

from microduck_qwen.truth_score import requirements, score_episode, summary

HOME = {"x": 0.0, "y": 0.0, "yaw": 0.0}
SCENE = {"objects": {
    "ball_0": {"color": "green", "x": 3.0, "y": 0.0, "shape": "sphere"},
    "ball_1": {"color": "red", "x": 1.0, "y": 0.5, "shape": "sphere"},
    "ball_2": {"color": "blue", "x": 1.0, "y": -0.5, "shape": "sphere"},
}}


def step(x, y, action, expected=None, execution=None, yaw_deg=0.0):
    out = {"memory": {"home_recorded": True, "odometry_relative_home": {"x": x, "y": y, "yaw_deg": yaw_deg}, "completed": []},
           "prediction": {"action": {"action": action, **({"color": "green"} if action in ("explore", "approach") else {})}},
           "expected": {"action": expected or action}}
    if execution is not None:
        out["execution"] = execution
    return out


def episode(specs, steps, error=None, contact=0):
    return {"teacher_specs": specs, "steps": steps, "error": error, "scene_contact_steps_delta": contact}


GREEN = [{"action": "explore", "color": "green"}, {"teacher_nearest": "green"}, {"action": "stop"}]
GREEN_HOME = [{"action": "explore", "color": "green"}, {"teacher_nearest": "green"}, {"action": "return_home"}]


class TruthScoreTests(unittest.TestCase):
    def verdict(self, ep, first=None):
        return score_episode(ep, SCENE, HOME, first)["verdict"]

    def test_done_for_stop_next_to_the_ball_is_a_success(self):
        ep = episode(GREEN, [step(0, 0, "explore"), step(2.0, 0, "approach"), step(2.7, 0, "done", expected="stop")],
                     error="failed physical skill or wrong task action (scored AFTER execution)")
        self.assertEqual(self.verdict(ep), "success")

    def test_ending_far_from_the_ball_fails(self):
        ep = episode(GREEN, [step(0, 0, "explore"), step(2.0, 0, "approach"), step(2.0, 0, "stop")])
        self.assertEqual(self.verdict(ep), "fail")

    def test_return_requires_reaching_first_and_ending_home(self):
        ok = [step(0, 0, "explore"), step(2.0, 0, "approach"), step(2.7, 0, "return_home"), step(0.1, 0.05, "done")]
        self.assertEqual(self.verdict(episode(GREEN_HOME, ok)), "success")
        far = ok[:3] + [step(0.4, 0, "done")]
        self.assertEqual(self.verdict(episode(GREEN_HOME, far)), "fail")
        never = [step(0, 0, "explore"), step(2.0, 0, "return_home"), step(0, 0, "done")]
        self.assertEqual(self.verdict(episode(GREEN_HOME, never)), "fail")

    def test_clarify_or_unsupported_ends_in_failure(self):
        self.assertEqual(self.verdict(episode(GREEN, [step(0, 0, "clarify", expected="explore")])), "fail")

    def test_touching_the_scene_fails(self):
        ep = episode(GREEN, [step(0, 0, "explore"), step(2.0, 0, "approach"), step(2.7, 0, "stop")], contact=3)
        self.assertEqual(self.verdict(ep), "fail")

    def test_teacher_refusal_and_model_deviation_are_cuts(self):
        refused = episode(GREEN, [step(0, 0, "explore")], error="ValueError: no confidently visible green ball to approach")
        self.assertEqual(self.verdict(refused), "cut_teacher")
        deviated = episode(GREEN, [step(0, 0, "search", expected="explore")],
                           error="failed physical skill or wrong task action (scored AFTER execution)")
        self.assertEqual(self.verdict(deviated), "cut_deviation")

    def test_a_failed_skill_that_was_the_right_choice_is_a_failure(self):
        ep = episode(GREEN, [step(0, 0, "explore")], error="failed physical skill or wrong task action (scored AFTER execution)")
        self.assertEqual(self.verdict(ep), "fail")

    def test_no_steps_or_failed_reset_is_infra(self):
        self.assertEqual(self.verdict(episode(GREEN, [], error="RuntimeError: cannot restore task start")), "cut_infra")

    def test_move_within_the_executor_tolerance(self):
        specs = [{"action": "move", "direction": "forward", "distance_m": 0.2}]
        self.assertEqual(self.verdict(episode(specs, [step(0, 0, "move"), step(0.18, 0.01, "done")])), "success")
        self.assertEqual(self.verdict(episode(specs, [step(0, 0, "move"), step(0.10, 0.0, "done")])), "fail")
        self.assertEqual(self.verdict(episode(specs, [step(0, 0, "move"), step(0.2, 0.0, "unsupported")])), "fail")

    def test_leftmost_target_comes_from_the_first_frame_segmentation(self):
        first = {"objects": {"ball_1": {"visible_pixels": 40, "pixel": [100, 50]},
                             "ball_2": {"visible_pixels": 40, "pixel": [400, 50]},
                             "ball_0": {"visible_pixels": 0, "pixel": None}}}
        specs = [{"teacher_target": "left"}, {"action": "return_home"}]
        steps = [step(0, 0, "approach"), step(0.75, 0.45, "return_home"), step(0.05, 0.0, "done")]
        self.assertEqual(self.verdict(episode(specs, steps), first), "success")
        wrong = [step(0, 0, "approach"), step(0.75, -0.45, "return_home"), step(0.05, 0.0, "done")]
        self.assertEqual(self.verdict(episode(specs, wrong), first), "fail")

    def test_requirements_read_the_task_definition(self):
        self.assertEqual(requirements(GREEN_HOME), {"reach": ("colour", "green"), "move": None, "home": True})

    def test_summary_bounds(self):
        s = summary({1: {"verdict": "success", "strict": False}, 2: {"verdict": "cut_teacher", "strict": False},
                     3: {"verdict": "fail", "strict": False}})
        self.assertEqual((s["truth"], s["truth_upper"], s["strict"]), (1, 2, 0))


if __name__ == "__main__":
    unittest.main()
