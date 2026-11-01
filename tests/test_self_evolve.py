import random
import unittest

from microduck_qwen.agent_curriculum import decide
from microduck_qwen.self_evolve import balance, hindsight_terminal, self_tasks, self_verify

INVENTORY = [{"color": "blue", "x": 1.0, "y": 4.6}, {"color": "red", "x": 1.0, "y": 0.3}]
FIND = {"color": "blue", "tail": "none", "kind": "explore_find"}


def step(action, success=True, **extra):
    result = {"success": success}
    result.update(extra.pop("result", {}))
    return dict({"action": action, "result": result}, **extra)


def good_find(tail_steps=()):
    bound = {"color": "blue", "world_position": [1.05, 4.55, 0.07]}
    return [step({"action": "explore", "color": "blue"}),
            step({"action": "approach", "color": "blue", "selector": "nearest"}, result={"bound_target": bound},
                 final_distance_m=0.32), *tail_steps, step({"action": "done"}, success=False)]


class SelfVerifyTests(unittest.TestCase):
    def test_a_clean_find_and_approach_is_verified(self):
        self.assertEqual(self_verify(FIND, good_find(), 0, INVENTORY), (True, []))

    def test_each_failure_is_named_and_none_leaks_through(self):
        cases = {
            "scene contact": (good_find(), 1),
            "did not end with done": (good_find()[:-1], 0),
            "a skill failed": ([step({"action": "explore", "color": "blue"}, success=False)] + good_find()[1:], 0),
            "approached the wrong colour": ([good_find()[0], step({"action": "approach", "color": "red", "selector": "nearest"},
                                             result={"bound_target": {"color": "red", "world_position": [1.0, 0.3, 0.07]}}, final_distance_m=0.3),
                                             good_find()[-1]], 0),
            "did not end near the bound target": ([good_find()[0], dict(good_find()[1], final_distance_m=1.4), good_find()[-1]], 0),
            "expected exactly one approach": ([good_find()[0], good_find()[-1]], 0),
        }
        for reason, (steps, contact) in cases.items():
            ok, reasons = self_verify(FIND, steps, contact, INVENTORY)
            self.assertFalse(ok, reason)
            self.assertTrue(any(reason in r for r in reasons), (reason, reasons))

    def test_a_target_it_never_perceived_while_exploring_is_rejected(self):
        ok, reasons = self_verify(FIND, good_find(), 0, [{"color": "blue", "x": -3.0, "y": -3.0}])
        self.assertFalse(ok)
        self.assertIn("match anything", " ".join(reasons))

    def test_exploring_is_required_for_find_tasks_but_not_plain_approaches(self):
        no_explore = good_find()[1:]
        self.assertFalse(self_verify(FIND, no_explore, 0, INVENTORY)[0])
        plain = dict(FIND, kind="colour_approach", color="red")
        bound = {"color": "red", "world_position": [1.0, 0.35, 0.07]}
        steps = [step({"action": "approach", "color": "red", "id": 3}, result={"bound_target": bound}, final_distance_m=0.31),
                 step({"action": "done"}, success=False)]
        self.assertEqual(self_verify(plain, steps, 0, INVENTORY), (True, []))

    def test_home_tail_needs_a_real_return_and_none_tail_forbids_one(self):
        home = dict(FIND, tail="home")
        back = step({"action": "return_home"}, result={"distance_home_m": 0.09, "home_yaw_error_deg": 4.0})
        far = step({"action": "return_home"}, result={"distance_home_m": 0.5, "home_yaw_error_deg": 4.0})
        self.assertTrue(self_verify(home, good_find([back]), 0, INVENTORY)[0])
        self.assertFalse(self_verify(home, good_find([far]), 0, INVENTORY)[0])
        self.assertFalse(self_verify(home, good_find(), 0, INVENTORY)[0])
        self.assertFalse(self_verify(FIND, good_find([back]), 0, INVENTORY)[0])

    def test_stop_tail_must_end_with_stop(self):
        stop = dict(FIND, tail="stop")
        steps = good_find()[:-1] + [step({"action": "stop"})]
        self.assertTrue(self_verify(stop, steps, 0, INVENTORY)[0])
        self.assertFalse(self_verify(stop, good_find(), 0, INVENTORY)[0])


class SelfTaskTests(unittest.TestCase):
    def test_hidden_colours_become_find_tasks_and_visible_ones_plain_approaches(self):
        tasks = self_tasks(INVENTORY, {"red"}, random.Random(0), 1)
        kinds = {meta["color"]: meta["kind"] for _, meta in tasks}
        self.assertEqual(kinds, {"blue": "explore_find", "red": "colour_approach"})

    def test_nothing_perceived_means_nothing_to_pose(self):
        self.assertEqual(self_tasks([], {"red"}, random.Random(0), 2), [])


class HindsightAndBalanceTests(unittest.TestCase):
    def rows_for(self, steps):
        return [{"expected": s["action"]} for s in steps]

    def test_only_a_wrong_closing_action_is_relabelled(self):
        steps = good_find()[:-1] + [step({"action": "stop"})]
        rows = self.rows_for(steps)
        ok, reasons = self_verify(FIND, steps, 0, INVENTORY)
        self.assertFalse(ok)
        self.assertTrue(hindsight_terminal(FIND, steps, reasons, rows))
        self.assertEqual(rows[-1]["expected"], {"action": "done"})
        self.assertEqual(rows[0]["expected"], {"action": "explore", "color": "blue"})

    def test_other_failures_and_non_terminal_endings_are_never_relabelled(self):
        steps = good_find()[:-1]  # ended on a skill, not on a closing action
        ok, reasons = self_verify(FIND, steps, 0, INVENTORY)
        self.assertFalse(hindsight_terminal(FIND, steps, reasons, self.rows_for(steps)))
        broken = [step({"action": "explore", "color": "blue"}, success=False)] + good_find()[1:-1] + [step({"action": "stop"})]
        ok, reasons = self_verify(FIND, broken, 0, INVENTORY)
        self.assertFalse(hindsight_terminal(FIND, broken, reasons, self.rows_for(broken)))

    def test_balance_caps_the_easy_kind(self):
        rows = [{"task_meta": {"kind": "colour_approach"}, "n": i} for i in range(30)] +                [{"task_meta": {"kind": "explore_find"}, "n": i} for i in range(4)]
        kept = balance(rows)
        self.assertEqual(sum(r["task_meta"]["kind"] == "explore_find" for r in kept), 4)
        self.assertEqual(sum(r["task_meta"]["kind"] == "colour_approach" for r in kept), 8)

    def test_noise_slack_tolerates_small_drops_only(self):
        self.assertTrue(decide((30, 36), (29, 36), {}, {}, 0.25, slack=0.05)[0])
        self.assertFalse(decide((30, 36), (26, 36), {}, {}, 0.25, slack=0.05)[0])


if __name__ == "__main__":
    unittest.main()
