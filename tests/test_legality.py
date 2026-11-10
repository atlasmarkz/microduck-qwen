import unittest

from microduck_qwen.legality import ACTIONS, enforce, prefix, type_legal, violation


def memory(*completed, home=True):
    return {"home_recorded": home, "odometry_relative_home": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0} if home else None,
            "completed": [dict(action) for action in completed]}


EXPLORE_BLUE = {"action": "explore", "color": "blue"}
APPROACH_BLUE = {"action": "approach", "color": "blue", "selector": "nearest"}


class ViolationTests(unittest.TestCase):
    def test_clarify_is_illegal_when_the_instruction_names_a_colour(self):
        for instruction in ("旁边房间里有那个绿色的球儿，过去", "the blue ball is in another room, go next to it"):
            self.assertIsNotNone(violation({"action": "clarify"}, instruction, memory()))

    def test_clarify_stays_legal_for_a_missing_referent(self):
        self.assertIsNone(violation({"action": "clarify"}, "去那边", memory()))
        self.assertIsNone(violation({"action": "clarify"}, "Go over there", memory()))

    def test_without_a_home_clarify_is_legal_and_return_home_is_not(self):
        self.assertIsNone(violation({"action": "clarify"}, "走到红球旁边然后回家", memory(APPROACH_BLUE, home=False)))
        self.assertIsNotNone(violation({"action": "return_home"}, "回家", memory(home=False)))

    def test_clarify_after_a_verified_skill_is_illegal(self):
        self.assertIsNotNone(violation({"action": "clarify"}, "去那边", memory({"action": "move"})))

    def test_done_needs_a_verified_skill(self):
        self.assertIsNotNone(violation({"action": "done"}, "向前走", memory()))
        self.assertIsNone(violation({"action": "done"}, "向前走", memory({"action": "move"})))

    def test_repeating_a_verified_skill_is_illegal_but_a_new_colour_is_not(self):
        self.assertIsNotNone(violation(EXPLORE_BLUE, "找蓝球", memory(EXPLORE_BLUE)))
        self.assertIsNotNone(violation(APPROACH_BLUE, "走到蓝球", memory(APPROACH_BLUE)))
        self.assertIsNone(violation({"action": "approach", "color": "red", "selector": "only"}, "先蓝后红", memory(APPROACH_BLUE)))
        self.assertIsNotNone(violation({"action": "return_home"}, "回家", memory({"action": "return_home"})))

    def test_stop_and_done_are_both_legal_after_an_approach(self):
        for name in ("stop", "done", "return_home"):
            self.assertIsNone(violation({"action": name}, "走到蓝球旁边", memory(APPROACH_BLUE)))

    def test_unsupported_after_a_move_is_left_to_the_model(self):
        self.assertIsNone(violation({"action": "unsupported"}, "向前走然后踢球", memory({"action": "move"})))

    def test_type_level_legality_ignores_colour_specific_repeats(self):
        self.assertTrue(type_legal("approach", "x", memory(APPROACH_BLUE)))
        self.assertFalse(type_legal("done", "x", memory()))


class EnforceTests(unittest.TestCase):
    def setUp(self):
        self.regenerated = []

    def run_enforce(self, result, instruction, mem, scores, outputs):
        def regenerate(name):
            self.regenerated.append(name)
            return outputs[name]
        return enforce(result, instruction, mem, lambda names: {n: scores.get(n, -50.0) for n in names}, regenerate)

    def test_a_legal_decision_is_returned_unchanged_without_scoring(self):
        result = {"raw": '{"action":"done"}', "action": {"action": "done"}}
        out = enforce(result, "向前走", memory({"action": "move"}), lambda names: self.fail("scored"), lambda n: self.fail("regen"))
        self.assertIs(out, result)

    def test_an_illegal_clarify_is_replaced_by_the_best_legal_type(self):
        result = {"raw": '{"action":"clarify"}', "action": {"action": "clarify"}, "latency_s": 1.0}
        out = self.run_enforce(result, "旁边房间里有那个绿色的球儿，过去", memory(),
                               {"clarify": -0.1, "explore": -2.5, "approach": -3.0},
                               {"explore": '{"action":"explore","color":"green"}'})
        self.assertEqual(out["action"], {"action": "explore", "color": "green"})
        self.assertEqual(out["legality"]["chosen"], "explore")
        self.assertTrue(out["legality"]["top_scored_is_original"])
        self.assertEqual(out["legality"]["original_raw"], '{"action":"clarify"}')
        self.assertEqual(out["latency_s"], 1.0)
        self.assertNotIn("error", out)

    def test_an_unparseable_output_is_regenerated(self):
        result = {"raw": '{"action":"explore","colour":"red"}', "error": "unknown action or unexpected fields"}
        out = self.run_enforce(result, "去别处找红球", memory(), {"explore": -0.2},
                               {"explore": '{"action":"explore","color":"red"}'})
        self.assertEqual(out["action"]["action"], "explore")
        self.assertTrue(out["legality"]["violation"].startswith("unparseable"))

    def test_a_regeneration_that_is_still_illegal_falls_through_to_the_next_type(self):
        result = {"raw": "x", "action": dict(APPROACH_BLUE)}
        out = self.run_enforce(result, "走到蓝球", memory(APPROACH_BLUE), {"approach": -0.1, "stop": -1.0},
                               {"approach": '{"action":"approach","color":"blue","selector":"nearest"}', "stop": '{"action":"stop"}'})
        self.assertEqual(self.regenerated, ["approach", "stop"])
        self.assertEqual(out["action"], {"action": "stop"})

    def test_no_legal_regeneration_leaves_an_error_and_no_action(self):
        result = {"raw": '{"action":"clarify"}', "action": {"action": "clarify"}}
        out = self.run_enforce(result, "找蓝球", memory(), {}, {name: "garbage" for name in ACTIONS})
        self.assertNotIn("action", out)
        self.assertIn("no legal regeneration", out["error"])
        self.assertEqual(len(out["legality"]["attempts"]), 3)

    def test_prefix_matches_the_adapter_output_format(self):
        self.assertTrue('{"action":"explore","color":"blue"}'.startswith(prefix("explore")))


if __name__ == "__main__":
    unittest.main()
