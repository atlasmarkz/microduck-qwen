import random
import unittest

from microduck_qwen.agent_data import teacher_action
from microduck_qwen.agent_language import COLORS, colour_tasks, paraphrases, phrasings
from microduck_qwen.skills import NeedsUserInput, bind_target, validate_action
from microduck_qwen.vision_scene import make_scene


class LanguageGrammarTests(unittest.TestCase):
    def test_heldout_surface_forms_never_appear_in_train(self):
        rng = random.Random(0)
        for color in COLORS:
            train = {t for t, _ in phrasings("train", color, rng)}
            for split in ("valid", "test"):
                self.assertFalse(train & {t for t, _ in phrasings(split, color, rng)}, (split, color))

    def test_holdout_color_is_only_in_heldout_splits(self):
        present = ["red", "blue", "yellow"]
        train = colour_tasks("train", present, random.Random(1), holdout_color="yellow")
        self.assertEqual({m["color"] for _, _, m in train}, {"red", "blue"})
        for split in ("valid", "test"):
            tasks = colour_tasks(split, present, random.Random(1), holdout_color="yellow")
            self.assertEqual({m["color"] for _, _, m in tasks}, {"yellow"})

    def test_tail_decides_the_second_action(self):
        for text, specs, meta in colour_tasks("train", ["red"], random.Random(3), per_color=20):
            expected = {"none": 1, "home": 2, "stop": 2}[meta["tail"]]
            self.assertEqual(len(specs), expected, text)
            self.assertEqual(specs[-1].get("action"), {"home": "return_home", "stop": "stop"}.get(meta["tail"], specs[-1].get("action")))

    def test_paraphrases_keep_the_tail_kind(self):
        text, _, meta = colour_tasks("train", ["blue"], random.Random(5), per_color=1)[0]
        for other in paraphrases("blue", meta["tail"], text):
            self.assertNotEqual(other, text)
            self.assertEqual(("home" in meta["tail"]), ("回到" in other or "返回" in other))

    def test_varied_scene_draws_four_colors_and_default_is_unchanged(self):
        colors = {o["color"] for seed in range(40) for o in make_scene(seed, varied=True)["objects"].values()}
        self.assertEqual(colors, set(COLORS))
        self.assertEqual({o["color"] for o in make_scene(42)["objects"].values()} | {"yellow"}, set(COLORS))
        self.assertEqual({o["color"] for o in make_scene(42)["objects"].values()}, {"red", "blue", "green"})

    def test_yellow_ball_label_uses_kind_selector_not_a_guessed_id(self):
        candidate = {"id": 2, "visibility": "visible"}
        verified = [{"name": "ball_0", "color": "yellow", "status": "visible", "candidate": candidate},
                    {"name": "ball_1", "color": "red", "status": "visible", "candidate": {"id": 1, "visibility": "visible"}}]
        self.assertEqual(teacher_action({"teacher_color": "yellow"}, verified),
                         {"action": "approach", "color": "yellow", "kind": "ball", "selector": "only"})
        self.assertEqual(teacher_action({"teacher_color": "red"}, verified), {"action": "approach", "color": "red", "id": 1})
        verified[1]["status"] = "partial"
        with self.assertRaises(ValueError):
            teacher_action({"teacher_color": "red"}, verified)

    def test_kind_validation_and_binding_ignore_the_other_yellow_object(self):
        validate_action({"action": "approach", "color": "yellow", "kind": "ball", "selector": "only"})
        with self.assertRaises(ValueError):
            validate_action({"action": "approach", "color": "yellow", "kind": "cube", "selector": "only"})
        with self.assertRaises(ValueError):
            validate_action({"action": "approach", "color": "yellow", "kind": "ball", "id": 1})  # yellow still needs a selector
        ball = {"id": 1, "color": "yellow", "kind": "ball", "visibility": "visible", "world_position": [1, 0, 0.07], "pixel": [300, 150]}
        pillar = {"id": 2, "color": "yellow", "kind": "pillar", "visibility": "visible", "world_position": [0.5, 0.7, 0.15], "pixel": [100, 150]}
        action = {"action": "approach", "color": "yellow", "kind": "ball", "selector": "only"}
        self.assertIs(bind_target(action, [ball, pillar], [0, 0]), ball)
        with self.assertRaises(NeedsUserInput):
            bind_target({"action": "approach", "color": "yellow", "selector": "only"}, [ball, pillar], [0, 0])


if __name__ == "__main__":
    unittest.main()
