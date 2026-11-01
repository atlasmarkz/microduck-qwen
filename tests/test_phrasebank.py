import random
import unittest

from microduck_qwen.agent_language import COLORS, explore_phrasings, explore_tasks, phrasings
from microduck_qwen.phrasebank import COLOUR_CHAR, ROOM_CUES, SPLITS, bank, split_pool


class PhraseBankTests(unittest.TestCase):
    def test_every_pool_gives_each_split_at_least_one_item_and_no_item_twice(self):
        from microduck_qwen import phrasebank as p
        for name in ("OPENINGS", "APPROACH", "EXPLORE", "ROOMS", "COLOUR_FORMS", "NOUNS", "TAIL_HOME", "TAIL_STOP",
                     "EN_APPROACH", "EN_EXPLORE", "EN_HOME", "EN_STOP"):
            pool = getattr(p, name)
            for split in SPLITS:
                self.assertTrue(pool[split], (name, split))
            everything = sum(pool.values(), [])
            self.assertEqual(len(everything), len(set(everything)), name)
        self.assertEqual(split_pool(list("abcdefgh")), {"train": list("abcdefh")[:5] + ["h"], "valid": ["f"], "test": ["g"]})

    def test_sentences_never_leak_between_splits(self):
        for color in COLORS:
            for kind in ("approach", "explore"):
                pools = {s: {t for t, _ in bank(s, color, kind)} for s in SPLITS}
                self.assertFalse(pools["train"] & pools["valid"], (color, kind))
                self.assertFalse(pools["train"] & pools["test"], (color, kind))
                self.assertFalse(pools["valid"] & pools["test"], (color, kind))
            for fn in (phrasings, explore_phrasings):
                args = (random.Random(0),) if fn is phrasings else ()
                texts = {s: {t for t, _ in fn(s, color, *args)} for s in SPLITS}
                self.assertFalse(texts["train"] & texts["test"])
                self.assertFalse(texts["train"] & texts["valid"])
                self.assertFalse(texts["valid"] & texts["test"])

    def test_only_explore_sentences_carry_a_room_cue(self):
        for split in SPLITS:
            for text, _ in bank(split, "blue", "approach"):
                self.assertFalse(any(cue in text for cue in ROOM_CUES), text)
                self.assertFalse("another room" in text or "different room" in text or "other room" in text, text)
            for text, _ in bank(split, "blue", "explore"):
                chinese = any(cue in text for cue in ROOM_CUES) or "这里没有" in text
                english = "room" in text or "leave this room" in text
                self.assertTrue(chinese or english, text)

    def test_every_sentence_names_the_colour_and_the_pool_is_varied(self):
        for color in COLORS:
            for split in SPLITS:
                texts = bank(split, color, "approach")
                self.assertGreater(len(texts), 40)
                for text, _ in texts:
                    self.assertTrue(COLOUR_CHAR[color] in text or color in text, text)
        self.assertEqual(bank("train", "red", "explore"), bank("train", "red", "explore"))  # deterministic

    def test_explore_tasks_use_the_wide_pool_and_keep_their_specs(self):
        tasks = explore_tasks("train", {"yellow"}, random.Random(3), per_color=8)
        self.assertEqual(len(tasks), 8)
        self.assertGreater(len({t for t, _, _ in tasks}), 4)
        for text, specs, meta in tasks:
            self.assertEqual(specs[0], {"action": "explore", "color": "yellow"})
            self.assertEqual(meta["kind"], "explore_find")


if __name__ == "__main__":
    unittest.main()
