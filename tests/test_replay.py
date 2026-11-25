import json
import unittest

from microduck_qwen import ROOT, replay


def scores(rows):
    """rows: {(seed, task): (raw, frozen, frozterm, lora)} verdicts -> load_scores() shape."""
    order = [arm[0] for arm in replay.ARMS]
    return {name: {key: {"verdict": verdicts[i], "reason": "", "instruction": "x"} for key, verdicts in rows.items()}
            for i, name in enumerate(order)}


class SelectionTests(unittest.TestCase):
    def test_rule_covers_repeated_patterns_then_other_task_types_then_failures_of_the_full_system(self):
        common, second = ("fail", "cut_deviation", "success", "success"), ("fail", "success", "success", "success")
        rows = {(1, 3): common, (1, 4): common, (2, 4): common, (1, 200): second, (2, 200): second,
                (3, 200): ("fail", "fail", "fail", "fail"), (4, 200): ("fail", "fail", "success", "success")}
        chosen, keys = replay.select(scores(rows))
        self.assertEqual(len(keys), 7)
        self.assertEqual({item["key"] for item in chosen}, {(1, 3), (1, 4), (1, 200), (3, 200)})
        self.assertEqual(next(i for i in chosen if i["key"] == (1, 3))["count"], 3)

    def test_both_kinds_of_cut_are_one_outcome(self):
        self.assertEqual({replay.outcome(v) for v in ("cut_deviation", "cut_teacher")}, {replay.CUT})

    def test_action_text_is_readable(self):
        self.assertEqual(replay.action_text({"action": {"action": "explore", "color": "blue"}}), "探索找蓝球")
        self.assertEqual(replay.action_text({"action": {"action": "move", "direction": "backward", "distance_m": 0.2}}), "向后移动 20 厘米")
        self.assertEqual(replay.action_text({"error": "bad"}), "输出无法执行")


class PublishedReplayTests(unittest.TestCase):
    def setUp(self):
        self.index = json.loads((ROOT / "assets/replay/index.json").read_text(encoding="utf-8"))

    def test_every_clip_is_published(self):
        for clip in self.index["clips"]:
            self.assertTrue((ROOT / "assets/replay" / clip["file"]).is_file(), clip["file"])

    def test_totals_add_up(self):
        for counts in self.index["totals"].values():
            self.assertEqual(sum(counts.values()), self.index["tasks"])

    def test_notebook_cell_matches_the_index(self):
        notebook = json.loads(replay.NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(c for c in notebook["cells"] if c["id"] == replay.CELL)
        self.assertEqual("".join(cell["source"]), replay.markdown(self.index))


if __name__ == "__main__":
    unittest.main()
