import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from microduck_qwen import ROOT, skill_showroom, upstream_tour


def notebook_cells():
    notebook = json.loads((ROOT / "notebooks/microduck-baseline.ipynb").read_text(encoding="utf-8"))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def form_options(source):
    return json.loads(re.search(r"# @param (\[.*\])", source).group(1))


class UpstreamTourTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = upstream_tour.load()
        self.base = [task for task in self.snapshot["tasks"] if "base" not in task]

    def test_every_registered_task_belongs_to_exactly_one_family(self):
        listed = []
        for *_text, flat_id, _policies, _bundle in upstream_tour.FAMILIES.values():
            base, backlash = upstream_tour.variants(self.snapshot, flat_id)
            listed += base + backlash
        self.assertEqual(sorted(listed), sorted(task["id"] for task in self.snapshot["tasks"]))

    def test_chinese_wording_covers_the_snapshot_and_nothing_else(self):
        used = {term["func"] for task in self.base for term in task["rewards"]}
        self.assertEqual(used, set(upstream_tour.REWARD_ZH))
        self.assertEqual({task["robot_xml"] for task in self.base}, set(upstream_tour.ROBOT_ZH))
        self.assertEqual({cfg["type"] for task in self.base for cfg in task["commands"].values()}, set(upstream_tour.COMMAND_ZH))
        for task in self.base:
            self.assertEqual(task["actor_obs"], list(upstream_tour.OBS_ZH))

    def test_pages_render(self):
        self.assertIn("<b>33</b>", upstream_tour.task_table_html())
        for key in upstream_tour.FAMILIES:
            page = upstream_tour.family_html(key)
            self.assertIn(upstream_tour.FAMILIES[key][3], page)

    def test_snapshot_matches_the_commit_the_notebook_downloads(self):
        self.assertIn(f'upstream_rev = "{self.snapshot["revision"]}"', notebook_cells()["baseline-assets"])

    def test_notebook_forms_list_every_family_and_skill(self):
        cells = notebook_cells()
        self.assertEqual([o.split(" · ")[0] for o in form_options(cells["upstream-family"])], list(upstream_tour.FAMILIES))
        self.assertEqual([o.split(" · ")[0] for o in form_options(cells["upstream-skills"])], [*skill_showroom.SKILLS, "all"])

    def test_every_published_policy_is_demonstrated(self):
        published = {name for family in upstream_tour.FAMILIES.values() for name in family[4]}
        shown = {s.policy for skill in skill_showroom.SKILLS.values() for s in skill.segments if s.main}
        self.assertEqual(published, shown)

    def test_commands_fill_the_thirteen_slots(self):
        for skill in skill_showroom.SKILLS.values():
            for segment in skill.segments:
                for t in (0.0, segment.seconds / 2, segment.seconds):
                    self.assertEqual(len(segment.command(t)), 13)

    def test_measurement_reports_a_failed_rise_and_a_fallen_end(self):
        skill = skill_showroom.SKILLS["velstand"]
        row = {"main": True, "caption": "前进", "xyz": [0.0, 0.0, 0.05], "yaw": 0.0, "gz": 0.0, "gyro_y": 0.0, "mouth": 0.0, "ball": None}
        result = skill_showroom.measure(skill, [dict(row) for _ in range(100)])
        self.assertIsNone(result["起身用时_s"])
        self.assertFalse(result["结束时直立"])
        log = [dict(row) for _ in range(50)] + [dict(row, gz=-1.0) for _ in range(50)]
        self.assertTrue(math.isclose(skill_showroom.measure(skill, log)["起身用时_s"], 1.0))

    def test_skill_page_embeds_the_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = skill_showroom.save_clip([Image.new("RGB", (48, 36), c) for c in ("red", "blue", "green")], Path(tmp) / "x.webp")
            with Image.open(clip) as picture:
                self.assertEqual(picture.n_frames, 3)
            page = upstream_tour.skill_html({"skill": "walk", "clip": str(clip), "metrics": {"结束时直立": True, "起身用时_s": None}})
        self.assertIn("data:image/webp;base64,", page)
        self.assertIn("未达到", page)


if __name__ == "__main__":
    unittest.main()
