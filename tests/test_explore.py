import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

from microduck_qwen.agent import agent_prompt
from microduck_qwen.agent_data import teacher_action
from microduck_qwen.agent_language import explore_paraphrases, explore_tasks
from microduck_qwen.explore import NARROW_INFLATE_M, TIGHTEST_INFLATE_M, Explorer, clearance_tiers, navigator_pad
from microduck_qwen.mapping import FREE, OCCUPIED, UNKNOWN, OccupancyMap
from microduck_qwen.skills import validate_action
from microduck_qwen.vision import Frame, unproject
from microduck_qwen.vision_world import make_world


def wall_points(x=2.0, half=1.0, z=0.2, count=41):
    return np.array([[x, y, z] for y in np.linspace(-half, half, count)])


class MappingTests(unittest.TestCase):
    def test_rays_carve_free_space_mark_the_hit_and_leave_the_rest_unknown(self):
        m = OccupancyMap()
        m.update_from_points((0.0, 0.0), wall_points())
        cell = lambda x, y: m.grid[tuple(int(v) for v in m.index(x, y))]
        self.assertEqual(cell(2.0, 0.0), OCCUPIED)
        self.assertEqual(cell(1.0, 0.0), FREE)
        self.assertEqual(cell(3.0, 0.0), UNKNOWN)  # behind the wall is never assumed free
        self.assertEqual(cell(1.0, 3.0), UNKNOWN)

    def test_floor_hits_are_free_and_obstacles_stay_occupied(self):
        m = OccupancyMap()
        m.update_from_points((0.0, 0.0), np.array([[1.5, 0.0, 0.0]]))
        self.assertEqual(m.grid[tuple(int(v) for v in m.index(1.5, 0.0))], FREE)
        m.update_from_points((0.0, 0.0), wall_points(x=1.5, half=0.2, count=5))
        m.update_from_points((0.0, 0.0), np.array([[3.0, 0.0, 0.0]]))  # a later floor ray does not erase it
        self.assertEqual(m.grid[tuple(int(v) for v in m.index(1.5, 0.0))], OCCUPIED)

    def test_open_rays_carve_free_space_without_marking_anything(self):
        m = OccupancyMap()
        m.update_from_points((0.0, 0.0), np.full((1, 3), np.nan), open_ends=[[3.0, 0.0]])
        self.assertEqual(m.grid[tuple(int(v) for v in m.index(2.0, 0.0))], FREE)
        self.assertFalse((m.grid == OCCUPIED).any())

    def test_plan_ends_on_a_frontier_and_never_crosses_unknown_or_the_wall(self):
        m = OccupancyMap()
        m.update_from_points((0.0, 0.0), wall_points(), open_ends=[[3.5, y] for y in np.linspace(-2.5, 2.5, 30)])
        path = m.plan_to_frontier((0.0, 0.0))
        self.assertIsNotNone(path)
        blocked = m.blocked()
        for x, y in path:
            self.assertFalse(blocked[tuple(int(v) for v in m.index(x, y))], (x, y))
        # Nothing known to explore: a closed, fully scanned pocket has no frontier.
        self.assertIsNone(OccupancyMap().plan_to_frontier((0.0, 0.0)))

    def test_avoided_frontiers_are_not_chosen_again(self):
        m = OccupancyMap()
        m.update_from_points((0.0, 0.0), wall_points(), open_ends=[[3.5, y] for y in np.linspace(-2.5, 2.5, 30)])
        first = m.plan_to_frontier((0.0, 0.0))[-1]
        again = m.plan_to_frontier((0.0, 0.0), avoid=[first])
        self.assertTrue(again is None or np.hypot(again[-1][0] - first[0], again[-1][1] - first[1]) >= 0.6)

    def test_frame_unprojection_matches_vision_unproject(self):
        meta = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 90.0, "camera_position": [0.0, 0.0, 0.12],
                "camera_rotation": [[0, 0.0998, -0.995], [-1, 0, 0], [0, 0.995, 0.0998]]}
        depth = np.full((180, 320), 2.0, dtype="<f4")
        frame = Frame(np.zeros((180, 320, 3), np.uint8), depth, meta)
        m = OccupancyMap()
        m.update_from_frame(frame, row_stride=60, column_stride=80)
        expected = unproject(40, 30, 2.0, meta)  # first sampled pixel (u=40, v=30)
        ix, iy = (int(v) for v in m.index(expected[0], expected[1]))
        nearby = m.grid[ix - 1:ix + 2, iy - 1:iy + 2]  # the mark is nudged a quarter cell past the surface
        self.assertTrue(0.03 <= expected[2] <= 0.8)
        self.assertTrue((nearby == OCCUPIED).any())


class ExploreSkillTests(unittest.TestCase):
    def test_schema_accepts_explore_and_rejects_extra_fields(self):
        self.assertEqual(validate_action({"action": "explore", "color": "blue"})["color"], "blue")
        for bad in ({"action": "explore"}, {"action": "explore", "color": "pink"},
                    {"action": "explore", "color": "red", "selector": "nearest"}):
            with self.assertRaises(ValueError):
                validate_action(bad)

    def test_prompt_names_explore_and_gives_feedback_after_it(self):
        memory = {"home_recorded": True, "odometry_relative_home": {"x": 0, "y": 0, "yaw_deg": 0},
                  "completed": [{"action": "explore", "color": "blue"}]}
        prompt = agent_prompt("去找蓝色球", memory)
        self.assertIn('"action":"explore"', prompt)
        self.assertIn("不要重复explore", prompt)

    def test_explore_tasks_only_cover_hidden_colours_and_end_with_the_right_tail(self):
        tasks = explore_tasks("train", {"blue"}, random.Random(0), per_color=3)
        self.assertEqual(len(tasks), 3)
        for text, specs, meta in tasks:
            self.assertEqual(specs[0], {"action": "explore", "color": "blue"})
            self.assertEqual(specs[1], {"teacher_nearest": "blue"})
            self.assertEqual(meta["kind"], "explore_find")
            self.assertEqual(len(specs) == 3, meta["tail"] != "none")
            self.assertTrue("蓝" in text or "blue" in text, text)
        self.assertEqual(explore_tasks("train", set(), random.Random(0)), [])
        # Held-out phrasing never reuses a training sentence.
        train = {t for t, _, _ in explore_tasks("train", {"red"}, random.Random(1), per_color=99)}
        test = {t for t, _, _ in explore_tasks("test", {"red"}, random.Random(1), per_color=99)}
        self.assertFalse(train & test)
        self.assertTrue(all(t != "" for t in explore_paraphrases("blue", "none", tasks[0][0], rng=random.Random(2))))

    def test_teacher_approaches_the_nearest_confidently_visible_ball_of_the_colour(self):
        visible = {"name": "ball_4", "color": "blue", "status": "visible", "candidate": {"visibility": "visible"}}
        self.assertEqual(teacher_action({"teacher_nearest": "blue"}, [visible]),
                         {"action": "approach", "color": "blue", "selector": "nearest"})
        self.assertEqual(teacher_action({"teacher_nearest": "yellow"}, [dict(visible, color="yellow")])["kind"], "ball")
        for hidden in (dict(visible, status="partial", candidate=None), dict(visible, color="red")):
            with self.assertRaises(ValueError):
                teacher_action({"teacher_nearest": "blue"}, [hidden])


class WorldSearchTests(unittest.TestCase):
    def test_the_colour_missing_from_the_start_row_exists_once_in_another_room(self):
        for seed in range(30):
            world = make_world(seed, (2, 2), 1.5)
            balls = world["objects"]
            row = {balls[f"ball_{i}"]["color"] for i in range(3)}
            missing = ({"red", "blue", "green", "yellow"} - row).pop()
            holders = [b for b in balls.values() if b["color"] == missing]
            self.assertEqual(len(holders), 1, seed)
            self.assertNotEqual(holders[0]["room"], world["start_room"], seed)

    def test_single_room_worlds_have_no_hidden_colour(self):
        world = make_world(3, (1, 1), 0.0)
        self.assertEqual(len({b["color"] for b in world["objects"].values()}), 3)


if __name__ == "__main__":
    unittest.main()


class RecoveryTests(unittest.TestCase):
    def gap_map(self, gap=(0.0, 0.2)):
        """A closed, known free pocket whose only way on is a gap (0.2 m by default): too narrow at normal clearance."""
        m = OccupancyMap()

        def fill(x0, x1, y0, y1, value):
            for a in range(int(m.index(x0, 0)[0]), int(m.index(x1, 0)[0])):
                for b in range(int(m.index(0, y0)[1]), int(m.index(0, y1)[1])):
                    m.grid[a, b] = value

        fill(-1.2, 1.55, -1.7, 1.7, OCCUPIED)   # pocket walls (outer ring)
        fill(-1.0, 1.55, -1.5, 1.5, FREE)
        fill(1.55, 1.75, -1.5, 1.5, OCCUPIED)
        fill(1.55, 1.75, gap[0], gap[1], FREE)  # the gap
        fill(1.75, 2.2, -0.3, 0.5, FREE)        # known floor beyond it, unknown all around
        return m

    def test_a_gap_too_narrow_for_normal_clearance_leaves_no_reachable_frontier(self):
        m = self.gap_map()
        self.assertIsNone(m.plan_to_frontier((0.0, 0.0), inflate_m=0.32))

    def test_vantage_point_is_the_reachable_cell_nearest_the_unreachable_frontier(self):
        m = self.gap_map()
        target = m.vantage_targets((0.0, 0.0), 0.32)
        self.assertIsNotNone(target)
        (cell, (x, y)), = target.items()
        self.assertEqual(tuple(int(v) for v in m.index(x, y)), cell)
        self.assertFalse(m.blocked(0.32)[cell])  # standable at normal clearance
        self.assertGreater(x, 0.8)  # as close to the gap as the clearance allows
        path = m.plan_to_frontier((0.0, 0.0), (), min_distance=0.3, inflate_m=0.32, targets_override=target)
        self.assertIsNotNone(path)
        self.assertAlmostEqual(path[-1][0], x, places=6)

    def test_a_half_metre_gap_opens_only_at_the_narrowest_tier(self):
        m = self.gap_map((-0.2, 0.3))
        self.assertIsNone(m.plan_to_frontier((0.0, 0.0), inflate_m=TIGHTEST_INFLATE_M))
        path = m.plan_to_frontier((0.0, 0.0), inflate_m=NARROW_INFLATE_M)
        self.assertIsNotNone(path)
        self.assertGreater(path[-1][0], 1.7)  # it went through the gap

    def test_narrowest_tier_is_last_and_can_be_switched_off(self):
        tiers = clearance_tiers([(1.0, 1.0)])
        self.assertEqual(tiers[0], (0.32, [(1.0, 1.0)]))  # normal clearance first, avoiding past failures
        self.assertEqual([t[0] for t in tiers[1:]], [0.32, 0.26, 0.22, 0.20])
        self.assertLess(navigator_pad(NARROW_INFLATE_M), navigator_pad(TIGHTEST_INFLATE_M))
        self.assertEqual(navigator_pad(0.22), 0.03)
        with mock.patch("microduck_qwen.explore.NARROW", False):
            self.assertEqual([t[0] for t in clearance_tiers(())], [0.32, 0.32, 0.26, 0.22])

    def test_no_vantage_when_the_frontier_is_already_reachable(self):
        m = OccupancyMap()
        m.update_from_points((0.0, 0.0), wall_points(), open_ends=[[3.5, y] for y in np.linspace(-2.5, 2.5, 30)])
        self.assertIsNotNone(m.plan_to_frontier((0.0, 0.0)))
        self.assertIsNone(m.vantage_targets((0.0, 0.0)))


class FakeBody:
    info = {"port": 0}

    def __init__(self):
        self.state = {"trunk": [0.0, 0.0, 0.0], "yaw": 0.0}
        self.turns = []

    def healthy(self):
        return self.state

    def _bounded(self, action, trace):
        self.turns.append(action)
        sign = 1 if action["direction"] == "left" else -1
        self.state = dict(self.state, yaw=self.state["yaw"] + sign * math.radians(action["degrees"]))
        return {"success": True}


class SettleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.body = FakeBody()
        self.explorer = Explorer(self.body, Path(self.tmp.name) / "out.json")
        self.explorer.frame_width = 640

    @staticmethod
    def ball(x, y, x0, x1):
        return {"color": "blue", "world_position": [x, y, 0.07], "bbox": [x0, 100, x1, 160], "visibility": "visible"}

    def script(self, *frames):
        frames = iter(frames)
        self.explorer.look = lambda color=None: next(frames)

    def test_a_ball_clear_of_the_edge_is_kept_without_turning(self):
        seen = [self.ball(2.0, 0.0, 280, 340)]
        self.script()
        self.assertEqual(self.explorer.settle("blue", seen), seen)
        self.assertEqual(self.body.turns, [])

    def test_a_ball_at_the_right_edge_is_faced_and_confirmed_in_a_fresh_frame(self):
        edge = [self.ball(2.0, -1.0, 600, 640)]
        centred = [self.ball(2.0, -1.0, 290, 350)]
        self.script(centred)
        self.assertEqual(self.explorer.settle("blue", edge), centred)
        (turn,) = self.body.turns
        self.assertEqual(turn["direction"], "right")
        self.assertAlmostEqual(turn["degrees"], 27.0)

    def test_a_ball_that_vanishes_after_the_turn_is_kept_as_a_suspect(self):
        self.script([])
        self.assertEqual(self.explorer.settle("blue", [self.ball(2.0, 1.0, 5, 60)]), [])
        self.assertEqual(self.explorer.suspects, [(2.0, 1.0)])
        self.assertEqual(self.body.turns[0]["direction"], "left")

    def test_still_at_the_edge_after_the_attempts_is_not_confirmed(self):
        edge = [self.ball(2.0, 0.5, 610, 640)]
        self.script(edge, edge)
        self.assertEqual(self.explorer.settle("blue", edge), [])
        self.assertEqual(len(self.explorer.suspects), 1)


class ReacquireTests(unittest.TestCase):
    def ball(self, x, y, color="blue", visibility="visible", kind="ball"):
        return {"color": color, "kind": kind, "visibility": visibility, "world_position": [x, y, 0.07]}

    def test_rebinding_picks_the_same_ball_by_position_not_another_ball_of_the_colour(self):
        from microduck_qwen.skills import rebind_near
        near, far = self.ball(2.1, 0.1), self.ball(2.0, 1.4)
        self.assertIs(rebind_near([far, near], "blue", "ball", (2.0, 0.0)), near)
        self.assertIsNone(rebind_near([far], "blue", "ball", (2.0, 0.0)))  # the only blue ball is elsewhere

    def test_rebinding_ignores_other_colours_unseen_balls_and_other_kinds(self):
        from microduck_qwen.skills import rebind_near
        candidates = [self.ball(2.0, 0.0, color="red"), self.ball(2.0, 0.1, visibility="hidden"),
                      self.ball(2.0, 0.05, kind="pillar")]
        self.assertIsNone(rebind_near(candidates, "blue", "ball", (2.0, 0.0)))
