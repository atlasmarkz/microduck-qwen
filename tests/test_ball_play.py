import math
import unittest

from microduck_qwen import ball_play
from microduck_qwen.colab_ball import ball_color, summary


class ParseTests(unittest.TestCase):
    def test_verb_is_cut_out_and_the_rest_goes_to_the_model(self):
        self.assertEqual(ball_play.parse("去别的房间找蓝色的球，走到它旁边，把球带回起点"),
                         ("去别的房间找蓝色的球，走到它旁边", ball_play.PUSH))
        self.assertEqual(ball_play.parse("去别的房间找蓝色的球，走到它旁边，踢一脚，然后回起点"),
                         ("去别的房间找蓝色的球，走到它旁边", ball_play.KICK))

    def test_missing_approach_is_added(self):
        self.assertEqual(ball_play.parse("找到红球然后踢一脚"), ("找到红球，走到它旁边", ball_play.KICK))
        self.assertEqual(ball_play.parse("去旁边房间找绿球，带球回来"), ("去旁边房间找绿球，走到它旁边", ball_play.PUSH))

    def test_plain_instruction_has_no_verb(self):
        self.assertEqual(ball_play.parse("回到起点"), ("回到起点", ball_play.NONE))

    def test_color(self):
        self.assertEqual(ball_color("找蓝色的球"), "blue")
        self.assertIsNone(ball_color("找球"))


class GridTests(unittest.TestCase):
    SCENE = {"bounds": [-2.0, 4.0, -2.0, 2.0], "obstacles": [], "objects": {},
             "walls": [{"name": "w", "x1": 1.0, "y1": -2.0, "x2": 1.0, "y2": 0.6, "thickness": 0.1, "height": 0.4}]}

    def test_route_goes_through_the_gap(self):
        path = ball_play.Grid(self.SCENE).plan((0.0, -0.5), (2.2, -0.5))
        self.assertIsNotNone(path)
        crossing = [p for p in path if abs(p[0] - 1.0) < 0.15]
        self.assertTrue(crossing and min(p[1] for p in crossing) > 0.7)

    def test_closed_wall_has_no_route(self):
        wall = dict(self.SCENE["walls"][0], y2=2.0)
        self.assertIsNone(ball_play.Grid(dict(self.SCENE, walls=[wall])).plan((0.0, 0.0), (2.2, 0.0)))

    def test_ahead_interpolates_along_the_path(self):
        point = ball_play.ahead([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], (0.0, 0.0), 1.5)
        self.assertTrue(math.isclose(point[0], 1.0) and math.isclose(point[1], 0.5))


class SummaryTests(unittest.TestCase):
    def test_failures_are_reported_as_failures(self):
        self.assertIn("没有走到球旁", summary({"find": {"status": "step_limit_stopped"}}))
        report = {"find": {"status": "model_stop"}, "play": {"verb": "push", "distance_home_m": 0.1, "return": {"success": True},
                  "ball": {"success": False, "error": "超时", "ball_from_home_m": 2.1}}}
        self.assertIn("没带成", summary(report))


if __name__ == "__main__":
    unittest.main()
