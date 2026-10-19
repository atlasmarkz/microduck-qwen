import math
import unittest

from microduck_qwen.contract import arena, validate
from microduck_qwen.navigation import pathfind, segment_clear


class NavigationTests(unittest.TestCase):
    def test_direct_line_is_blocked_but_route_avoids_obstacle(self):
        scene = arena()
        target = (1.2, 0.25)
        self.assertFalse(segment_clear((0,0), target, scene))
        route = pathfind((0,0), target, 0.3, scene)
        self.assertGreater(len(route), 2)
        self.assertLessEqual(math.dist(route[-1], target), 0.3)
        self.assertTrue(all(segment_clear(a,b,scene) for a,b in zip(route,route[1:])))

    def test_point_inside_physical_obstacle_cannot_be_reached(self):
        with self.assertRaisesRegex(ValueError,"不可达"):
            pathfind((0,0),(0.6,0.12),0.1,arena())

    def test_safe_continuous_start_can_bridge_a_blocked_rounded_cell(self):
        scene = arena()
        route = pathfind((0.44,-0.25), (1.2,0.25), 0.3, scene)
        self.assertEqual(route[0], (0.44,-0.25))
        self.assertLessEqual(math.dist(route[-1], (1.2,0.25)), 0.3)
        self.assertTrue(all(segment_clear(a,b,scene) for a,b in zip(route,route[1:])))

    def test_short_goal_does_not_drop_rounded_terminal_cell(self):
        scene = {"bounds":[-1,1,-1,1],"objects":{},"obstacles":[]}
        route = pathfind((0.013,0.0),(0.12,0),0.1,scene)
        self.assertLessEqual(math.dist(route[-1],(0.12,0)),0.1)

    def test_rejects_duplicate_fields_and_nonfinite_numbers(self):
        with self.assertRaises(ValueError):
            validate('{"status":"stop","status":"ok","goals":[]}')
        with self.assertRaises(ValueError):
            validate('{"status":"ok","goals":[{"target":"point","speed":"normal","stop_distance_m":0.15,"x":NaN,"y":0}]}')

    def test_unknown_target_and_extra_control_fields_fail_closed(self):
        with self.assertRaises(ValueError):
            validate({"status":"ok","goals":[{"target":"green_ball","speed":"normal","stop_distance_m":0.3}]})
        with self.assertRaises(ValueError):
            validate({"status":"stop","goals":[],"shell":"anything"})


if __name__ == "__main__":
    unittest.main()
