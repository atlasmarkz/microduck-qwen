import json
import math
import time
import unittest

from microduck_qwen.navigation import clearance, pathfind, segment_clear
from microduck_qwen.vision_world import WORLD_LEVELS, make_world


def reachable(scene, goal, stop=0.30):
    try:
        pathfind((0.0, 0.0), goal, stop, scene)
        return True
    except ValueError:
        return False


class WorldTests(unittest.TestCase):
    def test_same_seed_is_reproducible_and_seeds_differ(self):
        self.assertEqual(make_world(7, (2, 2), 1.0), make_world(7, (2, 2), 1.0))
        self.assertNotEqual(json.dumps(make_world(7)), json.dumps(make_world(8)))
        json.dumps(make_world(7))  # must be serialisable for the simulator

    def test_start_room_is_at_the_origin_with_the_usual_ball_row(self):
        for seed in range(20):
            world = make_world(seed, (3, 3), 1.5)
            xmin, xmax, ymin, ymax = world["bounds"]
            self.assertTrue(xmin < 0 < xmax and ymin < 0 < ymax)
            balls = [world["objects"][f"ball_{i}"] for i in range(3)]
            self.assertTrue(all(0.8 < b["x"] < 1.3 for b in balls))
            self.assertEqual([round(b["y"]) for b in balls], [0, 0, 0])
            self.assertGreater(balls[0]["y"], balls[1]["y"])
            self.assertGreater(balls[1]["y"], balls[2]["y"])
            self.assertGreater(clearance((0.0, 0.0), world), 0.1)

    def test_every_placed_object_stays_in_bounds_and_off_walls(self):
        for seed in range(20):
            world = make_world(seed, (3, 3), 2.0)
            xmin, xmax, ymin, ymax = world["bounds"]
            for obj in list(world["objects"].values()) + world["obstacles"]:
                self.assertTrue(xmin < obj["x"] < xmax and ymin < obj["y"] < ymax)
                self.assertTrue(segment_clear((obj["x"], obj["y"]), (obj["x"], obj["y"]), {**world, "objects": {}, "obstacles": []}, 0.0))

    def test_every_room_is_connected_so_every_ball_is_reachable(self):
        for seed in range(12):
            world = make_world(seed, (3, 3), 1.0)
            for name, ball in world["objects"].items():
                self.assertTrue(reachable(world, (ball["x"], ball["y"])), (seed, name))

    def test_walls_block_straight_lines_but_doorways_do_not(self):
        world = {**make_world(3, (1, 2), 0.0), "objects": {}, "obstacles": []}  # walls only
        gate = next(d for d in world["doors"] if d["forward"])
        wall_x = gate["x"]
        # A straight line through the wall away from the doorway is blocked ...
        blocked_y = gate["y"] + gate["width"] / 2 + 0.6
        self.assertFalse(segment_clear((wall_x - 0.8, blocked_y), (wall_x + 0.8, blocked_y), world))
        # ... and through the doorway centre it is clear.
        self.assertTrue(segment_clear((wall_x - 0.8, gate["y"]), (wall_x + 0.8, gate["y"]), world))

    def test_routes_go_through_doorways_and_planning_is_fast_on_a_large_map(self):
        world = make_world(5, (3, 3), 1.5)
        far = max(world["objects"].values(), key=lambda b: math.hypot(b["x"], b["y"]))
        began = time.perf_counter()
        route = pathfind((0.0, 0.0), (far["x"], far["y"]), 0.30, world)
        elapsed = time.perf_counter() - began
        self.assertLess(elapsed, 1.0, f"planning took {elapsed:.2f}s")
        self.assertGreater(len(route), 2)  # not a straight line: walls force a detour
        for a, b in zip(route, route[1:]):
            self.assertTrue(segment_clear(a, b, world))

    def test_replanning_works_from_inside_a_balls_path_padding(self):
        world = {**make_world(1, (2, 2), 1.0)}
        ball = world["objects"]["ball_0"]
        # 0.03 m past the physical envelope: inside the 0.035 m path padding, outside contact.
        start = (ball["x"] + 0.14 + ball["radius"] + 0.03, ball["y"])
        self.assertGreater(clearance(start, world), 0.015)
        self.assertGreater(len(pathfind(start, (start[0] + 1.0, start[1]), 0.3, world)), 1)

    def test_levels_grow_in_size_and_clutter(self):
        sizes = [make_world(1, **level) for level in WORLD_LEVELS]
        counts = [len(w["objects"]) + len(w["obstacles"]) for w in sizes]
        self.assertEqual(counts, sorted(counts))
        self.assertLess(counts[0], counts[-1])
        with self.assertRaises(ValueError):
            make_world(1, (0, 2))


if __name__ == "__main__":
    unittest.main()
