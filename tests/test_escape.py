import math
import time
import unittest

from microduck_qwen.skills import SkillExecutor


class FakeBody:
    """Unicycle stand-in for the robot + observer: integrates move() commands at 12.5 Hz."""

    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw, self.cmd = x, y, yaw, (0.0, 0.0)

    def move(self, vx=0.0, vyaw=0.0):
        self.cmd = (vx, vyaw)

    def stop(self):
        self.cmd = (0.0, 0.0)

    def read(self):
        vx, wz = self.cmd
        self.yaw += wz * 0.08
        self.x += vx * 0.08 * math.cos(self.yaw)
        self.y += vx * 0.08 * math.sin(self.yaw)
        return {"trunk": [self.x, self.y, 0.1], "yaw": self.yaw, "trunk_z": 0.12, "imu": {"gravity": [0, 0, -1]}}


def executor(body, obstacle):
    e = SkillExecutor.__new__(SkillExecutor)
    e.robot = e.observer = body
    e.scene = {"bounds": [-3, 3, -3, 3], "obstacles": [obstacle], "objects": {}}
    e.cancel_file, e.started_ns = None, 0
    return e


class EscapeTests(unittest.TestCase):
    def test_backs_away_from_an_obstacle_it_is_pressed_against(self):
        # 0.30 m from a 0.1 m pillar: inflated clearance 0.30 - 0.1 - 0.14 = 0.06 -> push closer to 0.0
        body = FakeBody(0.27, 0.0, math.pi)  # facing the pillar at the origin
        e = executor(body, {"x": 0.0, "y": 0.0, "radius": 0.1})
        self.assertTrue(e.escape_cramped(trigger=0.07, target=0.12))
        self.assertGreaterEqual(math.hypot(body.x, body.y) - 0.1 - 0.14, 0.12 - 0.02)
        self.assertEqual(body.cmd, (0.0, 0.0))

    def test_does_nothing_when_there_is_room(self):
        body = FakeBody(1.0, 0.0, 0.0)
        e = executor(body, {"x": 0.0, "y": 0.0, "radius": 0.1})
        self.assertFalse(e.escape_cramped())
        self.assertEqual((body.x, body.y), (1.0, 0.0))


    def test_escapes_an_obstacle_that_exists_only_in_the_perceived_map(self):
        # True scene: the nearest real obstacle is far. Perceived map: a cell 0.18 m away (clearance ~0.015).
        body = FakeBody(0.0, 0.0, 0.0)
        e = executor(body, {"x": 2.0, "y": 0.0, "radius": 0.1})
        perceived = {"bounds": [-3, 3, -3, 3], "obstacles": [{"x": 0.18, "y": 0.0, "radius": 0.025}], "objects": {}}
        self.assertFalse(e.escape_cramped())  # the true scene alone sees no reason to move
        self.assertTrue(e.escape_cramped(scene=perceived))
        self.assertGreaterEqual(math.hypot(body.x - 0.18, body.y) - 0.025 - 0.14, 0.10 - 0.02)

    def test_a_perceived_escape_never_backs_into_a_real_obstacle(self):
        # Perceived cell just ahead, real pillar just behind: backing away from the cell must stop at the guard.
        body = FakeBody(0.0, 0.0, 0.0)
        e = executor(body, {"x": -0.30, "y": 0.0, "radius": 0.1})
        perceived = {"bounds": [-3, 3, -3, 3], "obstacles": [{"x": 0.18, "y": 0.0, "radius": 0.025}], "objects": {}}
        e.escape_cramped(scene=perceived, timeout_s=3.0)
        self.assertGreater(math.hypot(body.x + 0.30, body.y) - 0.1 - 0.14, 0.0)


if __name__ == "__main__":
    unittest.main()
