import math
import socketserver
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from microduck_qwen.navigation import Robot
from microduck_qwen.sim_native import Commands, ControlHandler
from microduck_qwen.sim_body import reset_bam


class NativeSimulationTests(unittest.TestCase):
    def test_bam_reset_preserves_current_pose_with_both_apis(self):
        import numpy as np

        data = SimpleNamespace(qpos=np.array([4.0, 5.0, 6.0]), time=10.0)
        calls = []
        reset_bam(SimpleNamespace(reset=lambda q: calls.append(q.copy())), data)
        self.assertEqual(calls[0].tolist(), [4.0, 5.0, 6.0])
        controller = SimpleNamespace(model=SimpleNamespace(reset=lambda: calls.append("model")),
            qpos_indexes=[1, 2], q_target=np.zeros(2), last_ts=0)
        reset_bam(controller, data)
        self.assertEqual(controller.q_target.tolist(), [5.0, 6.0])
        self.assertEqual(controller.last_ts, 10.0)
        self.assertEqual(calls[-1], "model")

    def test_invalid_commands_are_rejected_and_stale_commands_expire(self):
        commands = Commands()
        with patch("microduck_qwen.sim_native.time.monotonic", return_value=10):
            commands.set({"vx": 0.35, "vyaw": 1.5})
            self.assertEqual(commands.get(), (0.35, 0.0, 1.5))
        with patch("microduck_qwen.sim_native.time.monotonic", return_value=10.51):
            self.assertEqual(commands.get(), (0.0, 0.0, 0.0))
        for params in ({"vx": True}, {"vx": math.nan}, {"vx": 0.41}, {"vy": 0.1},
                       {"vyaw": 1.51}, {"code": "x"}, []):
            with self.assertRaises(ValueError):
                commands.set(params)

    def test_existing_robot_client_can_move_and_stop_over_tcp(self):
        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True

        with Server(("127.0.0.1", 0), ControlHandler) as server:
            server.commands = Commands()
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            robot = Robot(list(server.server_address))
            try:
                robot.move(0.35, 1.2)
                self.assertEqual(robot.call("robot.move", {"vx": 0.3}), {"ok": True})
                self.assertEqual(server.commands.get(), (0.3, 0.0, 0.0))
                self.assertEqual(robot.stop(), {"ok": True})
                self.assertEqual(server.commands.get(), (0.0, 0.0, 0.0))
                with self.assertRaises(RuntimeError):
                    robot.call("robot.move", {"vx": 2})
                self.assertEqual(server.commands.get(), (0.0, 0.0, 0.0))
            finally:
                robot.close()
                server.shutdown()
                thread.join(timeout=2)
