from importlib.util import find_spec
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from microduck_qwen.sim_native import DEFAULT_RL


@unittest.skipUnless(find_spec("mujoco") and DEFAULT_RL.is_dir(), "optional MuJoCo/upstream camera dependencies")
class SegmentationTests(unittest.TestCase):
    def test_sphere_segmentation_boxes_do_not_include_other_spheres(self):
        import mujoco
        from microduck_qwen.vision_sensor import RGBDCamera

        sys.path.insert(0, str(DEFAULT_RL / "src"))
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody>
            <camera name="vision_camera" pos="0 0 0.2" xyaxes="0 -1 0 0 0 1" fovy="65"/>
            <geom name="goal_a" type="sphere" size="0.07" pos="1 0.36 0.07"/>
            <geom name="goal_b" type="sphere" size="0.07" pos="1 0 0.07"/>
            <geom name="goal_c" type="sphere" size="0.07" pos="1 -0.36 0.07"/>
          </worldbody></mujoco>''')
        model.vis.quality.offsamples = 4
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        import threading
        world = SimpleNamespace(model=model, data=data, lock=threading.Lock())
        with patch.dict("os.environ", {"MICRODUCK_COLLECT_GT": "1"}):
            camera = RGBDCamera(model, "vision_camera")
        try:
            camera.render(world)
            objects = camera.collection_packet()["ground_truth"]["objects"]
            self.assertEqual(len(objects), 3)
            for obj in objects.values():
                self.assertGreater(obj["visible_pixels"], 200)
                self.assertLess(obj["bbox"][2] - obj["bbox"][0], 100)
                self.assertLess(obj["bbox"][3] - obj["bbox"][1], 100)
        finally:
            camera.renderer.close()
