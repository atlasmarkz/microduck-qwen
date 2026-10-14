"""Deployment diagnostic using the official BAM inference implementation."""

import importlib.util
import argparse
import json
import math
from pathlib import Path


def main():
    import mujoco

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='velstand.onnx')
    args = parser.parse_args()

    rl = Path('/Users/mark/Code/microduck_rl')
    spec = importlib.util.spec_from_file_location('official_infer', rl/'scripts/infer_policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bam = module.load_bam_model(200, 7.4, None)
    model, data, controller, names = module.load_mujoco_with_bam(
        str(rl/'src/mjlab_microduck/robot/microduck/scene.xml'), bam, 0.005, 0.0, 6.0)
    policy = module.PolicyInference(model, data, bam_ctrl=controller,
        walking_onnx_path=str(Path('/Users/mark/Code/duck-sim-state/policies/current')/args.policy),
        use_projected_gravity=True, new_cmd_obs=True)
    data.qpos[:7] = [0,0,0.125,1,0,0,0]
    for index, address in enumerate(policy.joint_qpos_indices):
        data.qpos[address] = policy.default_pose[index]
    controller.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model,data)
    for vx, yaw in [(0,0),(0.12,0),(0.3,0),(0.4,0),(0,1.0),(0.2,1.0)]:
        policy.set_vel_cmd(vx,0,yaw)
        before = data.qpos[:7].copy()
        for _ in range(150):
            policy.apply_action(policy.infer())
            for _ in range(4):
                controller.update()
                mujoco.mj_step(model,data)
        after = data.qpos[:7].copy()
        print(json.dumps({'command':[vx,yaw],'before':before.tolist(),'after':after.tolist()}),flush=True)


if __name__ == '__main__':
    main()
