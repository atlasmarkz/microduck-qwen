# Third-party attribution

This is an independent research integration, not an official Pollen Robotics or Qwen repository.

- [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck): native robot daemon and protocol. Used as an external dependency; the upstream repository is not bundled.
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl): MuJoCo robot assets, official body server, pretrained locomotion interfaces, and BAM M6 actuator conventions. `microduck_qwen/sim_body.py` adapts the body server in-process; `sim_native.py` mirrors the 61D projected-gravity observation layout and action conventions from `scripts/infer_policy.py`. The RGB-D sensor is a project extension. Upstream copyright: Copyright 2026 Pollen Robotics. A copy of the upstream Apache-2.0 license is retained at `third_party/microduck_rl_LICENSE.txt`. Externally downloaded STL meshes remain under the upstream [CC BY-NC-SA 4.0 model license](https://github.com/pollen-robotics/microduck_rl/blob/develop/LICENSE-MODELS.md).
- [bam](https://github.com/Rhoban/bam): external motor/friction modeling dependency used by the simulator. Its implementation and weights are not bundled.
- [Qwen](https://huggingface.co/Qwen): text/vision model backbones. Model files are downloaded separately and retain their respective model licenses.
- [MLX-LM](https://github.com/ml-explore/mlx-lm) and [MLX-VLM](https://github.com/Blaizzy/mlx-vlm): external training/inference libraries; not bundled.

The upstream Apache license copy applies to the attributed upstream material. No new project-wide open-source license is granted by this notice; choosing a license for original research code remains the project owner's decision.
