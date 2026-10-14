# MicroDuck 第一视角视觉导航

已完成的 6 场景 smoke tests 与保留失败记录见 [VISION_RESULTS.md](VISION_RESULTS.md)。其中 3 个随机布局完成了正确左球接近，3 个遮挡/缺失案例由安全层拒绝动作；这不等于 VLM 自身正确拒答。

这是可运行的视觉分支，不替换已有 Qwen3-0.6B 文本 SFT。当前导航仍使用本地 Qwen3-VL-2B-Instruct 的 4-bit MLX 冻结基线。另已完成独立的 3 步图文 LoRA 内存探针，不接入导航，也不代表任务级后训练完成；实测与计划见 [TRAINING_PLAN.md](TRAINING_PLAN.md)。

## 已实现的链路

机器人机身前向 RGB-D 虚拟相机 → RGB 彩色球候选检测与随机编号 → Qwen-VL 看图选择目标 → 颜色/编号与简单左右关系校验 → 深度反投影成目标位置 → A* 与官方 ONNX 步态 → 持续可见性保护。

“左边”指发出指令时画面中可见球的左边。编号随机打乱，编号大小不代表左右。模型只接收标注后的 RGB 图片和用户指令，不接收世界坐标、场景文件、真实物体 ID 或测试标签。标注候选来自 RGB 颜色连通域，不是 MuJoCo segmentation ID。

执行使用标定的深度、相机外参和已知球半径定位。相机安装在机器人机身前方，随身体移动，不是第三人称截图；它是本项目添加的虚拟 RGB-D 相机，尚未接真实硬件相机，也不是官方侧置头部相机的原始画面。

## 直接运行

两个终端均先进入项目：

```bash
cd /Users/mark/Code/microduck-qwen
```

终端一，启动随机布局和可视化：

```bash
.venv/bin/python -m microduck_qwen.sim --vision --viewer --seed 42
```

终端二，输入视觉指令：

```bash
.venv-vlm/bin/python -m microduck_qwen.vision chat
```

示例：“走到画面左边的球旁边”“走到最右边的球旁边”“走到蓝色球旁边”。当前固定以 slow 步态接近，停靠距离 0.30 米，加 0.025 米执行容差；还没有把任意速度/距离语言参数接入视觉接口。

只检查图像与模型计划，不移动机器人：

```bash
.venv-vlm/bin/python -m microduck_qwen.vision observe
.venv-vlm/bin/python -m microduck_qwen.vision plan '走到画面左边的球旁边'
```

`observe` 打印保存的 PNG 路径。每次 plan/run 保存原始 RGB、随机编号图、深度、标定信息、模型原文、目标选择和执行轨迹到 `runs/vision/`。模型格式错误或颜色/编号绑定错误也会保留原文并拒绝动作。

Ctrl+C 中断执行会停车；也可以在另一个终端直接停止，不经过模型：

```bash
.venv-vlm/bin/python -m microduck_qwen.vision stop
```

视觉实例默认端口 7822，已有文本实例默认 7821，元数据各自独立。端口已被占用时不会擅自关闭其他进程；先在原仿真终端 Ctrl+C，再重启需要的场景。更换 seed/遮挡场景后也请重新启动 chat。

## 遮挡场景

关闭前一个视觉仿真后，用下面任一命令重新启动：

```bash
.venv/bin/python -m microduck_qwen.sim --vision --viewer --seed 42 --occlusion partial
.venv/bin/python -m microduck_qwen.sim --vision --viewer --seed 42 --occlusion full
.venv/bin/python -m microduck_qwen.sim --vision --viewer --seed 42 --occlusion absent
```

它们是三种替代场景，不要同时在默认端口启动。seed 42 的初始左球是蓝色；可用“走到蓝色球旁边”测试。换 seed 后颜色顺序也可能改变，不要假定左球永远是蓝色。

部分可见时，依据 RGB 可见面积、深度和已知半径估计可见比例，低于 0.65 就停止。执行过程中若此前选中的球消失，系统把它投影到新画面，与前景深度比较：有更近遮挡物则报告 occluded；没有对应球则报告 missing；离开视野则报告 out_of_view，均拒绝继续走。

**完全被挡住且此前从未见过的目标，无法仅凭当前图像区分“藏在后面”与“根本不存在”。** 这两种情况都应不行动。本版不会自动绕柱寻找目标，不提供未知目标位置猜测。当前 VLM 本身仍有误选/不拒答问题，遮挡与颜色约束的安全层不能当成模型已经学会遮挡推理的证据。

## 重现环境与评测

本机视觉环境已经安装；换机器重建：

```bash
uv venv --python 3.12 .venv-vlm
uv pip install --python .venv-vlm/bin/python -r requirements-vision.txt
.venv-vlm/bin/python -m microduck_qwen.vision download
.venv-vlm/bin/python -m unittest discover -s tests -v
```

模型仓库：`mlx-community/Qwen3-VL-2B-Instruct-4bit`，固定 revision `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`。加载不执行远程代码；模型缓存在项目 `.cache/vlm-hub/`，不调用付费 API。依赖版本固定在 `requirements-vision.txt`。

批量重置三个布局，并测试部分遮挡、完全遮挡和目标缺失：

```bash
.venv-vlm/bin/python -m microduck_qwen.vision_eval --navigate --seeds 42 43 44
```

评测拥有独立端口 7832、子进程和每场元数据，不覆盖交互视觉实例。结果在 `runs/vision-eval/`，包括启动失败/重试、模型目标选择正确性、真值下的最终距离、定位误差和安全拒绝。真值仅用于物理环境、安全地图与事后评分，不进入 VLM 提示词。少量 smoke tests 不代表统计意义上的泛化成功率。

## 明确限制

- 自身位姿、相机外参和全局安全地图仍来自仿真；本版是视觉 grounding + oracle 定位/地图的混合系统，不是无真值视觉导航或 SLAM。
- 只对本实验的红/蓝/绿 7 厘米球做颜色候选检测；不是任意物体检测器。左右/颜色校验是显式规则，只拒绝错误，不替模型选另一个球。
- 目标在一次指令的初始画面中确定并锁定；跟踪不重新选择别的同色球。当前静态布局，不保证动态球、复杂指代、强遮挡或未知光照。
- VLM 推理时机器人先停下；执行阶段使用轻量 RGB-D 跟踪，不让慢速模型堵塞步态。渲染在主线程、50 Hz 物理循环在独立线程，避免 RGB-D 渲染暂停物理时间。
- 视觉实例从 HOME 开始，固定通过原生控制器的初始化渐变，平衡策略接管后释放；只是一种仿真初始化方式，不是额外训练出的起身技能。
- 当前控制分支没有加载后训练 VLM；只有独立的图文 LoRA 内存探针。已有图像/指令/错误记录可用于下一步构建按场景隔离的数据集，但不能直接拿模型自己的输出当正确标签。

源码入口：`vision.py`（感知/模型/执行）、`vision_sensor.py`（RGB-D/实时循环）、`vision_scene.py`（随机场景）、`vision_eval.py`（集成测试）。已有文本导航说明仍在 `README.md`。
