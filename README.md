# MicroDuck × Qwen：中文目标导航

## 项目目标与当前状态

研究目标是后训练 Qwen-VL，使其根据视觉、自然语言和历史反馈产生高层决策，结合 MicroDuck 已有 RL 步态完成仿真控制。已实现 **Qwen3-0.6B 文本 SFT**、**冻结 Qwen3-VL-2B 视觉基线**、内存探针，以及 **直接从 MuJoCo 采集图文数据、LoRA 训练和显式适配器控制的初轮 pilot**。步骤与结果见 [SIM_TRAINING.md](SIM_TRAINING.md)。多轮自改进、学习型 RM、多步策略和高层 RL 尚未实现；不把安全规则或文本指标当成视觉训练成果。硬件与后续计划见 [TRAINING_PLAN.md](TRAINING_PLAN.md)。

获取源码：

```bash
git clone git@github.com:AMark-CS/microduck-qwen.git
cd microduck-qwen
```

GitHub 只同步源码、依赖配置、测试及说明，不包含本机虚拟环境、模型缓存、LoRA 权重、生成数据或原始运行日志。新机器需要按下面流程创建环境、下载模型并训练；本机“已训练完成”的描述仅指现有本地安装。`data-v1/` 和历史权重未发布，因此历史两阶段训练不能仅凭本仓库完整复现；当前数据生成器可重新生成当前版本的数据。

文本分支新训练示例（模型已下载、数据已生成后）：

```bash
.venv/bin/python -m microduck_qwen.train --iters 300 --adapter adapters/nav-v2
```

这会新建一组适配器，不等同于历史 900 步权重；已有同名权重时训练器会拒绝覆盖。MicroDuck 仿真仓库与策略需另行准备，路径可通过 `sim --microduck PATH --rl PATH --policies PATH` 指定。本机默认路径不是跨机器安装保证。

视觉分支已独立实现：看图选择左右球、随机布局、RGB-D 定位和遮挡保护。默认仍为冻结 Qwen3-VL-2B，可用 `vision chat --adapter PATH` 显式加载本机图文 SFT 权重，见 [VISION.md](VISION.md) 与 [SIM_TRAINING.md](SIM_TRAINING.md)。本页下面的训练与 61.25% 指标仍专指 Qwen3-0.6B 文本 SFT，不是视觉训练结果。

在 Apple Silicon Mac 上对 Qwen3-0.6B 做 LoRA 监督微调，输入中文导航指令，输出目标及速度、停靠距离；导航器使用 MuJoCo 位姿与场景地图规划避障路径，向官方 `robotd` 发送速度命令，既有 ONNX 步态策略在 50 Hz 控制机器人。

这是实际更新 Qwen 适配器参数的 SFT 后训练。原始 Qwen 权重冻结；步态策略保持为官方预训练策略。

## 第一版任务与边界

- “绕开障碍物走到红球旁边，离球30厘米停下。”
- “慢慢走到蓝色标记附近。”
- “移动到世界坐标（0.2，-0.4）。”
- “停下来。”

多目标序列有输出接口，但模型仍会漏掉后续目标，暂不作为可靠功能。导航前会打印模型生成的目标，建议先用 `plan` 检查陌生表达。停止控制优先使用 Ctrl+C 或下述 `stop` 命令，不依赖模型识别停止措辞。

场景在 `microduck_qwen/arena.json`：红球 (1.2, 0.25)，蓝色标记 (1.1, -0.65)，障碍柱 (0.6, 0.12)。单位为米，x 向前、y 向左，停靠距离是机器人躯干中心与目标中心的平面距离，执行容差 2.5 厘米。红球为固定球形几何体；本实验做接近导航，不做踢球或抓取。

当前观察使用**仿真真实位姿与已知静态地图**，没有通过摄像头识别目标；避障来自几何路径规划与闭环速度控制。Qwen 学的是语言到目标/约束的映射，不是在本次微调中学会步态或避障动力学。

## 直接体验（本机权重已训练完成）

在项目目录开两个终端。两个终端都先进入：

```bash
cd /Users/mark/Code/microduck-qwen
```

终端一启动有窗口的仿真：

```bash
.venv/bin/python -m microduck_qwen.sim --viewer
```

终端二输入自然语言：

```bash
.venv/bin/python -m microduck_qwen.cli chat
```

也可以只执行一条指令：

```bash
.venv/bin/python -m microduck_qwen.cli run '绕开障碍物走到红球旁边，离球30厘米停下'
```

只查看模型输出而不移动机器人：

```bash
.venv/bin/python -m microduck_qwen.cli plan '移动到世界坐标（0.2，-0.4）'
```

在正在执行指令的终端按 Ctrl+C 会停止本次导航。也可在另一个终端运行以下命令取消正在运行的导航并发送停止命令：

```bash
.venv/bin/python -m microduck_qwen.cli stop
```

仿真终端的 Ctrl+C 关闭本项目创建的 body 与 robotd。原有 MicroDuck 仿真不受影响；本项目默认使用 7821 端口、独立临时 socket 和日志目录，不修改上游仓库。

## 环境与重现训练

训练环境：Python 3.12、MLX 0.32.3、MLX-LM 0.32.0、Transformers 5.19.0。Mac 16GB 内存上完成训练。默认加载 `adapters/nav-v2/`；该目录的 `provenance.json` 记录模型 revision、数据哈希和训练来源。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m microduck_qwen.download
.venv/bin/python -m microduck_qwen.dataset
.venv/bin/python -m microduck_qwen.train --iters 5 --adapter adapters/smoke-new
.venv/bin/python -m microduck_qwen.train --iters 300 --adapter adapters/retrain
```

训练更新后 16 层的 rank-16 LoRA，共约 5.77M 参数（模型约 0.97%），学习率 1e-4，batch size 2。训练和推理都使用 `enable_thinking=False`；损失只计算 JSON 答案及 EOS，训练样本超长时直接报错，避免静默截断目标。

现有权重经过两阶段训练：v1 在原始 767 行数据上训练 300 步；v2 在增强后的 1,542 行数据上继续更新 LoRA 600 步（重新创建优化器，不是恢复优化器状态），累计 900 步。另有 5 步 smoke 测试，其权重没有接入最终模型。保留的历史数据 `data-v1/` 可用于复现第一阶段：

```bash
.venv/bin/python -m microduck_qwen.train --data data-v1 --iters 300 --adapter adapters/repro-v1
.venv/bin/python -m microduck_qwen.train --data data --iters 600 --adapter adapters/repro-v2 --resume-adapter-file adapters/repro-v1/adapters.safetensors
```

示例数据由可复现的模板教师生成。当前训练 1,542 行（1,374 条独立指令，停止/不支持类有重采样），验证 148 条，测试 226 条。措辞模板跨分割隔离；部分验证/测试坐标值不出现在训练中。增强依据验证集错误设计，验证和测试文件保持原哈希。它们是开发用合成数据，不能替代真实用户指令评测，也不能支持论文级泛化结论。

## 评测

```bash
.venv/bin/python -m microduck_qwen.evaluate --base --limit 80
.venv/bin/python -m microduck_qwen.evaluate --adapter adapters/nav-v2 --limit 80 --output runs/test-adapter-v2.json
python3 -m unittest discover -s tests -v
```

读取 `runs/test-base.json` 与 `runs/test-adapter-v2.json`。同样的 80 条分层抽样指令，原始模型完整计划精确匹配 20.0%，v2 为 61.25%；合法输出率从 80% 提升到 95%。v2 的单物体目标正确率 76.92%、坐标 61.54%、序列 35%；中位推理延迟约 0.73 秒。第二轮没有改善总精确匹配率，但提高了合法率、坐标和序列解析表现。`runs/test-adapter.json` 是第一轮权重的结果，不是默认模型。

指标是严格的 JSON 合法率、完整目标序列精确匹配率、各任务类型正确率和模型推理延迟；不把语言解析正确率当成物理导航成功率。错误可能是合法但语义不对的 JSON，因此 schema 校验无法代替语义正确性检查。

运行实际导航后，`runs/command-*.json` 保存模型原始输出、解析结果与执行结果，`runs/trajectory-*.jsonl` 保存真实轨迹、目标距离和障碍物净空。导航使用 A*、障碍膨胀及闭环航向控制，10 Hz 刷新速度意图；跌倒、位姿停更、目标不可达或超时会停止。障碍净空是平面圆形包络估算，不是全身碰撞证明。

步态输入经过本机校准：slow/normal/fast 对应 0.30/0.35/0.40 的前进条件输入，转向条件输入最高 1.5；这些是送给步态网络的命令，不保证等于真实速度。当前 actor 在过小命令下会保持站立。本项目把仿真电机对齐为官方 `infer_policy.py` 的 BAM M6 XL330（7.4V、kp=200），并把 robotd 动作幅度设为 1.0、附加关节低通设为 1.0，避免硬件默认配置与该仿真配置混用。这些设置仅用于仿真，不能直接当作真机部署参数。

## 官方仿真依赖

本机已安装的依赖默认读取：

- `/Users/mark/Code/microduck/target/debug/robotd`
- `/Users/mark/Code/microduck_rl/.venv/`（MuJoCo、ONNX Runtime 与官方 body server）
- `/Users/mark/Code/duck-sim-state/policies/current/`（官方 ONNX 策略）

换机器时用 `sim --microduck PATH --rl PATH --policies PATH` 指定。Python 模型环境与官方仿真环境分开，避免依赖冲突。非 Mac 环境需要替换 MLX 训练/推理后端；当前启动器按 macOS ONNX 动态库和 mjpython 写成。

## 后续研究

模型选型与边界见 [ANALYSIS.md](ANALYSIS.md)。初轮仿真采集与图文 LoRA pilot 已实现，详见 [SIM_TRAINING.md](SIM_TRAINING.md)；仍需扩大数据、独立评测与多轮探索，不把初轮结果视为可靠通用策略。

1. 收集真实中文表达与失败案例，扩大独立测试集；比较原始模型、SFT 模型及规则/更大模型基线。
2. 将仿真真实状态替换为视觉检测与状态估计，评测感知误差下的任务成功率。
3. 引入动态障碍、多目标组合、反馈与失败恢复，再考虑用仿真成功信号对高层策略进行 RL 后训练。

原始来源：[Qwen 模型](https://huggingface.co/Qwen/Qwen3-0.6B)、[MLX LoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)、[MicroDuck](https://github.com/pollen-robotics/microduck)、[官方仿真](https://github.com/pollen-robotics/microduck_rl)。

## 本机目录迁移

源代码、虚拟环境、模型缓存、LoRA 权重、数据和实验记录统一存放在 `/Users/mark/Code/microduck-qwen`。原来的 Codex 工作区项目路径保留为指向该目录的兼容软链接，不是另一份代码；现有虚拟环境脚本和历史记录中的旧绝对路径仍可访问。不要单独删除该兼容链接，除非先在 Code 目录重建虚拟环境并检查旧路径引用。

模型加载优先解析项目内缓存，使迁移不依赖修改历史 provenance。若要把项目复制到另一台机器，按上面的环境安装流程重建 `.venv`，不要直接复用本机虚拟环境。
