# MicroDuck × Qwen-VL：中文视觉导航（技能 agent）

**VLM 看图 + 中文指令 → 每一步输出一个有界技能 JSON → 校验 → 执行 → 重新观察。** 模型只负责「看图 + 理解指令 + 选技能」，路径、避障与步态由几何规划与官方冻结策略完成。

![Qwen3-VL-4B 实录：在 2×2 房间世界里找到蓝球、走近并返回起点](assets/explore-rooms-4b-v1.gif)

**多房间探索实录（2026-11-04）。** 输入「去别的房间找蓝色的球，走到它旁边，然后回起点」，`lang-v2` 权重输出 `explore / blue → approach / blue / nearest → return_home → done`：用深度图增量建图，朝最近的可达前沿探索，看到蓝球后走近，再返回起点，最终距起点约 **0.079 m**。种子 98107，全程没有教师接管；GIF 为 **4 倍速**（画面已标注），原始运行约 125 秒。这是一次成功的单次运行，不是成功率；`done` 仍只是模型声明。为控制体积，GIF 重新编码为 780×420、隔帧保留，原始动作与哈希见 [记录](assets/explore-rooms-4b-v1.json)。

```powershell
.venv/Scripts/python.exe -X utf8 -m microduck_qwen.record_demo --adapter adapters/lang-v2 --world --world-seed 98107
```

## 技能与流程

- 技能：前进/后退、左右转向、接近球（`approach`）、搜索、探索（`explore`）、返回起点、停止、完成、澄清、拒绝、绕黄柱并返回（`bypass_return`）。每步重新观察，最多 5 步。
- 执行层：A* 导航 + 官方 ONNX 步态；`OccupancyMap` 由深度图增量建图，`explore` 朝最近的可达前沿走，直到看到目标颜色的球；导航护栏在净空过小时停车。
- 模型：`Qwen/Qwen3-VL-4B-Instruct` + LoRA（Windows PyTorch/CUDA，NF4 QLoRA，RTX 3060 8 GiB 可训练）。
- 数据：`agent_data` 生成专家纠错数据（actor 自己做，专家只在出错时接管，只有物理验证通过的轨迹进入训练）；`vision_torch train` 训练；`agent_eval --rollout` 做留出的独立物理评测并给 Wilson 95% 区间。
- 详细命令与格式见 [AGENT_TRAINING.md](AGENT_TRAINING.md)。

## 最新结果 · 2026-11

多房间世界（2×2 房间，门宽 1.1 m，起点房间前有三球一排）。同一批 80 个留出任务：种子 98000 / 98100，每组 8 个布局，每个布局含「靠近最左球再回起点」「往前移动 20 厘米」和每个首视野看不到的颜色各 3 条「去别处找某色球并走到旁边」。

| 版本 | 总成功 | 找球任务 | 说明 |
|---|---:|---:|---|
| lang-v1，改进前 | 52/80 (65%) | 20/48 | 基线 |
| + return_home 重试、目标记忆、安全停车后继续 | 61/80 | 29/48 | 执行层小修 |
| + 忠实建图 | 64/80 (80%) | 32/48 | 相机高于 0.4 m 墙，越墙射线曾被当作可通行，造出场外假前沿 |
| lang-v2 权重 | 63/80 | 32/48 | 动作类型错误 6 → 1，但探索预算耗尽 2 → 7 |
| + 导航余量下限与四档净空（**当前默认**） | **69/80 (86%)** | **41/48 (85%)** | 预算耗尽 7 → 0，平均探索 117 s |
| lang-v3（未采用） | 66/80 | 34/48 | "往前移动 20 厘米" 12/16 → 16/16，但找球退步，出现 JSON 格式错误 |

- 提升主要来自**执行层**（建图、净空档位、重试），不是重训。
- 同配置两次运行的波动约 ±4 / 40：69 对 63–64 超出噪声，但仍只是单一世界等级、两组种子、仿真环境。
- 已知弱点：`lang-v2` 下「往前移动 20 厘米」走完后有 4/16 输出 `unsupported`（`lang-v3` 修复它但让找球退步，已记录未采用）；最窄门口的布局（98101）仍因净空护栏停下；走近球时偶发护栏停车。
- 逐步记录与失败分析：[experiments/lang-v1/summary.md](experiments/lang-v1/summary.md)；诊断脚本 `diag_explore.py`、`render_map.py`、`diag_stop.py`。

## 快速开始（Windows）

需要 Python 3.12 与匹配驱动的 CUDA wheel（CUDA 12.8）：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -e '.[vision-torch,simulation]'
.venv/Scripts/python.exe -X utf8 -m microduck_qwen.vision download
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

重新评测当前默认权重（权重与数据不随仓库分发，需先按 [AGENT_TRAINING.md](AGENT_TRAINING.md) 生成）：

```powershell
.venv/Scripts/python.exe -X utf8 -m microduck_qwen.agent_eval --data data/lang-v2 --adapter adapters/lang-v2 --output runs/lang/eval-98000 --rollout --scene-mix world --world-level 2 --seed 98000 --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3
```

## 边界

- 自身位姿、相机外参与全局安全真值仍来自仿真；没有真机验证。
- 目标检测只覆盖本实验的红/蓝/绿 7 厘米球，不是通用物体检测器。
- 语言校验、可见性保护与停止规则是显式规则，只拒绝错误，不能算作模型能力。
- 训练与评测使用仿真数据与有限中文表达，不能替代真实用户表达评测。
- `done` 是模型声明，不等于客观任务成功。
- 当前验证在 Windows 本机完成；Mac（MLX）后端保留但未在本阶段验证。

## 官方仿真依赖

将上游依赖放在仓库同级目录：`../microduck_rl/.venv/`（MuJoCo、ONNX Runtime 与官方 body server）、`../duck-sim-state/policies/current/`（官方 ONNX 策略）。路径可用 `--rl / --policies` 指定。生成文件（`.venv/`、`.cache/`、`adapters/`、`data/`、`runs/`）不提交到 Git。

原始来源：[Qwen3-VL 模型](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)、[MicroDuck](https://github.com/pollen-robotics/microduck)、[官方仿真](https://github.com/pollen-robotics/microduck_rl)。

## 贡献者 / Contributors

- [Mark Zhang（@atlasmarkz）](https://github.com/atlasmarkz)：项目作者与维护者。
- Codex（OpenAI）：AI 协作贡献者，协助整理项目主题标签与贡献者文档。
