<p align="center">
  <img src="assets/cover.png" alt="MicroDuck × Qwen-VL" width="100%">
</p>

<p align="center">
  <b>一句中文指令，一张图，一个技能 JSON。</b><br>
  视觉语言模型负责「看图 + 理解指令 + 选技能」，路径、避障与步态交给几何规划和冻结的官方策略。
</p>

---

## 最终效果

<p align="center">
  <img src="assets/explore-rooms-4b-v1.gif" alt="在多房间世界里找到蓝球、走近并返回起点" width="100%">
</p>

指令「去别的房间找蓝色的球，走到它旁边，然后回起点」：模型依次输出 `explore → approach → return_home → done`，机器人用深度图增量建图、朝最近的可达前沿探索，看到蓝球后走近，再回到起点（最终距起点约 8 cm）。画面左上为俯视全景（橙点是机器人，蓝点是走过的路），右侧是模型实际看到的带编号画面和机器人第一视角。GIF 为 5 倍速、全程无人工接管；这是一次成功的单次运行，不是成功率。

在 2×2 房间世界里，对 80 个留出任务的独立物理评测：

| | 总成功率 | 去别处找球 |
|---|---:|---:|
| 改进前 | 52/80（65%） | 20/48 |
| 当前 | **69/80（86%）** | **41/48（85%）** |

提升主要来自建图与导航执行层，而不是重训。结果来自两组种子、单一世界等级的仿真评测，同配置两次运行约有 ±4/40 的波动；位姿与安全真值取自仿真，没有真机验证。

## 架构

<p align="center">
  <img src="docs/architecture.png" alt="系统架构" width="100%">
</p>

- 模型：`Qwen/Qwen3-VL-4B-Instruct` + LoRA，每步只输出一个技能：`move` · `turn` · `approach` · `search` · `explore` · `return_home` · `stop` · `done` · `clarify` · `unsupported` · `bypass_return`。
- 执行：深度图建图 + 前沿探索 + A* 路径，净空过小时护栏停车；官方 ONNX 步态跑在 MuJoCo 里。
- 训练：专家纠错数据聚合（模型自己做，专家只在出错时接管，只有物理验证通过的轨迹进入训练），命令与格式见 [AGENT_TRAINING.md](AGENT_TRAINING.md)；逐步实验记录见 [experiments/lang-v1/summary.md](experiments/lang-v1/summary.md)。

## 快速开始

需要 Windows、Python 3.12 与匹配驱动的 CUDA（RTX 3060 8 GB 即可训练和推理）。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -e '.[vision-torch,simulation]'
.venv/Scripts/python.exe -X utf8 -m microduck_qwen.vision download
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

仿真依赖放在仓库同级目录：`../microduck_rl/.venv/`（MuJoCo、ONNX Runtime、官方 body server）与 `../duck-sim-state/policies/current/`（官方 ONNX 策略）。

录制上面的演示（权重不随仓库分发，先按 [AGENT_TRAINING.md](AGENT_TRAINING.md) 训练 `adapters/lang-v2`）：

```powershell
.venv/Scripts/python.exe -X utf8 -m microduck_qwen.record_demo --adapter adapters/lang-v2 --world --world-seed 98107
```

复现留出评测：

```powershell
.venv/Scripts/python.exe -X utf8 -m microduck_qwen.agent_eval --data data/lang-v2 --adapter adapters/lang-v2 --output runs/lang/eval-98000 --rollout --scene-mix world --world-level 2 --seed 98000 --rollout-seeds 8 --task-ids 3 4 --explore-tasks 3
```

---

<sub>作者与维护者：[Mark Zhang（@atlasmarkz）](https://github.com/atlasmarkz) · AI 协作：Codex（OpenAI）· 模型：[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) · 机器人：[MicroDuck](https://github.com/pollen-robotics/microduck) · 仿真：[microduck_rl](https://github.com/pollen-robotics/microduck_rl)</sub>
