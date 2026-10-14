# 从 MicroDuck 仿真采集数据并训练 Qwen-VL

本轮实现的是“真实 MuJoCo 图像与专家交互 → 仿真核验标签 → 图文 LoRA → 显式加载回控制器”。不是用文字模板冒充仿真数据，也不是重新训练官方走路策略。多轮模型探索、自改进和学习型 RM 尚未实现。

## 数据从哪里来

`vision_data.py` 启动独立的官方 MuJoCo body + robotd 实例，随机化球颜色、位置、障碍物及遮挡。默认采集端口 7842，不改交互实例的 7822 元数据。采集完成后关闭本轮拥有的进程，再进行模型训练。

机器人前向 RGB-D 与 MuJoCo segmentation 使用同一个渲染场景快照。RGB 颜色候选检测负责画框和随机编号；独立 segmentation 验证候选框对应哪个真实物体，避免直接拿检测器或模型输出当真值。部分可见比例由分割像素数与投影球面积估计，明显可见/部分可见采用 0.85/0.55 阈值，中间区间丢弃，非精确遮挡率真值。检测遗漏且真值可见时丢弃对应标签，不标成不存在。

模型输入只有**带编号的 RGB 图片与用户指令**。世界坐标、深度、物体 ID、碰撞、场景 seed、分割信息都保存在侧文件，仅用于监督、控制和评分。普通 `vision` 接口不返回 segmentation；采集专用端点需要独立实例显式启用 `--collect-gt`。

清晰训练场景中，oracle 专家调用现有 A*、仿真位姿和官方 ONNX 步态完成接近任务，保存运动前、中、后的真实图像与完整轨迹。轨迹成功还要求到达正确目标、距离不超过 0.325 米、未记录与场景物体的穿透接触。碰撞计数是物理步上接触数，不是独立碰撞事件数。地面正常接触不算场景障碍碰撞。

当前 SFT 学习的是每帧颜色/左右指代、不可见/部分可见判断、停止及不支持任务的 JSON，不是连续关节动作或完整轨迹模仿。即使专家运动失败，独立核验正确的图像标签仍可用于 grounding；失败运动不被标成成功示范。

## 第一批数据

本地目录：`data/sim-pilot-v1/`。

- 训练：seed 10000–10005，6 个场景、18 帧、124 条图文样本。
- 验证：seed 20000–20002，3 个场景、3 帧、21 条样本。
- 测试：seed 30000–30002，3 个场景、3 帧、21 条样本；尚未用于模型评测或调参。
- 12 个场景全部采集完成；3 次 oracle 专家导航经终点距离与接触记录核验成功。
- 每帧最多 7 类指令：红/蓝/绿球、画面左/右球、停止、不支持。只有 7 种固定表达，因此这里只检验小规模视觉布局泛化，不代表自然语言多样性泛化。

同一 scene seed 的图像与不同指令只属于一个 split，禁止相邻帧跨分割；训练前检查 JSONL 哈希、图片哈希、scene seed 和跨分割重复图像。图像/深度/轨迹保留在本机，不进入 Git；审计用清单和训练评测报告保存在 `experiments/sim-pilot-v1/`。实时仿真存在启动时序差异，相同 seed 不保证每个像素完全一致，不能仅凭重新采集就宣称逐位复现实验指标。

## 重新采集和训练

本机已有依赖；新机器先按 README 与 VISION.md 准备官方仓库、策略、MLX 环境及视觉模型。输出目录必须是新的，程序不会覆盖历史数据或权重。

```bash
cd /Users/mark/Code/microduck-qwen

# 自动采集；此阶段不加载 VLM，减少内存并发。
.venv-vlm/bin/python -m microduck_qwen.vision_data \
  --output data/sim-new-v1 --train-scenes 6 --valid-scenes 3 --test-scenes 3 --navigate

# 冻结基线，默认仅评测 valid。
.venv-vlm/bin/python -m microduck_qwen.vision_sft evaluate \
  --data data/sim-new-v1 --split valid --output runs/sim-new-base-valid.json

# 一遍 124 条样本仅适用于本轮实际 train=124 的数据；新数据先查看 manifest 行数。
.venv-vlm/bin/python -m microduck_qwen.vision_sft train \
  --data data/sim-new-v1 --steps 124 --accumulation 1 \
  --learning-rate 5e-5 --output adapters/vision-sim-new-v1

.venv-vlm/bin/python -m microduck_qwen.vision_sft evaluate \
  --data data/sim-new-v1 --split valid --adapter adapters/vision-sim-new-v1 \
  --output runs/sim-new-sft-valid.json
```

`steps` 是优化器更新数，`accumulation` 是每次更新的 microbatch 数，microbatch 固定为 1。训练只取 train；valid 前 3 条计算辅助 loss，完整决策正确性由独立 evaluate 判断。test 只做完整性校验，不参与训练、loss 或模型选择；只有显式 `--split test` 才评测它。

两个本轮适配器均从同一基座初始化，不互相覆盖，也不是恢复优化器状态：`adapters/vision-sim-pilot-v1/` 为 20 步短测，`adapters/vision-sim-pilot-epoch1/` 为覆盖全部 124 条样本的一遍训练。训练使用 rank 8、alpha 16、语言最后 8 层 q/v LoRA（458,752 参数）、冻结视觉塔、completion-only loss、梯度检查点、AdamW 和随机种子 42。

训练器记录输入 token 长度、训练样本 ID、参数是否变化、基座 revision、数据与权重哈希、MLX 峰值和系统 swap。图片/答案超长时拒绝静默截断；输出文件存在时拒绝覆盖。3 步内存探针不能作为正式任务适配器加载。

## 用新权重控制仿真

终端一：

```bash
cd /Users/mark/Code/microduck-qwen
.venv/bin/python -m microduck_qwen.sim --vision --viewer --seed 20000
```

终端二（本机训练权重）：

```bash
cd /Users/mark/Code/microduck-qwen
.venv-vlm/bin/python -m microduck_qwen.vision chat --adapter adapters/vision-sim-pilot-epoch1
```

先试“走到画面左边的球旁边”“走到红色球旁边”“停下来”。先用 `plan` 检查新表达，Ctrl+C 或独立 `vision stop` 停止。不传 `--adapter` 仍使用冻结基线。`--manifest` 可以选择独立实例，适配器必须有完整且匹配哈希的 `training_report.json`，预测日志记录实际适配器来源。

“走到绿色球的右边”这种相对位置任务尚未接入执行器，不属于本轮能力。官方步态保持冻结；机器人位姿、深度标定与安全地图仍为 oracle。部分遮挡/目标缺失时的规则保护不能算作模型判断正确。

## 结果如何解释

20 步短测：验证集完整 JSON 选择正确率为 13/21，冻结基线为 12/21。短测改善停止指令，却使部分颜色/左右选择退化，不能只报道总分上升。短测进入仿真后，seed 20000 正确接近左球；部分遮挡靠安全层拒绝动作，完全遮挡与缺失目标时错误输出 stop，安全不行动不代表语义判断正确。这些失败记录均保留。

覆盖全部 124 个不同训练样本的一遍 SFT：验证集为 **18/21（85.7%）**，基线为 **12/21（57.1%）**。同一批固定图像、指令、提示词和解码配置，统计的是安全层介入前的完整 JSON 决策，而非最终导航成功率：

| 验证标签 | 冻结基线 | 20 步短测 | 124 步一遍 SFT |
| --- | --- | --- | --- |
| 可见目标 | 12/12 | 9/12 | 12/12 |
| 停止指令 | 0/3 | 3/3 | 3/3 |
| 不支持任务 | 0/3 | 0/3 | 3/3 |
| 部分可见 | 0/2 | 0/2 | 0/2 |
| 不可见 | 0/1 | 1/1 | 0/1 |
| 合计 | 12/21 | 13/21 | 18/21 |

**总分增益全部来自停止与不支持指令；没有证据表明可见球 grounding 比基线更强。遮挡/不可见判断仍然失败，而且不可见项相对 20 步短测发生退化。** 不以总分掩盖分项结果。

最终训练耗时 170.12 秒；MLX 活跃分配峰值 5,255,458,459 bytes（5.255 GB，4.895 GiB），实际输入长度 331–336 tokens，不是 768 tokens 的最坏情况。32 个 LoRA 张量确实更新，重新加载后完成验证。valid 前 3 条辅助 loss 为 1.1770 → 0.0417，它只覆盖同一验证图像的三个颜色问题，不能代替完整评测。全机 swap 从约 7,969 MiB 增至 8,320 MiB，仍处于有交换的并发桌面环境，不是无压力训练证明。

最终适配器的独立仿真控制 smoke tests：

- seed 20000、20003、20004 均选对初始画面左球并完成接近；真实终点距离分别约 0.308、0.303、0.303 米。仅为 3 个未训练布局，不是可靠性统计，也没有本轮配对冻结模型导航对照。
- seed 20000 部分遮挡：模型仍判断可见，由 RGB-D 安全层停止。
- seed 20000 完全遮挡：模型输出 `{"status":"visible","color":null,"id":null}`，触发绑定/schema 错误，未执行、未成功；集成报告保留此失败，不能把整组写成全部通过。
- seed 20000 目标缺失：输出 unsupported，安全不行动，但不是正确的 not_visible 判断。

6 个集成案例中 5 个达到报告定义的导航/安全检查，1 个保留为模型输出错误；其中遮挡/缺失案例均没有执行移动，不等于模型学会了正确拒答。普通模型错误停止现会标为安全停止而非导航成功；这一语义修正发生在 20 步历史集成测试之后，历史原始记录不重写。最终回归测试 33 项通过，两个官方上游仓库保持未修改。

原始记录：`experiments/sim-pilot-v1/` 内的 collection manifest、两次 training report、三次 valid report、两次 integration report，以及最终完全遮挡的错误原文。训练集图片与适配器保留本机；本轮没有报告 test 集结果。

这还是极小规模 pilot：训练只有 6 个布局，验证只有 3 个布局，同一图像的多条指令不独立；没有多训练随机种子、独立最终测试或真机结果。后续先扩展场景和语言，再采集模型在**新的训练 seeds**上的探索，使用仿真验证筛选成功动作与纠错样本；验证/测试轨迹不得回灌训练。学习型 RM、历史多图、多步策略和在线 RL 必须单独实现与评测。
