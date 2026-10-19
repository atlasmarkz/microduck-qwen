# MicroDuck：仿真采集与迭代式 Qwen-VL 后训练

## 这版具体做什么

Qwen3-VL-2B 根据编号 RGB 图、中文指令、自己的位姿和本条指令的已完成动作，**逐步选择一个有界技能**。每步执行后重新观察，最多执行 5 步。这是本项目**唯一**的 VLM 数据链路：原先只做选球的 `vision` 指代分支已移除，选球能力并入 `approach` 技能。Windows（PyTorch + CUDA）与 Mac（MLX）两套后端都支持训练与推理。

技能：前进/字面后退、左右转向、接近当前图像里的球、返回仿真启动位置、搜索、停止、完成、澄清、拒绝未实现功能。一次平移 0.05–0.35 米，转向 5–120 度。本 agent 未接入踢球、抓取、跳舞；这些不能被“走到球旁”冒充完成。

低层仍使用官方预训练 RL 步态；这次更新的是高层 Qwen-VL 的 LoRA，**没有重新训练步态，也没有高层在线 RL**。

## 本地体验

两个终端均进入 `microduck-qwen`。需要上游 MicroDuck 仓库和缓存的固定模型；换机器需先参考 `README.md` / `WINDOWS_VISION.md` 安装，GitHub 不包含这些大文件。下面命令以 Mac（`.venv-vlm`，MLX）为例；Windows 把解释器换成 `.venv/Scripts/python.exe -X utf8`、把 `microduck_qwen.sim` 换成 `microduck_qwen.sim_native` 并补 `--manifest runs/vision-simulator.json` 即可。

终端一（重启仿真很重要，旧 manifest 没有启动时的 home）：

```bash
.venv/bin/python -m microduck_qwen.sim --vision --viewer --seed 42
```

终端二（使用已训练的 **agent** 权重）：

```bash
.venv-vlm/bin/python -m microduck_qwen.agent chat --adapter adapters/agent-pilot-v1-r1
```

可以从以下训练任务附近的表达开始；具体可靠程度以本页实测记录为准：

- 前进0.2米，再回到初始位置
- 后退0.2米，再回到初始位置
- 向左转45度，然后向右转45度
- 走到画面左边的球旁边，然后回到初始位置
- 停下来

以上只有 7 条固定模板（本轮新增场景并不在训练分布内）。想换更难的场景，可在仿真命令上加 `--occlusion partial|full|absent`（最左侧目标被遮挡／缺失）或 `--clutter`（5 球，两对同色，必须靠图中编号区分，另加两根柱体）；两者可叠加。采集端用 `--scene-mix hard` 就能把同一批难场景纳入训练，见下文「数据与训练闭环」。

先预览而不执行：

```bash
.venv-vlm/bin/python -m microduck_qwen.agent plan '后退20厘米' --adapter adapters/agent-pilot-v1-r1
```

执行中按 Ctrl+C 停车；也可以另开终端：

```bash
.venv-vlm/bin/python -m microduck_qwen.agent stop
```

`return_home` 回到启动时记录的位置并恢复朝向，途中可以转身前进；这不等于“全程倒着回家”。`move/backward` 才是保持朝向的字面后退。旧 manifest 缺少 home 时必须重启，禁止把当前球旁位置伪装成初始位置。

## 数据与训练闭环

```text
MuJoCo 随机场景 → 专家实际执行 → 位姿/分割/碰撞验证 → 首轮图文 LoRA
       ↑                                             ↓
新训练场景 ← 下一轮模型 ← 混合原始数据再训练 ← 模型提议与专家纠错
```

本版是 **expert-intervened data aggregation**：收集纠错时，模型提出下一步，有限任务 DSL 教师独立给出正确动作；两者不同时先记录原始预测，再由专家接管。Windows 的自动迭代现已加入动作奖励排序与专家反馈回流，见下节。失败专家轨迹整条排除，原始失败保留。它覆盖专家接管后的访问状态，不是任由错误模型自由探索后再标注的完整 DAgger，也不是完整 UI-Genie 复现。

独立物理评估不做上述语义接管：模型动作实际执行后，教师仅评分；只保留一般安全边界。模型输出 `done` 必须出现在全部预定动作已正确执行之后才算任务成功。交互时没有任意自然语言的语义真值，因此只显示 `model_done`，不会声称已经客观证明用户目标完成。

直接从仿真采集（不使用网上合成截图）：

```bash
.venv-vlm/bin/python -m microduck_qwen.agent_data \
  --output data/agent-new-v1 --train-scenes 3 --valid-scenes 1 --test-scenes 1 --seed-offset 500
```

加 `--scene-mix hard` 可换到更难场景（默认 `plain` 复现上面的敞开的 3 球行）：训练场景依次用「同色双球行 + 无遮挡 / 完全遮挡 / 缺失最左球」，验证用「部分遮挡 + 同色双球」，测试用「完全遮挡 + 同色双球」。每行数据与 manifest 都记录 `scene_occlusion` / `scene_clutter`，`agent_eval --rollout` 与 `agent_cycle` 也接同一个 `--scene-mix`，保证采集、训练、评测同分布。

教师策略不变：最左目标不是「完全可见」时直接拒绝生成标签（宁可丢这条任务，也不教一个做不到的动作）。执行层改了两点，让这个任务在遮挡场景里真的能被专家演示出来：绑定目标可见比例 ≥ 0.35 即视为可接近；视觉守卫只在正对目标直行的那一段生效，绕开遮挡柱的绕行段交给 A* 路线，那时目标滑出画面不再中止（详见 [VISION.md](VISION.md) 遮挡场景一节）。

一次启动最多三轮的有预算流程，默认两轮，每轮 120 次优化器更新：

```bash
.venv-vlm/bin/python -m microduck_qwen.agent_cycle \
  --initial-data data/agent-new-v1 --name agent-new \
  --rounds 2 --steps 120 --correction-scenes 1 --seed-offset 1000 --rollout
```

Windows（PyTorch + CUDA）走同一闭包，只是命令不同：采集用
`.venv/Scripts/python.exe -X utf8 -m microduck_qwen.agent_data ...`，单轮技能 SFT 用
`.venv/Scripts/python.exe -X utf8 -m microduck_qwen.vision_torch train --task agent --data <data> --output <adapter> --steps 100`，
技能动作奖励用 `--task reward`；`agent_cycle` 会自动选择 `vision_torch`（Windows）或 `vision_sft`（Mac）。
RTX 3060 8 GiB 上 2B + NF4 QLoRA 实测训练显存峰值：技能任务约 2.35 GiB、奖励任务约 1.91 GiB。

### 奖励辅助自动迭代（Windows）

```powershell
.venv/Scripts/python.exe -X utf8 -m microduck_qwen.agent_cycle --initial-data data/agent-hard-v3 --initial-adapter adapters/agent-hard-v3-sft300 --name agent-reward-v1 --rounds 2 --steps 120 --reward-steps 50 --reward-samples 3 --reward-threshold 0.5 --correction-scenes 1 --seed-offset 3000 --scene-mix hard --reward-guided --rollout
```

数据、权重必须先在本地采集／训练；示例名称不能与已有输出重复。Windows 默认启用奖励迭代，可用 `--no-reward-guided` 回到原专家纠错流程；Mac 默认保持原流程，目前显式启用奖励会在启动前报错。`--initial-adapter` 可选，指定时继续已有技能权重；省略则从基座训练。首轮奖励从头训练，之后加载上轮奖励适配器，但优化器均重新创建。

顺序为：冻结基座 valid → actor SFT → actor valid → 奖励 SFT → 奖励 valid → 可选独立物理评估 → 下一轮奖励辅助纠错采集 → actor/RM 续训。测试集不用于训练、采样或轮次选型；选型规则仍要求 actor 不低于基座，启用 `--rollout` 时还须全部开发物理任务通过。奖励 valid 只报告，不通过这次验证调阈值。

每步保留全部原始候选（第一个贪心，其余 temperature 0.7，采样种子 42），合法动作去重后评分，按最高 yes/no 概率选出提议。低于固定阈值时拒绝所有候选并由专家接管；即使高分，通过教师独立动作校验后才放行。正确候选低分会增加接管，错误候选高分也不会变成成功标签。共享一个 4-bit 基座，交替激活 actor/reward LoRA；独立物理评估仍只使用 actor，不使用奖励排序或教师接管。

训练数据保留经核验的专家纠错动作作为 actor 标签；奖励接受且与专家动作相同的样本标记为 `reward_verified_accepted`，整条轨迹核验成功后，在下一轮 actor 训练中额外回放一次（固定二倍行权重、固定优化步数，不改变 valid/test）。报告记录 `reward_selected_replay_rows`，因此奖励会影响 actor 的训练采样分布，专家纠错示范也保留。同一步的实际错误候选进入 `reward_proposals`，只用于奖励训练，与两个固定构造的反事实负例去重合并。只有整条轨迹通过物理核验才回流；失败轨迹和无效 JSON 保留在原始日志，尚未训练失败结果奖励。valid/test 的输入和标签保持固定，图像／适配器哈希沿用原有校验。奖励提示词版本升为 v2（含可观测记忆），旧版奖励适配器会被拒绝，需要重新训练。

`runs/agent-cycle/<name>/cycle.json` 记录每轮 actor 指标、奖励准确率／正例召回／假接受率及适配器；纠错数据的 `manifest.json` 记录原始采样数、去重数、接受／拒绝、假接受与专家接管次数。奖励附加训练与采样增加计算成本；应与无奖励、等 actor 更新预算的对照比较后再判断收益。

**本机接入自检（2026-10-19）**：在已有 300 步 actor 上继续两轮，每轮 actor 5 步、奖励 10 步，每个观察采样 2 次、阈值 0.5。首轮／次轮 actor valid 为 **16/19、13/19**，同一开发场景的独立物理任务为 **1/4、0/4**，未选出部署权重。两轮奖励 valid 都是 **45/57**，正例召回 **9/19**，错误动作误接受 **2/38**，这次短预算运行没有显示能力提升。

奖励辅助纠错在一个新训练布局采样 34 次、得到 20 个去重候选，8 步接受、9 步拒绝；1 次错误高分被专家拦下，8/9 条专家轨迹通过物理核验。两轮验证了排序、专家反馈和 actor/RM 续训；该次采集早于显式回放标记的加入，因此其中 actor 奖励额外回放行数为 0。随后用最终代码、同一采集种子单独复跑，实际得到 **8 条**可额外回放样本，1 步真实 actor 训练报告确认 **219 + 8 = 227** 条训练行，且未评分的改写不获得奖励认可标记。这是集成自检，不是独立实验。

最终代码 **64 项单元测试通过**；验证／测试输入与标签保持一致，未进行最终测试推理，也没有等预算收益对照。相对路径、小型训练报告、错误高分原始候选及完整指标见 [接入验证记录](experiments/reward-loop-v1/summary.json)。

### Windows 真机自检（2026-10-19）

在 RTX 3060 8 GiB 上按顺序实跑，确认重构后的单链路可用。这是小样本**管线冒烟**，数字不构成能力结论；下文命令省略 `.venv/Scripts/python.exe -X utf8` 前缀，Mac 换成 `.venv-vlm/bin/python` 与 `sim` 入口即可。

| 环节 | 命令要点 | 观察结果 |
| --- | --- | --- |
| 感知 | `sim_native --vision` + `capture`/`propose_balls` | 640×360 帧，红/蓝/绿 3 球全部检出 |
| 采集 | `agent_data --train-scenes 1 --valid-scenes 1 --test-scenes 1 --seed-offset 500` | 27 个专家任务全部 verified，0 次干预 |
| 技能 SFT | `vision_torch train --task agent --steps 150` | 458,752 可训练参数，valid_loss 0.122，峰值 2.35 GiB |
| 奖励 SFT | `vision_torch train --task reward --steps 10` | train_rows 171 = 57×3，峰值 1.91 GiB |
| 奖励评测 | `vision_torch evaluate-reward --split valid` | accuracy 0.70，正例召回 0.32，假接受率 0.11 |
| 离线动作 | `agent_eval --split valid` / `--split test` | 冻结基座 3/19 → 150 步 15/19（valid）、11/19（test） |
| 物理闭环 | `agent_eval --rollout --split test` | 模型输出合法 JSON、技能真实执行到位，再按教师动作判分并停止 |

流程依次做：冻结基线验证、首轮训练、固定验证、独立物理 rollout、新训练场景纠错、继续 LoRA、再次验证。每轮单独进程，避免同时驻留多份 VLM。继续训练加载上轮适配器，**重新创建 AdamW，不恢复优化器状态**。视觉塔与基座冻结，最后 8 层 q/v rank-8 LoRA，共 458,752 个可训练参数；completion-only loss，microbatch 1，累积 1，学习率 5e-5，梯度检查点，序列上限 1024，超长报错而非截断。

原始验证/测试场景与训练场景按 seed 隔离；纠错只增加新训练场景，验证/测试行保持固定。模型输入显式白名单，不接收任务 DSL、目标真值坐标或分割 ID。新旧数据、图像、适配器与配置都有 SHA-256 校验。基于验证集选择只是开发选型，不是最终泛化证明；固定测试集没有用于迭代或选型。

后续新采集默认每个训练意图构造 3 种有限模板改写，使用同一实际成功轨迹对应的图像、历史与动作标签；这些增加的是语言样本，不是独立物理实验。本次初始 pilot 的采集早于该改动，实际行数以审计报告为准。新采集也把验证和测试措辞分开；现有固定 pilot 文件不会被回写修改。

流程不会覆盖已有输出，也不会自动部署权重。只有不低于冻结基线，且启用物理评估时全部小测试通过的轮次，才会被报告为 `selected_adapter`。这也不是安全认证或大规模 benchmark。

## 2026-10-15 本机实测

已完成两轮真实 LoRA，每轮 120 次优化器更新，32 个适配器张量均改变。首轮 42 条训练行来自 3 个训练布局；第二轮在新训练布局 61200 收集 9 个任务、19 条验证通过的纠错行，发生 8 次专家接管，混合训练集增至 61 行。验证集固定 19 行，测试集 14 行未用于推理评估、训练或选型。

固定验证集的合法 JSON + 动作完全匹配：冻结基线 **1/19**，R0 **12/19**，R1 **18/19**。基线有 17 次输出代码块；仅在诊断评分时去掉外层代码块，基线动作匹配为 **6/19**，R0/R1 不变。因此 1→18 不能全部解释为语言理解提升。R1 剩余错误是一个验证画面中的“先左转、再右转”首步被选为右转。

训练以外的同一开发布局 91010，实际执行以下 4 个任务：前进后回家、字面后退后回家、左转后右转、接近图像最左球后回家。R0 **0/4**，R1 **4/4**，R1 四个任务均无场景物体接触步骤。注意这是 **一个布局上的四个任务**，而不是四个独立布局，也不是最终测试结果。

R1 实测前进 0.201 米、后退 0.223 米（要求均 0.2 米），回家最终位置误差分别约 0.106、0.075、0.095 米；接近左侧球的真实最终距离约 0.299 米。左右转实际约 47.6、45.0 度。评分检查动作顺序、各技能实际位姿结果和碰撞统计，不是模型自报 `done`。

另在保存的开发图像上做 6 条无执行诊断：“后退”“后退20厘米”“前进”“左转45度”“停下来”“跳舞”均输出预期动作（跳舞正确输出 unsupported），**6/6 只表示解析检查，不是六次物理实验**。

首轮训练约 266.5 秒，第二轮约 383.5 秒；MLX 峰值均约 **4.90 GiB**，实际序列 502–541 tokens。MLX 峰值不是总系统内存：首轮期间系统已用 swap 从约 5521 MiB 增至 8004 MiB，第二轮从约 7724 增至 7843 MiB；其他应用同时运行，不能把这些数值视为训练独占用量。16GB Mac 上已完成运行，但不声称无交换内存。

初始采集有 13/45 个任务尝试未通过，主要是恢复朝向后越过回家位置误差阈值；失败已保留并排除。随后为回家收紧了导航停车容差（不改变最终验收阈值），新纠错场景 9/9 任务通过。R0/R1 的独立物理评估均使用同样收紧后的控制器。

本次 pilot 尚未使用之后新增的训练措辞增强；R1 提升同时包含更多训练更新和新纠错数据，**没有等训练预算的消融，因此不能单独归因于纠错方法**。小型审计记录见 `experiments/agent-pilot-v1/summary.json`，完整本地记录见 `runs/agent-cycle/agent-pilot-v1/`。推荐仅在上述有限范围内体验 R1，先用 `plan` 检查陌生表达；后续应增加独立布局、语言覆盖和等预算基线。

## 指令到底被读进去了多少（2026-10-19 消融）

对每一行 held-out 数据，用**同一张图、同一段历史**预测两次：一次用它自己的指令，一次换成本，另一个任务的指令。

| split | 自己的指令 | 换错指令 | 指令的净贡献 | 换指令后输出变化 |
|---|---|---|---|---|
| valid | 15/19 | 4/19 | **+11 行** | 12/19 |
| test | 10/16 | 6/16 | **+4 行** | 7/16 |

所以模型**确实在读指令**，不是纯背状态先验。但错得很有规律——按任务类型拆开看：

| 任务类型 | valid 正确率 | test 正确率 |
|---|---|---|
| `move` / `done` / `stop` / `unsupported` / `approach` | 4/4、6/6、1/1、1/1、1/1 | 4/4、5/5、1/1 |
| `return_home` | 1/2 | 0/2 |
| `turn` | 1/2 | 0/2 |
| `clarify` | 0/1 | 0/1 |

**结论：模型的瓶颈不是容量，是"指令有没有第二个分句"。** 凡是第一步动作（`move`）和收尾（`done`）都学会了；凡是需要读出"…然后回到起点 / …再右转 45 度"才该发的动作（`return_home`、`turn`）几乎全错，而且错成 `done`。物理 rollout 0/4 也正是这个：走完第一步就发 `done`，或不区分左右。

原因是任务措辞只有 3 个改写版本，模型可以把「这句话 → 这串动作」整个背下来，不必真的去理解"有没有第二分句"；而 valid/test 用的是**没见过的另一种说法**，这个背下来的映射立刻失效。下一步应先加措辞覆盖和改写数量（让模型无法靠背字符串解决），再考虑更大模型或更多步数。

## 边界与后续

- 模型使用仿真真实机器人位姿；不是仅靠 RGB 的视觉里程计。控制器使用已知静态地图；后退安全不是来自后置摄像头。
- 编号候选由 RGB-D 提案生成，Qwen 负责语言与图像指代、高层动作与顺序，不输出关节角。
- 首轮只有有限中文任务模板、少数布局，主要验证工程闭环；不能据此声称开放词汇、任意长指令泛化。
- 搜索技能代码有界旋转并重新拍摄，但首轮数据未覆盖遮挡主动探索；有代码不等于已经训练可靠。
- 返回起点只验证约 12.5 厘米的位置范围和 12 度朝向范围，不是精密重定位。
- `vx` 是步态条件命令，不是实际米/秒。已观察到 -0.30 几乎不后退，-0.40 可后退；仅在本仿真策略上做了校准，不适用于真机安全保证。
- 自动循环只处理有限任务教师能验证的目标。下一阶段应加入距离/角度课程、更多语言改写和布局、遮挡搜索、失败反馈状态，以及独立任务结果评判；之后才考虑训练 RM 或高层 PPO/GRPO。

运行数据、模型和轨迹在本地 `data/`、`adapters/`、`runs/`，不上传 GitHub。源码和小型审计结果可以同步；RTX 3060 8GB 的 CUDA QLoRA 训练器已经接入（`vision_torch train --task agent|reward`），与 MLX 共用同一份 `agent_data` 数据与技能 schema，两套权重不互换。
