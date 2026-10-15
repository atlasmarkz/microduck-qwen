# MicroDuck × Qwen-VL 训练计划

后续进展（2026-10-15）：已在本机完成多技能 agent 的两轮专家纠错 LoRA 与小规模物理评估，见 [AGENT_TRAINING.md](AGENT_TRAINING.md)。本文件保留 10 月 14 日计划与 CUDA 后续路线；学习型 RM、高层在线 RL 仍未实现。

更新：2026-10-14（香港时间）。当前先在 M4 Mac 做小模型图文 LoRA；RTX 3060 台式机作为下一阶段训练设备。本文件区分“已实测”“计划”“尚未实现”，不把模型能加载等同于能训练，也不把内存测试当作能力评测。

## 1. 硬件与当前结论

当前执行环境实测为 MacBook Air、Apple M4、10 核 GPU、16 GiB 统一内存，MLX 设备报告总内存 17,179,869,184 bytes。没有可访问的 NVIDIA GPU，也没有 `nvidia-smi`。这是共享给 CPU、GPU 和系统的内存，不是 16GB 独立显存。

台式机按 Hechuan Zhang 提供的信息记为 RTX 3060、8GB 显存；尚未远程连接、验证型号、驱动、主机 RAM 或训练内存。后续先运行 `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` 核验。不要把 Mac 实测内存直接当作 CUDA 显存需求。

已完成 **Qwen3-VL-2B 4-bit 图文 LoRA 内存测试**，并新增第一轮真实仿真采集、图文 SFT 与适配器控制 pilot，见 [SIM_TRAINING.md](SIM_TRAINING.md)。独立最终测试、多轮模型探索、学习型 RM 和高层 RL 尚未完成。下面第 2 节保留此前 3 步内存探针的历史结果，不与新一轮任务训练混淆。

## 2. 本轮 Mac 实测

可复现入口：`microduck_qwen/vision_lora_probe.py`。使用仓库内一张 640×360 MuJoCo 标注图，人工核对“画面左球是蓝色、编号 3”；输入仍为图像与自然语言，真值仅作为 assistant 监督标签，不作为模型输入。这一历史 smoke-test 场景现在属于开发数据，必须从将来的独立测试中排除。

配置：

- 基座 `mlx-community/Qwen3-VL-2B-Instruct-4bit`，固定 revision `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`。
- MLX 0.32.3、MLX-VLM 0.7.8、Transformers 5.19.0；现有 `.venv-vlm`，未升级运行环境。
- 冻结基座和视觉编码器；语言 Transformer 最后 8 层的 `q_proj`、`v_proj` 加 LoRA，rank 8、alpha 16、dropout 0；可训练参数 458,752。
- microbatch 1；AdamW，学习率 1e-4，weight decay 0；梯度检查点开启，随机种子 42。
- 配置最大序列长度 768；**实际输入 336 tokens，padding 后 352，其中图像 tokens 220、监督答案 tokens 15**。768 长度的训练尚未实测，不能用本结果保证其内存相同。
- 单样本重复 3 次更新，assistant completion-only loss；没有 train/valid/test 能力评测。
- 保留 MLX 原有 4-bit 表示，只训练浮点 LoRA；属于量化基座 + LoRA，**不是 CUDA bitsandbytes/NF4 的同一实现**。

结果：

- 模型加载后 MLX 活跃分配约 1.782 GB。
- 训练过程 MLX 活跃分配峰值 **5,217,584,355 bytes = 5.218 GB = 4.859 GiB**。
- 第 3 步后 MLX 缓存约 0.271 GB；这不是峰值缓存，也不能与活跃峰值相加来声称同步总峰值。
- Darwin `ru_maxrss` 进程峰值 RSS 为 844,087,296 bytes，约 0.786 GiB。它与 MLX/Metal 分配口径不同，不能当作训练总内存，不能与 MLX 指标相加。
- 完整流程约 22.42 秒；3 次更新分别约 6.64、4.00、4.28 秒。单张重复样本速度不能外推为多场景正式训练速度。
- 32 个 LoRA 张量发生变化，权重保存成功；同一训练样本 loss 为 1.2191 → 1.1111 → 0.8448。它只证明优化器在更新参数，**不证明模型已学会泛化控制**。

**重要限制：**成功测试开始前，全机已使用约 11,842 MiB swap；结束时约 14,093 MiB，增加约 2,251 MiB（2.20 GiB）。两套仿真和其他应用仍在运行；这些是全机统计，不能全部归因于训练，但说明测试并非低压力、无交换的环境。本轮没有关闭用户程序，也没有修改系统内存设置。正式长训前应由用户关闭不需要的仿真/大模型/浏览器负载，确认内存压力下降后，再跑 10–20 步确认 swap 不持续增长。现有结果只支持“此配置短程可运行”，不支持“16GB 上可以无压力长训”。

MLX 的 `set_memory_limit(8 GiB)` 是分配建议值，不是硬性内存上限或防 OOM 保证；缓存上限设置为 256 MiB。原始成功和失败记录均保留在 `experiments/memory/`。首次尝试因上游配置使用 `image_token_id` 而非 `image_token_index`，在更新参数前失败；脚本现已兼容两种字段并增加测试。测试结束后，当前 fixture 增加了图像哈希校验字段；运行时的原始 fixture 单独归档为 `20261014-probe-fixture-at-run.json`，其 SHA-256 与原始报告一致，图片及监督标签没有改变。归档 JSON 仅用于审计，重跑请用 `tests/fixtures/` 的当前文件。

重跑命令（每次自动创建新输出目录，不覆盖历史权重）：

```bash
cd /Users/mark/Code/microduck-qwen
.venv-vlm/bin/python -m microduck_qwen.vision_lora_probe --steps 3
.venv-vlm/bin/python -m unittest discover -s tests -v
```

新机器先按 `VISION.md` 重建环境并执行 `vision download`。内存测试适配器保存到 `adapters/vl-memory-probe-*/`，不接入 `vision chat`，也不上传 GitHub。该入口是受限内存探针，不是通用数据集训练器；增加步数只会重复同一张图。

## 3. 要训练什么，以及不训练什么

链路：图像 + 指令 + 必要的历史观察 → 后训练 Qwen-VL 的高层决策 → 目标定位/路径控制 → 官方 RL 步态 → MuJoCo MicroDuck。

第一阶段训练视觉指代、颜色/编号绑定、左右关系和不确定时拒绝猜测。初期保持现有 JSON `{status, color, id}`，先在不改控制接口的情况下比较冻结模型与图文 SFT。候选框属于有检测器辅助的设置，必须明确报告，不能声称模型从原始像素端到端学会了全部感知。

“左边的球”是选择对象；“走到绿色球的右边”是相对目标位置，**两者不同**。当前视觉接口只覆盖前者及颜色目标选择，不支持后者。下一阶段再新增明确的 action/offset schema、相机/机器人/世界坐标系约定和导航执行测试，不能指望只训练 LoRA 就让现有执行器自动支持新动作。

官方已有 RL 步态作为低层控制器保持冻结。本阶段不重新训练走路、不让 VLM 直接输出关节角，不做端到端 VLA，也不把高层 SFT 称作机器人步态 RL。第一版仍使用仿真位姿、标定 RGB-D 和已知安全地图，必须在结果里注明 oracle 定位/地图条件。

## 4. 数据与监督信号

计划扩展到约 300–500 个训练场景、50 个验证场景、100 个独立测试场景；每场训练图像配 3–6 条指令，得到约 1,000–3,000 条图文样本。数量不是质量或论文成果保证。目前仅有 6/3/3 个 train/valid/test 场景与 124/21/21 条样本的初轮 pilot，**尚未达到这一扩展规模**。

按场景 seed、布局、轨迹、渲染配置分组切分，再生成指令；同一布局的相邻帧、同一图像的同义句和编号重排不得跨分割。测试场景和标签不得用于采样失败、修改提示词或选训练参数。训练只用新划分的 train；调参与失败挖掘只用 valid。

数据覆盖颜色指令、左右指令、同色多个对象、随机编号、不同距离、部分遮挡、目标不可见、含糊指代和停止指令。当前场景生成器尚无同色多球/复杂指代，需要单独扩展与测试。标签由仿真对象投影、可见性与人工抽查产生；检测失败要单独标记，不能把检测器输出一律当真值，更不能直接把 VLM 原答案当正确监督。

从未见过且完全不可见的目标，在单帧中通常不能区分“被挡住”与“不存在”。单帧统一监督 `not_visible`/不行动；只有具备历史观察和可验证遮挡证据时，才监督具体遮挡判断。不要向模型输入隐藏物体坐标来制造虚假的视觉能力。

训练最小清单应记录：样本 ID、scene/episode ID、图像相对路径及 SHA-256、指令、人工/程序标签来源、输出 JSON、split 和数据版本。相机位姿/深度/隐藏物体信息可以留在评分侧，不进入纯 RGB 模型输入。

## 5. Mac 正式 SFT 起步配置与验收

优先沿用已测试的 2B 4-bit、microbatch 1、最后 8 层 q/v LoRA rank 8；保持视觉编码器冻结。正式训练器支持梯度累积 4–8（有效 batch 4–8）和每次完整答案的 loss mask；未实测前不要开启全部线性层、视觉塔解冻、多图或长历史。

先限制单图和实际 token 数，记录 processor 生成的真实图像网格，不只看 JPEG 分辨率。以当前约 336 tokens 为内存起点；更高分辨率、更长指令和序列上限都要重新测最坏样本。若样本超限，显式报错或调整图像/指令，不允许静默截断监督答案。

正式 SFT 学习率先验证 2e-5、5e-5 两档，rank 8/16 和最后 8/16 层在资源允许后比较。3 步探针的 1e-4 只用于确认更新，不是正式超参数结论。先 10–20 步稳定性检查，再做 100–300 步 pilot，以验证集目标选择/拒答指标选 checkpoint。不要自动追加长训；先检查温度、内存压力、swap 增长及验证结果。

接入控制前验收：图文梯度与非空 completion mask 正确；只更新指定 LoRA；权重保存、重新加载后预测一致；冻结基线可复现；适配器显式选择，日志记录基座 revision、adapter 哈希、prompt 和数据版本。旧的冻结 `vision chat` 默认行为保持不变。

## 6. RTX 3060 8GB 后续路径

建议在 Linux 或已验证的 WSL2/CUDA 环境使用 PyTorch + Transformers + PEFT + bitsandbytes 4-bit LoRA。官方 Qwen3-VL 微调代码用于核对模型输入、图像处理和训练约定；**不是把官方大显存训练脚本原样套在 8GB 上**。MLX 环境不能直接移植到 CUDA，现有 `pyproject.toml` 也不是 CUDA 安装配置；后续应单独添加 `requirements-cuda`、训练器和经过验证的版本锁定。

起步仍选 Qwen3-VL-2B，不承诺 7B 或更大模型能在 8GB 上训。先 4-bit NF4 + double quantization，冻结视觉塔，q/v rank 8，microbatch 1、梯度累积 4–8、梯度检查点、`use_cache=False`；FP16 或 BF16 根据设备和训练库实际支持测试选择。以上是**待验证配置**，不是已经证明能放进 8GB。

CUDA 验收同时记录 `torch.cuda.max_memory_allocated()`、`max_memory_reserved()`、`nvidia-smi` 进程占用、CPU RAM、实际 tokens、步速和 OOM。即使 2B 也可能因激活/分辨率超限。失败时依次降低图像 token、序列长度和 LoRA 覆盖层数；microbatch 已为 1 时不能靠梯度累积降低单步峰值。不要通过静默 CPU offload 或频繁交换来宣称“8GB 原生稳定训练”。

MLX 与 PEFT 的命名、缩放、量化格式和 adapter 元数据可能不同。原始图文数据、任务 schema、scene split 可以共享；权重不能假定直接互载。默认在 CUDA 上重训相同实验；若要转换适配器，必须做参数映射与 logits/任务预测对齐测试。

## 7. 实验与 RL 的先后顺序

先做冻结 VLM、相同提示词的图文 SFT、简单颜色/左右规则三组。规则作为透明基线而非模型训练成果。模型层统计原始输出 schema 合法率、颜色/编号绑定正确率、目标选择正确率、不确定场景错误执行意图率和拒答率；先记录原始预测，再统计安全层是否拦截，不能把 safety guard 的成功算作 VLM 自身判断正确。

再做相同 scene seeds、指令和控制器下的完整导航比较：正确目标到达率、碰撞/跌倒、目标距离、路径长度、时延与安全停止。相对侧方停靠另设成功条件，不与接近目标混在一起。独立测试先覆盖约 100 场，训练至少 3 个随机种子；按场景报告置信区间，不能把同一图像的多条同义指令当完全独立样本。小规模数据只支持 pilot 结论。

只有 SFT、adapter 载入和闭环评测稳定后，才研究高层 RL 或偏好优化。可先用仿真规则检查后的成功/失败轨迹做离线纠错 SFT，或构造偏好对；DPO 不能直接等同于已完成在线环境 RL。后续 RL 才用任务完成、进展、碰撞、跌倒、超时等信号优化高层决策，保持官方步态冻结。必须审计奖励投机，避免靠一直拒绝动作拿高安全分；完整在线 RL 对 8GB 显存尚无可行性保证。

## 8. 时间安排与阶段门槛

以下从下一次工作日开始按工作量估算，不是假定会议/投稿截止日期，现阶段不建议直接承诺投稿档期。

1. 第 1–2 天：本轮内存探针已完成；清理并发负载后确认稳定性，完善图文样本导出与按场景划分。
2. 第 3–5 天：小规模图文数据集、标签抽查、冻结模型离线基线；接通正式 SFT 保存/加载，不改变低层步态。
3. 第 2 周：100–300 步 SFT pilot，验证超参数与内存；接入显式 adapter，完成冻结/SFT 的配对导航评测。若无改善，先诊断数据和目标，而非立即上 RL。
4. 第 3–4 周：补充同色多球、布局/光照变化和有历史证据的遮挡；定义相对位置动作及执行器。条件允许再在 3060 上复现实验。
5. 第 5–6 周：根据 SFT 失败模式决定是否开展离线偏好/高层 RL，做消融与统计；在线 RL 资源不足时保留为后续，不包装成已完成贡献。

历史内存测试轮交付包括计划、内存脚本与样本；现新增仿真采集器、场景隔离的数据、图文 SFT、显式适配器载入和初轮控制验证，见 [SIM_TRAINING.md](SIM_TRAINING.md)。CUDA 训练器、任务扩展和多轮自改进仍待实现。

## 9. 一手参考与统计口径

- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)：Mac 上 VLM 加载与训练库；实际使用固定本地版本 0.7.8，而非未经验证的 main。
- [Qwen3-VL 官方微调目录](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)：CUDA 路径与数据/模型输入约定参考。
- [PEFT 量化与 LoRA 文档](https://huggingface.co/docs/peft/en/developer_guides/quantization)：量化基座配合可训练适配器，以及 bitsandbytes NF4 配置。具体 8GB 可行性仍需实测。
- [MLX get_peak_memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html)：MLX 分配峰值统计，不是整机总内存。

单位统一：GB = 10^9 bytes，GiB = 2^30 bytes；`sysctl vm.swapusage` 的 M 值按 MiB 换算。保留原始 bytes，避免把推理内存、训练峰值、RSS、缓存和系统 swap 混为同一指标。
