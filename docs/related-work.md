# 借鉴的论文与做法

只读到摘要或搜索摘要的论文标注「摘要级」；做法是参考其思路，按本项目的仿真与技能接口改写，不是复现。标注 2026 的预印本没有逐篇核实，引用前请核对原文。

## 当前要解决的问题与对应做法

| 问题 | 论文 | 我们借鉴了什么 | 状态 |
|---|---|---|---|
| 探索时「没有可达前沿」就宣告失败 | [SEER](https://arxiv.org/pdf/2209.11034)（摘要级） | 检测到的门洞先标「待确认」，走到才确认；走到最近的可达点再观察一次 | 已实现并评测：`mapping.vantage_targets`、`explore.Explorer.recover`，最多 2 次；6 个任务触发，1 个因此找到了球，总分无变化（69/80 对 69/80，p=1.0）；新布局上的干净开关对照（73 个有效任务）：恢复开 61，关 60，p=1.0 |
| 同上 | [模式切换前沿搜索](https://sgvr.kaist.ac.kr/publication/mode-switching-frontier-search-for-robust-autonomous-exploration-in-complex-terrain/)（摘要级）、[前沿可达性分析](https://scholar.hit.edu.cn/en/publications/a-graph-based-method-for-efficient-frontier-detection-and-travers/)（摘要级） | 卡死时进入专门的恢复模式，而不是继续用同一种规划 | 同上；轨迹历史式卡死检测没做 |
| 走近球时一停就结束整条任务 | [RePLan](https://ar5iv.labs.arxiv.org/html/2401.04157)（摘要级）、[EMAC+](https://arxiv.org/pdf/2505.19905)（摘要级） | 技能失败后交给恢复步骤，而不是直接结束；去掉执行反馈后会反复重复失败动作 | 已实现并评测：`skills.SkillExecutor._approach` / `_reacquire`，最多 2 次恢复；近距离停车时有效（救回 1 个），3.5 米外目标丢失时无效（2 个没能重新找到）；新布局上又在同一个布局的 3 个任务里触发并全部成功，但这是同一几何，不算 3 个独立样本 |
| 同上（最后一段目标丢失） | [MG-Nav](https://arxiv.org/html/2511.22609v1)（摘要级）、[检测驱动的目标修正 + 视觉伺服](https://cdn.amazon.science/57/24/5371b0584b20a9a9178cc0f2b918/cvprw-2026-camera-ready-compressed-camera-ready.pdf)（CVPR 研讨会 2026，摘要级） | 最后一段重新定位；检测时有时无时，用新的检测刷新目标 | 已实现：按位置重新绑定同一个球（`skills.rebind_near`），不换成别的球 |
| 重训后某类行为被悄悄破坏 | [CL-VISTA](https://arxiv.org/pdf/2604.00677)（摘要级）、[MLLM 数据混合](https://arxiv.org/pdf/2603.14493v1)（摘要级） | 按任务分别度量保留情况；窄任务微调要混入旧数据 | 已实现：`agent_canary --replay`，用旧评测走过的 226 个真实状态重新决策，按「上一步 → 期望动作」分类；同时抓到了 `lang-v2`（`move->done` 16→12）和 `lang-v3`（`explore->approach` 44→41）的回归；混入旧数据早已通过 `--base-data` 做 |
| 权重插值减轻回归 | [WiSE-FT](https://arxiv.org/pdf/2109.01903v3.pdf)（Wortsman 等） | 微调前后权重线性插值 | 未做：结果来自 CLIP 全量微调，对 LoRA 要插值权重增量，没有验证过 |
| 评测分不出 2–3 个百分点的差别 | McNemar 精确检验（成对二元结果的标准做法） | 只有「一个通过、一个失败」的任务提供差别的信息 | 已实现：`compare_runs`；留出说法增加 3 组变体（`agent_eval --phrase-variant`）。用它回看旧结果：52→69 显著（p=0.002），之后各步分不出 |

## 读过但没有采用的

| 论文 | 原因 |
|---|---|
| [VLFM](https://arxiv.org/pdf/2312.03275)、[ESC](https://arxiv.org/pdf/2301.13166)、[LGR](https://arxiv.org/pdf/2503.20241)（语义前沿价值） | 依赖「床在卧室」这类常识先验，而本项目的球是随机撒在各房间里的；先验帮不上多少。房间级信息（优先没去过的房间）留作以后 |
| [失败智能体 DPO](https://arxiv.org/html/2511.22254v2)、[TCPO](https://arxiv.org/html/2509.08500v1) | 模型类失败最多只占约 5–6 个百分点，而且训练数据很少；先做防回归检查 |
| [VLN-R1](https://arxiv.org/html/2506.17221v2)、[SRGPO](https://arxiv.org/pdf/2512.02631)（GRPO） | 每条任务约 2 分钟、显卡 8 GB，采样成本过高 |
| [TravExplorer](https://arxiv.org/pdf/2605.19958)（摘要级） | 面向跨楼层三维可达性；其「前沿耗尽只占约 2% 失败」的数字只作参考 |

## 更早的阶段读过的

ApexNav（目标中心的语义融合、疑似目标、语义与几何前沿切换）：疑似目标已用于 `explore` 的 `suspects`。SOAR（事后重标注、成功检测）：评估后认为价值低，未采用。
