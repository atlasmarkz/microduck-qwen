# 2026-10-31 本机独立物理评测

在一个 hard 场景（seed 93010，部分遮挡、同色干扰）上使用 `agent-yellow-v1`，逐项独立执行固定验证指令；教师只在执行后评分，不接管模型。四项结果详见 [报告](report.json) 和 [任务 0](task-00.json)、[任务 1](task-01.json)、[任务 2](task-02.json)、[任务 3](task-03.json)。

原评测器报告 3/4，四项均记录零场景物体接触。第四项模型的 `leftmost` 实际绑定教师指定的绿色球 ID 4，并成功接近，但因动作 JSON 与教师的 `id=4` 不完全相同被判错，评测随即中止；无法评价后续返程。不要把此项算作错误目标执行，也不要把它算作完整成功。

此批任务不含交互和故障注入，不足以计算澄清率、非必要提问率、恢复率或不应执行时行动率。物理状态与地图仍使用仿真真值，样本量不能支持总体性能推断。

```powershell
.venv/Scripts/python.exe -X utf8 -m microduck_qwen.agent_eval --data data/agent-yellow-mixed-v1 --adapter adapters/agent-yellow-v1 --output runs/local-review-reproduce --rollout --scene-mix hard --seed 93010 --port 8062
```

权重、数据和上游仿真资源需在本地准备，输出目录必须不存在。`collect.py` 从本次原始运行提取相对路径报告；完整轨迹与图像保存在本机 `runs/local-review-20261031`。
