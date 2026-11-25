"""Notebook views of what Pollen Robotics' MicroDuck repositories contain (no GPU, no mjlab).

Task facts come from upstream_tasks.json, exported from the live registry of the pinned
microduck_rl commit by tools/export_upstream_tasks.py. Only the Chinese wording lives here.
"""

import base64
import html
import json
from pathlib import Path
import subprocess
import sys

from . import ROOT

SNAPSHOT = Path(__file__).with_name("upstream_tasks.json")
TRAINING_REPO = "pollen-robotics/microduck-training"

# family -> (Chinese title, what it learns, how it is built, flat task id, published ONNX, training bundle).
# ONNX and bundle pairs are the ones pollen-robotics/microduck-training's README states (2026-11).
FAMILIES = {
    "Velocity": ("速度行走", "跟踪前后、横移、转向速度指令，同时能控制头部姿态。主任务，其他任务大多以它的配方为底板。",
                 "15% 的环境专门练原地转身；头部除了跟踪奖励，还惩罚时间平均的低头偏差。",
                 "Mjlab-Velocity-Flat-MicroDuck", ("alpha_walking.onnx",), "velocity2/441tzs6d_3750"),
    "VelStand": ("走站起一体", "一个网络同时负责走、站、保护性摔倒和倒地起身。真机默认步态。",
                 "从站立专家做行为克隆，并以行走策略为锚点；起身用势函数奖励，躺着不动拿不到分。",
                 "Mjlab-VelStand-Flat-MicroDuck", ("velstand.onnx",), "velstand/0045i2si_2749"),
    "StandUp": ("起身站立", "从趴、仰、坐三种姿势站起并站稳，可控制身体姿态。",
                "只奖励单一目标姿势，不给中间路点；起身方式由策略自己发现。",
                "Mjlab-StandUp-Flat-MicroDuck", ("alpha_stand.onnx",), "microduck_stand/69u48n8l_9750"),
    "SitStand": ("坐下起立", "一个姿势标志位控制坐或站，动作轻柔，坐着也能动头。",
                 "奖励跟踪匀速移动的内部目标，抢跑不得分，所以慢就是最优。",
                 "Mjlab-SitStand-Flat-MicroDuck", ("alpha_sitstand.onnx",), "microduck_sitstand/vfhxgrum_2250"),
    "GroundPick": ("低头啄地", "蹲下让嘴尖尽量靠近地面但不撞地，再站回来。",
                   "速度指令槽改放相位 [cos, sin]，奖励按下探、回升分段开关；回升时给嘴尖加负重。",
                   "Mjlab-GroundPick-Flat-MicroDuck", ("alpha_ground_pick.onnx",), "ground_pick/4ly1pcxj_2750"),
    "BallKick": ("踢球", "站着把 70 mm、15 g 的球踢向前方，然后站稳。左右脚各训一个。",
                 "策略看不到球，只有评论家能看到；靠球位随机化获得鲁棒性。",
                 "Mjlab-BallKick-Flat-MicroDuck", ("ball_kick_left.onnx", "ball_kick_right.onnx"),
                 "ball_kick_left/6bamlid8_2500、ball_kick_right/wy2x3p87_2000"),
    "Roulade": ("前滚翻", "越过头顶向前滚一圈，落回双脚站立。",
                "只有贴地时的旋转才计分，超速旋转被罚；从滚动中途出生做反向课程；开启左右镜像损失。",
                "Mjlab-Roulade-Flat-MicroDuck", ("roulade.onnx",), "microduck_roulade/blrqbcit_750"),
    "Rollers": ("轮滑", "脚下装被动轮，跟踪前进速度，学的是交替蹬冰。",
                "被动轮插在关节序列中间，所有关节按名字解析；另随机化轮轴摩擦。",
                "Mjlab-Velocity-Flat-MicroDuck-Rollers", ("roller.onnx",), ""),
    "Swizzle": ("轮滑葫芦步", "双刀不离地，两腿对称开合，前进或后退。",
                "复用轮滑任务，去掉反葫芦步的奖励，换成左右对称和双刀贴地。",
                "Mjlab-Velocity-Swizzle-MicroDuck", (), ""),
    "RollerCrouch": ("轮滑下蹲滑行", "带着速度蹲下滑行约 1 秒，再站起来。",
                     "相位时钟驱动站姿与蹲姿之间的插值目标。",
                     "Mjlab-RollerCrouch-Flat-MicroDuck", ("roller_crouch.onnx",), "roller_crouch/92d20zda_500"),
    "RollerSlope": ("轮滑下坡", "被动滑下斜坡并保持站立，不接受驾驶指令。",
                    "平地接坡道的自定义地形，坡度随课程变陡。",
                    "Mjlab-RollerSlope-Flat-MicroDuck", (), ""),
    "RollerStandUp": ("轮滑起身", "穿着轮子从地上站起来并站住。",
                      "把起身任务的配方搬到轮滑模型上。",
                      "Mjlab-RollerStandUp-Flat-MicroDuck", (), ""),
    "Spin": ("轮滑原地旋转", "约 3 rad/s 逆时针转一圈，然后停稳。",
             "相位驱动目标转速，两项逐渐衰减的引导奖励把策略推向左右轮差速。",
             "Mjlab-Spin-Flat-MicroDuck", (), ""),
}

# Chinese gloss per reward function; upstream's own docstring is kept in the snapshot's reward_docs.
REWARD_ZH = {
    "action_over_limit_penalty": "指令超出关节硬限位", "action_rate_l2": "动作变化率（平滑）",
    "action_rate_l2_fallen_scaled": "动作变化率，倒地时缩放", "angular_momentum_penalty": "全身角动量",
    "apply_mouth_payload_force": "给嘴尖施加负重（挂钩，本身不计分）", "ball_forward_velocity": "球的前向速度，封顶",
    "ball_speed_overshoot_penalty": "球速超过目标", "body_ang_vel_at_height": "到站高后的躯干角速度",
    "body_angular_velocity_penalty": "躯干角速度", "body_impact_cost": "部件与地面的冲击力",
    "body_pose_tracking_6d": "跟踪 6 维身体姿态指令", "body_pose_tracking_locomotion": "行走中跟踪身体姿态指令",
    "body_upright_gaussian": "躯干竖直（高斯）", "body_upright_linear": "躯干竖直（线性，任意倾角都有梯度）",
    "braking_reward": "收到刹车指令时停下", "com_height_target": "质心保持在目标高度带",
    "com_upward_velocity": "质心上升速度", "crouch_forward_lean": "蹲下时略前倾",
    "crouch_glide_pose_by_phase": "跟随相位插值的蹲/站姿势", "crouch_glide_pose_l1": "蹲/站姿势的 L1 引导",
    "fallen_state_penalty": "处于倒地状态的固定税", "feet_air_time": "抬脚时间合适",
    "feet_air_time_upright": "抬脚时间，倒地时为零", "feet_clearance": "摆动脚离地高度",
    "feet_flat_penalty": "脚底不平", "feet_grounded_reward": "双脚着地", "feet_slip": "脚打滑",
    "feet_swing_height": "落脚时检查摆动高度", "forward_lean_reward": "蹬冰时略前倾",
    "forward_speed_reward": "保持前向速度", "gait_symmetry_penalty": "左右脚使用不均",
    "glide_reward": "单刀滑行且腿安静", "ground_pick_return_pose_phased": "回升阶段恢复站姿",
    "ground_pick_return_upright_phased": "回升阶段恢复竖直", "grounded_reward": "双刀贴地",
    "head_pose_bias_penalty": "头部的长期偏差（1 s 均值）", "head_pose_tracking": "跟踪头部姿态指令",
    "heading_hold_reward": "保持出生朝向（走直）", "heading_tracking_reward": "跟踪指令朝向",
    "height_l1_penalty": "高度误差 L1", "height_progress": "高度进步（势函数）",
    "height_target_gaussian": "躯干到目标高度", "is_alive": "未终止", "joint_deviation_l1": "关节偏离默认姿势",
    "joint_pos_limits": "关节越过软限位", "joint_torque_rate_l2": "力矩变化率（护齿轮）",
    "joint_torque_rate_l2_fallen_scaled": "力矩变化率，倒地时缩放", "joint_torques_l2": "力矩大小（省能）",
    "leg_antisymmetry": "旋转时两腿前后剪开", "leg_symmetry_reward": "两腿左右镜像",
    "mouth_ground_proximity_phased": "下探阶段嘴尖靠近地面", "mouth_perpendicular_phased": "下探阶段嘴朝下",
    "neck_action_rate_l2": "颈部动作变化率", "neck_joint_pos_l2": "颈部偏离默认",
    "neck_torque_when_tilted": "身体倾斜时颈部受力", "neck_vel_descent_penalty": "下探时颈部过快",
    "pose_l1_penalty": "姿势误差 L1", "pose_target_match": "匹配固定目标姿势",
    "posture_composite": "坐/站综合达成分（乘积）", "posture_height_gaussian": "坐/站目标高度",
    "posture_height_l1": "坐/站高度误差 L1", "posture_pose_l1": "坐/站姿势误差 L1",
    "posture_pose_match": "坐/站目标姿势", "posture_rise_bootstrap": "起立初期的上升速度",
    "posture_stillness": "到位后保持静止", "recovery_success": "完成一次起身的一次性奖金",
    "roulade_flatness_penalty": "滚动时侧轴不水平", "roulade_head_pivot": "滚动中以头顶支撑",
    "roulade_height_after_roll": "滚完后的高度", "roulade_landing_composite": "滚完后的站立综合分",
    "roulade_landing_sharp": "滚完后的精确站立", "roulade_lateral_velocity_penalty": "侧向漂移",
    "roulade_overspeed_penalty": "滚得太快", "roulade_progress": "滚动进度（最多一圈）",
    "roulade_rise_velocity": "滚动末段起身速度", "roulade_sagittal_penalty": "偏离矢状面的旋转",
    "roulade_stand_tax": "滚完仍未站高的税", "roulade_upright_after_roll": "滚完后竖直",
    "self_collision_cost": "自碰撞", "servo_acc_spike_penalty": "舵机加速度尖峰",
    "servo_stall_penalty": "舵机顶死不动", "single_foot_grounded_reward": "支撑脚不离地",
    "single_support_reward": "单脚支撑（蹬冰步）", "skating_air_time_reward": "蹬冰时的抬脚时间",
    "soft_landing": "落脚冲击", "spin_grounded": "旋转时双刀贴地", "spin_rate_l1": "转速误差 L1 引导",
    "spin_rate_track": "跟踪目标转速", "spin_stay_in_place": "旋转时平移",
    "spin_wheel_differential": "靠左右轮差速旋转", "standing_composite_score": "站立综合分（三项乘积）",
    "track_angular_velocity": "跟踪转向指令", "track_linear_velocity": "跟踪线速度指令",
    "track_yaw_rate_fine": "精细跟踪转向速度", "trunk_downward_velocity_penalty": "下降太快",
    "trunk_upward_velocity_penalty": "上升太快", "trunk_vertical_accel_penalty": "躯干竖直加速度（轻柔）",
    "upright": "躯干竖直", "upright_body_cmd_relative": "相对身体指令的竖直",
    "upright_gaussian_at_height": "站高后才计分的竖直", "upright_progress": "变直的进步（势函数）",
    "upright_while_tall": "站高时保持竖直", "variable_posture": "接近默认姿势（宽容度随速度变）",
    "variable_posture_body_relaxed": "接近默认姿势，有身体指令时放宽",
    "wheel_glide_reward": "轮子向前滚动，封顶", "wheel_speed_reward": "按指令方向转轮",
}

OBS_ZH = {"base_ang_vel": "陀螺仪角速度 3", "projected_gravity": "重力方向 3", "joint_pos": "关节位置 14",
          "joint_vel": "关节速度 14", "actions": "上一动作 14", "command": "运动指令 3",
          "head_command": "头部指令 4", "body_command": "身体指令 6"}
ROBOT_ZH = {"robot_walk.xml": "行走模型（躯干和头部碰撞已简化）", "robot_groundcontact.xml": "触地模型（身体可躺地）",
            "robot_allcollisions.xml": "全碰撞模型", "robot_groundcontact_rollers.xml": "轮滑模型（4 个被动轮）"}
COMMAND_ZH = {"VelocityCommandCommandOnlyCfg": "速度指令", "RelativeHeadingVelocityCommandCfg": "速度 + 相对朝向指令",
              "GroundPickPhaseCommandCfg": "相位时钟 [cos, sin, 0]", "SitStandCommandCfg": "姿势标志（0 站，1 坐）",
              "UniformPoseCommandCfg": "姿态偏移", "StandingGatedPoseCommandCfg": "姿态偏移（仅站立时）"}

STYLE = ("<style>.duck{font:14px/1.6 system-ui,sans-serif;max-width:1080px}"
         ".duck table{border-collapse:collapse;margin:8px 0}.duck th,.duck td{border:1px solid rgba(128,128,128,.4);"
         "padding:4px 10px;text-align:left;vertical-align:top}.duck .num{text-align:right;font-variant-numeric:tabular-nums}"
         ".duck small{opacity:.75}</style>")


def load():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def variants(snapshot, flat_id):
    """(base ids, backlash ids) of the family whose flat task is flat_id."""
    rough = flat_id.replace("-Flat-", "-Rough-")
    base = [task["id"] for task in snapshot["tasks"] if "base" not in task and task["id"] in (flat_id, rough)]
    return base, [task["id"] for task in snapshot["tasks"] if task.get("base") in base]


def table(head, rows, numeric=()):
    def cells(row):
        return "".join(f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>" for i, c in enumerate(row))

    return ("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join(f"<tr>{cells(row)}</tr>" for row in rows) + "</table>")


def code(names, separator="、"):
    return separator.join(f"<code>{html.escape(name)}</code>" for name in names)


def value_text(value):
    if value is None:
        return "未达到"
    if isinstance(value, bool):
        return "是" if value else "否"
    return f"{value:g}"


def task_table_html():
    snapshot = load()
    by_id = {task["id"]: task for task in snapshot["tasks"]}
    rows = []
    for key, (title, goal, _how, flat_id, policies, bundle) in FAMILIES.items():
        base, backlash = variants(snapshot, flat_id)
        task = by_id[flat_id]
        terrain = "斜坡" if task["terrain"] == "slope" else "平地 + 崎岖" if len(base) > 1 else "平地"
        rows.append((f"<b>{title}</b><br><small>{key}</small>", goal, terrain, code([task["robot_xml"]]),
                     len(task["rewards"]), len(backlash) or "无", "<br>".join(policies) or "未发布",
                     "可续训" if bundle else "无"))
    total, twins = len(by_id), sum("base" in task for task in by_id.values())
    return (STYLE + f'<div class="duck"><p>microduck_rl 在提交 <code>{snapshot["revision"][:7]}</code> 注册了 <b>{total}</b> 个任务：'
            f"{total - twins} 个基础任务（{len(FAMILIES)} 个任务族）和 {twins} 个齿隙（Backlash）变体。"
            "齿隙变体给 14 个舵机各加 ±1° 间隙，奖励和观测项不变。</p>"
            + table(("任务族", "学什么", "地形", "机器人模型", "奖励项", "齿隙变体", "官方 ONNX", "训练检查点"), rows, numeric=(4, 5))
            + f"<p><small>“可续训”指 Hugging Face <code>{TRAINING_REPO}</code> 里有完整 rsl_rl 检查点（演员、评论家、归一化、优化器、课程进度）。"
            "ONNX 只能运行，不能继续训练。</small></p></div>")


def family_html(key):
    snapshot = load()
    title, goal, how, flat_id, policies, bundle = FAMILIES[key]
    task = next(item for item in snapshot["tasks"] if item["id"] == flat_id)
    base, backlash = variants(snapshot, flat_id)
    rewards = []
    for term in sorted(task["rewards"], key=lambda item: -abs(item["final"])):
        weight = f'{term["weight"]:g}' + ("" if term["weight"] == term["final"] else f' → {term["final"]:g}')
        rewards.append((code([term["name"]]), REWARD_ZH[term["func"]], weight,
                        f'<small>{html.escape(snapshot["reward_docs"][term["func"]])}</small>'))
    commands = [(code([name]), COMMAND_ZH[cfg["type"]],
                 f'<small>{html.escape(json.dumps(cfg["ranges"]))}</small>' if cfg["ranges"] else "无")
                for name, cfg in task["commands"].items()]
    ppo = task["ppo"]
    bundle_text = f"{TRAINING_REPO}/bundles/{bundle}" if bundle else "未公开"
    if key == "VelStand":
        bundle_text += "（这是 v8 的检查点；当前官方 ONNX 已回退到 v7 权重，v7 检查点未公开）"
    robot = f'{ROBOT_ZH[task["robot_xml"]]} {code([task["robot_xml"]])}'
    if task["other_entities"]:
        robot += f'，场景另有 {code(task["other_entities"])}'
    facts = [
        ("任务 id", "<br>".join(base)),
        ("齿隙变体", "<br>".join(backlash) or "无"),
        ("机器人模型", robot),
        ("演员观测（61 维）", "、".join(OBS_ZH[name] for name in task["actor_obs"])),
        ("仅评论家可见", code(name for name in task["critic_obs"] if name not in task["actor_obs"])),
        ("动作", f'14 个舵机的目标位置偏移，{task["control_hz"]} Hz'),
        ("每局时长", f'{task["episode_s"]:g} s'),
        ("终止条件", code(task["terminations"])),
        ("重置与域随机化", code(task["events"])),
        ("课程", code(task["curriculum"]) or "无"),
        ("PPO", f'网络 {ppo["hidden"]}，每环境每轮 {ppo["steps_per_env"]} 步，最多 {ppo["max_iterations"]} 轮，lr {ppo["learning_rate"]:g}，'
         f'γ {ppo["gamma"]}，λ {ppo["lam"]}，熵系数 {ppo["entropy"]}，镜像损失{"开" if ppo["symmetry"] else "关"}'),
        ("官方 ONNX", "、".join(policies) or "未发布"),
        ("训练检查点", bundle_text),
    ]
    return (STYLE + f'<div class="duck"><h3>{title} · {key}</h3><p>{goal}<br>{how}</p>' + table(("项目", "内容"), facts)
            + f"<h4>奖励（{len(rewards)} 项，{flat_id}）</h4>"
            + table(("奖励项", "含义", "权重", "上游说明原文"), rewards, numeric=(2,))
            + "<p><small>箭头表示课程把权重从初值调到终值。部分惩罚函数自己返回负数，所以权重写成正数。"
            "权重 0 的项用来保留观测槽位或作挂钩。</small></p>"
            + "<h4>指令</h4>" + table(("槽位", "含义", "初始采样范围"), commands)
            + "<p>训练命令（需要 CUDA GPU，本 Notebook 不执行）：<br>"
            + code([f"uv run train {flat_id} --env.scene.num-envs 4096"]) + "</p></div>")


def skill_html(report):
    from .skill_showroom import SHOWROOM_REVISION, SKILLS

    skill = SKILLS[report["skill"]]
    clip = base64.b64encode(Path(report["clip"]).read_bytes()).decode()
    clock, steps = 0.0, []
    for segment in skill.segments:
        caption = html.escape(segment.caption)
        steps.append(f"<li>{clock:.1f}–{clock + segment.seconds:.1f} s　{f'<b>{caption}</b>' if segment.main else caption}　"
                     f'<small>{segment.policy or "无策略"}</small></li>')
        clock += segment.seconds
    measured = table(("实测", "值"), [(html.escape(k), value_text(v)) for k, v in report["metrics"].items()], numeric=(1,))
    note = f"<p>{html.escape(skill.note)}</p>" if skill.note else ""
    return (STYLE + f'<div class="duck"><h3>{skill.title}</h3><p><small>训练任务：{html.escape(skill.task)} · '
            f'权重版本 {SHOWROOM_REVISION[:8]}</small></p><div style="display:flex;gap:20px;flex-wrap:wrap">'
            f'<img width="480" height="360" alt="{skill.title}" src="data:image/webp;base64,{clip}">'
            f'<div><ol style="margin:0;padding-left:1.4em">{"".join(steps)}</ol>{measured}</div></div>{note}</div>')


def show_tasks():
    from IPython.display import HTML, display

    display(HTML(task_table_html()))


def show_family(choice):
    from IPython.display import HTML, display

    display(HTML(family_html(choice.split(" · ")[0])))


def show_skill(choice):
    """Roll out in a subprocess, like the simulator, so notebook state and GL contexts stay separate."""
    from IPython.display import HTML, display
    from .skill_showroom import SKILLS

    key = choice.split(" · ")[0]
    out = ROOT / "runs/showroom"
    for name in SKILLS if key == "all" else [key]:
        handle = display(HTML(f"正在仿真并渲染：{SKILLS[name].title}…"), display_id=True)
        result = subprocess.run(
            [sys.executable, "-m", "microduck_qwen.skill_showroom", "--skill", name, "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(f"{name} 运行失败：\n{result.stderr[-3000:]}")
        handle.update(HTML(skill_html(json.loads((out / f"{name}.json").read_text(encoding="utf-8")))))
