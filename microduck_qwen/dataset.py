"""Reproducible synthetic teacher data with disjoint instruction template families."""

import argparse
import hashlib
import itertools
import json
import random

from . import ROOT
from .contract import compact, messages, validate

TEMPLATES = {
    "train": ["走到{a}旁边", "靠近{a}", "去{a}那边", "移动到{a}附近", "导航到{a}", "绕开障碍物走到{a}", "请前往{a}", "去找{a}"],
    "valid": ["把自己移动至{a}周围", "避开柱子后接近{a}"],
    "test": ["以{a}为目的地移动", "我要你抵达{a}附近", "朝{a}行进，路上的障碍请绕过去"],
}
ALIASES = {"red_ball": ["红球", "球", "小球"], "blue_marker": ["蓝色标记", "蓝点"]}
SPEEDS = {"slow": "慢慢地", "normal": "", "fast": "快速地"}
STOPS = {
    "train": ["停下", "停止移动", "不要继续走", "站着别动", "立即停下", "请停止导航", "停在原地", "暂停前进", "中止任务", "取消行走", "别再继续移动", "不要走了"],
    "valid": ["别走了", "马上停止行进"],
    "test": ["取消当前移动", "在原地停住", "终止导航"],
}
UNSUPPORTED = {
    "train": ["跳个舞", "踢球", "飞到天上", "抓起红球", "去绿球旁边", "走到坐标", "把球叼起来", "去桌子旁", "走到坐标(100,100)"],
    "valid": ["跳起来", "请找紫球", "向我汇报天气"],
    "test": ["请拾起球", "翻个跟头", "找黄色小鸭", "把蓝色物体拿过来", "走到坐标(50,50)"],
}


def named_goal(target, speed="normal", radius=0.3):
    return {"target": target, "speed": speed, "stop_distance_m": radius}


def build(split, seed):
    rng = random.Random(seed)
    rows = []

    def add(instruction, goals=None, status="ok", category="object"):
        plan = validate({"status": status, "goals": goals or []})
        rows.append({"instruction": instruction, "expected": plan, "category": category, "split": split})

    singles = []
    for template, target, speed, radius, units in itertools.product(
        TEMPLATES[split], ALIASES, SPEEDS, (0.25, 0.3, 0.4, 0.5), ("m", "cm")
    ):
        alias = rng.choice(ALIASES[target])
        text = SPEEDS[speed] + template.format(a=alias)
        if radius != 0.3 or units == "cm":
            distance = f"{int(radius * 100)}厘米" if units == "cm" else f"{radius}米"
            text += f"，与目标保持{distance}距离后停下"
        goal = named_goal(target, speed, radius)
        add(text, [goal])
        singles.append((text, goal))
    for _ in range(480 if split == "train" else 20):
        first, second = rng.sample(singles, 2)
        if split == "train":
            text = f"先{first[0]}，再{second[0]}"
        elif split == "valid":
            text = f"首先{first[0]}，随后{second[0]}"
        else:
            text = f"完成{first[0]}后，接着{second[0]}"
        add(text, [first[1], second[1]], category="sequence")
    point_templates = {
        "train": ["移动到世界坐标({x},{y})", "到坐标 x={x}，y={y} 的位置", "请去坐标（{x}，{y}）"],
        "valid": ["终点设置为坐标({x}, {y})，请前往"],
        "test": ["请将世界坐标({x}, {y})作为目的地", "我希望你抵达 x={x}、y={y} 这个点"],
    }
    coords = (0.2, 0.5, 0.8, 1.2) if split == "train" else (0.35, 0.7, 1.05)
    for template, x, y, speed in itertools.product(point_templates[split], coords, (-0.5, 0.0, 0.5), SPEEDS):
        goal = {"target": "point", "speed": speed, "stop_distance_m": 0.15, "x": x, "y": y}
        add(SPEEDS[speed] + template.format(x=x, y=y), [goal], category="point")
    if split == "train":
        for _ in range(500):
            x = rng.choice((0.15, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.15, 1.25, 1.35))
            y = rng.choice((-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8))
            speed = rng.choice(list(SPEEDS))
            template = rng.choice(point_templates[split])
            goal = {"target":"point","speed":speed,"stop_distance_m":0.15,"x":x,"y":y}
            add(SPEEDS[speed]+template.format(x=x,y=y),[goal],category="point")
    for text in STOPS[split]:
        add(text, status="stop", category="stop")
    for text in UNSUPPORTED[split]:
        add(text, status="unsupported", category="unsupported")
    rows = list({r["instruction"]: r for r in rows}.values())
    # Keep the small stop/unsupported classes visible during training, not just the navigation majority.
    if split == "train":
        negatives = [r for r in rows if r["expected"]["status"] != "ok"]
        rows += negatives * 8
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    seen = set()
    stats = {}
    for i, split in enumerate(("train", "valid", "test")):
        rows = build(split, args.seed + i)
        inputs = {r["instruction"] for r in rows}
        overlap = seen & inputs
        if overlap:
            raise ValueError(f"数据集跨 split 重复: {overlap}")
        seen |= inputs
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (out / f"{split}.jsonl").write_text(text, encoding="utf-8", newline="\n")
        chat = [{"messages": messages(r["instruction"]) + [{"role": "assistant", "content": compact(r["expected"])}]} for r in rows]
        (out / f"{split}_chat.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chat), encoding="utf-8", newline="\n")
        stats[split] = {"rows": len(rows), "unique_instructions": len(inputs), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    stats["method"] = "Synthetic grammar teacher; disjoint paraphrase templates, not human-validated language data."
    (out / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
