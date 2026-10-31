"""Compositional instruction grammar for colour-targeted approach tasks.

Teacher-side only: it turns (colour, template, tail) into an instruction and the
expected skill sequence. Train and held-out splits use DISJOINT templates, colour
words and tails, so a held-out sentence is a new surface form, not a memorised one.
Optionally one colour is withheld from train targets (``holdout_color``) so a
validation/test pass measures colour composition, not recall.
"""

import random

COLORS = ("red", "blue", "green", "yellow")
COLOR_WORDS = {
    "train": {"red": ["红色"], "blue": ["蓝色"], "green": ["绿色"], "yellow": ["黄色"]},
    "valid": {"red": ["红色", "红"], "blue": ["蓝色", "蓝"], "green": ["绿色", "绿"], "yellow": ["黄色", "黄"]},
    "test": {"red": ["红", "红色的"], "blue": ["蓝", "蓝色的"], "green": ["绿", "绿色的"], "yellow": ["黄", "黄色的"]},
}
# "{c}" is the colour word; every template names the object as a ball.
TEMPLATES = {
    "train": ["走到{c}球旁边", "靠近{c}球", "去{c}球那里", "朝{c}球走过去"],
    "valid": ["请接近那个{c}球", "移动到{c}球附近", "往{c}球的方向走"],
    "test": ["走向{c}球", "到{c}球边上去", "帮我靠近{c}球"],
}
# tail kind -> phrases; "none" ends after the approach (executor then expects done).
TAILS = {
    "train": {"none": [""], "home": ["，然后回到初始位置", "，再回到起点"], "stop": ["，接着停下来"]},
    "valid": {"none": [""], "home": ["，之后返回出发点"], "stop": ["，随后停下"]},
    "test": {"none": [""], "home": ["，完成后回到最初位置"], "stop": ["，到了就别动"]},
}


def approach_spec(color, ident=None):
    """Teacher placeholder resolved to a concrete verified action at collection time."""
    return {"teacher_color": color}


def phrasings(split, color, rng):
    """All (instruction, tail_kind) surface forms for a colour in a split."""
    words, templates, tails = COLOR_WORDS[split][color], TEMPLATES[split], TAILS[split]
    return [(t.format(c=w) + tail, kind) for t in templates for w in words
            for kind, phrases in tails.items() for tail in phrases]


def colour_tasks(split, scene_colors, rng, per_color=1, holdout_color=None):
    """One or more (instruction, specs, meta) per colour present in the scene."""
    tasks = []
    for color in sorted(set(scene_colors)):
        if split == "train" and color == holdout_color:
            continue
        if split != "train" and holdout_color and color != holdout_color:
            # Held-out evaluation focuses on the withheld colour once one is set.
            continue
        forms = phrasings(split, color, rng)
        for text, tail in rng.sample(forms, min(per_color, len(forms))):
            specs = [approach_spec(color)] + ([{"action": "return_home"}] if tail == "home"
                                               else [{"action": "stop"}] if tail == "stop" else [])
            tasks.append((text, specs, {"color": color, "tail": tail, "kind": "colour_approach"}))
    return tasks


# "Find it somewhere else" tasks: the colour is not in view, so the first skill is explore.
EXPLORE_TEMPLATES = {
    "train": ["这里看不到{c}球，去别的房间找到它并靠近", "去找一个{c}球，找到后走到它旁边", "到其他地方找{c}球，然后靠近它"],
    "valid": ["在别处找到{c}球并靠近它", "出去寻找{c}球，找到就走过去"],
    "test": ["帮我找一个{c}球，找到后靠过去", "去外面找{c}球然后接近"],
}


def explore_phrasings(split, color):
    words, tails = COLOR_WORDS[split][color], TAILS[split]
    return [(t.format(c=w) + tail, kind) for t in EXPLORE_TEMPLATES[split] for w in words
            for kind, phrases in tails.items() for tail in phrases]


def explore_tasks(split, hidden_colors, rng, per_color=1):
    """(instruction, specs, meta) to find and approach each colour that is NOT in the first view."""
    tasks = []
    for color in sorted(set(hidden_colors)):
        forms = explore_phrasings(split, color)
        for text, tail in rng.sample(forms, min(per_color, len(forms))):
            specs = [{"action": "explore", "color": color}, {"teacher_nearest": color}] + (
                [{"action": "return_home"}] if tail == "home" else [{"action": "stop"}] if tail == "stop" else [])
            tasks.append((text, specs, {"color": color, "tail": tail, "kind": "explore_find"}))
    return tasks


def explore_paraphrases(color, tail, text, count=3, rng=None):
    rng = rng or random.Random(text)
    pool = [t for t, k in explore_phrasings("train", color) if k == tail and t != text]
    return rng.sample(pool, min(count, len(pool)))


def paraphrases(color, tail, text, count=3, rng=None):
    """Extra TRAIN-pool surface forms for the same verified trajectory (not new physics)."""
    rng = rng or random.Random(text)
    pool = [t for t, k in phrasings("train", color, rng) if k == tail and t != text]
    return rng.sample(pool, min(count, len(pool)))
