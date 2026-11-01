"""Open phrase bank: many natural ways to say the same task, split into disjoint train/valid/test components.

The first grammar had a handful of templates, so the actor could latch onto a keyword ("找") instead of
the meaning. Here every slot (opening, verb phrase, colour word, noun, room word, closing) has its own pool;
each pool item is assigned to exactly one split, so a held-out sentence is built from components the actor
never saw in training, not only from a new combination of them. A few English forms are mixed in.

A room word ("另一个房间") is the cue that the ball is out of view, so such sentences belong to explore tasks;
plain approach sentences never contain one. Phrases never name behaviours the robot cannot do.
"""

import random

SPLITS = ("train", "valid", "test")


def split_pool(items):
    """Round-robin 5:1:1 so every split with >= 7 items has at least one; the order of `items` is the contract."""
    return {"train": [x for i, x in enumerate(items) if i % 7 < 5],
            "valid": [x for i, x in enumerate(items) if i % 7 == 5],
            "test": [x for i, x in enumerate(items) if i % 7 == 6]}


OPENINGS = split_pool(["请", "麻烦你", "帮我", "现在", "机器人，请", "拜托你", "请你", "快点", "先", "你可以", "我想让你", "接下来"])
# {b} is the ball phrase (colour + noun).
APPROACH = split_pool([
    "走到{b}旁边", "靠近{b}", "去{b}那里", "朝{b}走过去", "走向{b}", "到{b}边上去", "接近{b}", "往{b}那边走",
    "移动到{b}附近", "去到{b}身边", "走过去，停在{b}旁边", "到{b}那儿", "过去{b}那边", "向{b}靠拢", "来到{b}跟前",
    "朝着{b}前进，停在它旁边", "去{b}所在的位置", "走近{b}"])
# {r} is a room word, {b} the ball phrase. These name no search verb on purpose, or only some of them.
EXPLORE = split_pool([
    "走到{r}的{b}旁边", "去{r}的{b}那里", "到{r}，靠近{b}", "去{r}找{b}，走到它旁边", "这里没有{b}，去{r}看看",
    "{r}里有{b}，过去", "去{r}，走到{b}旁边", "找到{r}的{b}并靠近它", "出去，到{r}靠近{b}", "{r}那个{b}，走到它旁边",
    "去{r}寻找{b}，找到就走过去", "带我去{r}的{b}那边", "去{r}，找到{b}后走近它", "先出去，再接近{r}的{b}",
    "{b}在{r}，走到它旁边去", "从这里出发，到{r}接近{b}"])
ROOMS = split_pool(["另一个房间", "别的房间", "其他房间", "隔壁房间", "另一间屋子", "旁边房间", "对面房间",
                    "别处", "其他地方", "另外那个房间", "后面房间", "那边房间"])
COLOUR_FORMS = split_pool(["{c}色", "{c}", "{c}色的", "{c}的", "纯{c}色的", "那个{c}色的", "那颗{c}色的", "{c}颜色的",
                           "{c}色那个", "一个{c}色的", "{c}色的那个", "全{c}的"])
NOUNS = split_pool(["球", "小球", "圆球", "小圆球", "球体", "球儿", "球球", "皮球", "彩球"])
TAIL_HOME = split_pool(["，然后回到起点", "，再回到初始位置", "，之后返回出发点", "，最后回原位", "，完事后回到原来的地方",
                        "，再回去", "，接着回到最初位置", "，到了以后回起点", "，然后返回开始的位置", "，再回到刚才出发的地方",
                        "，之后回家", "，走完回到起点"])
TAIL_STOP = split_pool(["，接着停下来", "，随后停下", "，到了就别动", "，然后原地待命", "，再停住", "，到了停下", "，然后别再动了",
                        "，靠近后停止", "，之后保持不动"])
PUNCT = ("", "。", "吧", "哦", "！")

COLOUR_CHAR = {"red": "红", "blue": "蓝", "green": "绿", "yellow": "黄"}
COLOUR_EN = {"red": "red", "blue": "blue", "green": "green", "yellow": "yellow"}
EN_APPROACH = split_pool(["go to the {c} ball", "walk up to the {c} ball", "move next to the {c} ball", "approach the {c} ball",
                          "get close to the {c} ball", "head over to the {c} ball", "go stand beside the {c} ball",
                          "come up to the {c} ball"])
EN_EXPLORE = split_pool(["go to the {c} ball in another room", "find a {c} ball in a different room and walk up to it",
                         "there is no {c} ball here, look for it in another room and go next to it",
                         "walk to the {c} ball in the other room", "search the other rooms for a {c} ball and approach it",
                         "the {c} ball is in another room, go next to it", "head to the {c} ball that is in a different room",
                         "leave this room and get close to the {c} ball"])
EN_HOME = split_pool([", then return to the start", ", and then come back to where you started", ", then go back to the starting point",
                      ", and after that return home", ", then head back to the start position", ", then return to your initial position",
                      ", and finally go back to the beginning"])
EN_STOP = split_pool([", then stop", ", and then stand still", ", then stay there", ", and then stop moving", ", then halt",
                      ", and hold still after that", ", then wait there"])
EN_SHARE = 0.15

ROOM_CUES = tuple(sorted({r for pool in ROOMS.values() for r in pool}, key=len, reverse=True))


def _chinese(split, color, kind, rng):
    form = rng.choice(COLOUR_FORMS[split]).format(c=COLOUR_CHAR[color])
    ball = form + rng.choice(NOUNS[split])
    if kind == "explore":
        body = rng.choice(EXPLORE[split]).format(r=rng.choice(ROOMS[split]), b=ball)
    else:
        body = rng.choice(APPROACH[split]).format(b=ball)
    tail_kind = rng.choice(("none", "home", "stop"))
    tail = {"none": "", "home": rng.choice(TAIL_HOME[split]), "stop": rng.choice(TAIL_STOP[split])}[tail_kind]
    opening = rng.choice([""] + OPENINGS[split])
    return opening + body + tail + rng.choice(PUNCT), tail_kind


def _english(split, color, kind, rng):
    body = rng.choice((EN_EXPLORE if kind == "explore" else EN_APPROACH)[split]).format(c=COLOUR_EN[color])
    tail_kind = rng.choice(("none", "home", "stop"))
    tail = {"none": "", "home": rng.choice(EN_HOME[split]), "stop": rng.choice(EN_STOP[split])}[tail_kind]
    return rng.choice(("", "please ", "now ")) + body + tail, tail_kind


def bank(split, color, kind, count=400):
    """Deterministic list of (instruction, tail_kind) for `kind` in ("approach", "explore")."""
    rng = random.Random(f"bank-{split}-{color}-{kind}")
    seen, out = set(), []
    for _ in range(count * 4):
        make = _english if rng.random() < EN_SHARE else _chinese
        text, tail = make(split, color, kind, rng)
        if text not in seen:
            seen.add(text)
            out.append((text, tail))
        if len(out) == count:
            break
    return out
