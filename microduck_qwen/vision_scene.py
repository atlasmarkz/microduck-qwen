"""Seeded visual scenes. Ground truth is for physics, safety planning and evaluation only."""

import random


def make_scene(seed=42, occlusion="none", clutter=False, yellow_course=False):
    if occlusion not in ("none", "partial", "full", "absent"):
        raise ValueError("unknown occlusion scenario")
    rng = random.Random(seed)
    if yellow_course:
        if occlusion != "none" or clutter:
            raise ValueError("yellow course cannot be mixed with ball occlusion/clutter profiles")
        return {"bounds": [-0.7, 3.4, -1.4, 1.4], "objects": {},
                "obstacles": [{"name": f"yellow_{i}", "x": 0.75 + i * 0.95 + rng.uniform(-0.04, 0.04),
                    "y": (-1 if i % 2 else 1) * rng.uniform(0.21, 0.29), "radius": 0.10,
                    "height": 0.30, "rgba": [0.9, 0.65, 0.06, 1]} for i in range(2 + seed % 2)],
                "seed": seed, "occlusion": "none", "clutter": False, "yellow_course": True,
                "evaluation_only_initial_left_target": None}
    colors = [
        ("red", [0.9, 0.06, 0.04, 1]),
        ("blue", [0.04, 0.16, 0.95, 1]),
        ("green", [0.04, 0.8, 0.12, 1]),
    ]
    rng.shuffle(colors)
    objects = {}
    # Image-left is +y when facing +x. Jitter changes both range and azimuth.
    for index, y in enumerate((0.36, 0.0, -0.36)):
        color, rgba = colors[index]
        if occlusion == "absent" and index == 0:
            continue
        objects[f"ball_{index}"] = {
            "label": color + " ball",
            "color": color,
            "x": rng.uniform(0.9, 1.2),
            "y": y + rng.uniform(-0.055, 0.055),
            "radius": 0.07,
            "collidable": True,
            "shape": "sphere",
            "rgba": rgba,
            "path_margin": 0.035,
        }
    obstacles = [
        {
            "name": "obstacle_1",
            "x": rng.uniform(0.5, 0.65),
            "y": rng.choice([-1, 1]) * rng.uniform(0.65, 0.85),
            "radius": 0.11,
            "height": 0.3,
            "rgba": [0.9, 0.65, 0.06, 1],
        }
    ]
    if clutter:
        # Duplicate two ball colors and widen the row so identity can no longer be
        # inferred from color alone. Extra pillars sit behind the ball row: they are
        # guaranteed clear of every ball and of every 0.30 m approach stopping point,
        # and far enough from obstacle_1's ±0.65..0.85 bay not to overlap it.
        for index, y in enumerate((0.62, -0.62)):
            color, rgba = colors[index]
            objects[f"ball_{index + 3}"] = {
                "label": color + " ball",
                "color": color,
                "x": rng.uniform(0.9, 1.2),
                "y": y + rng.uniform(-0.03, 0.03),
                "radius": 0.07,
                "collidable": True,
                "shape": "sphere",
                "rgba": rgba,
                "path_margin": 0.035,
            }
        for index, y in enumerate((0.52, -0.52)):
            obstacles.append(
                {
                    "name": f"pillar_{index}",
                    "x": rng.uniform(1.45, 1.60),
                    "y": y + rng.uniform(-0.17, 0.17),
                    "radius": 0.10,
                    "height": 0.3,
                    "rgba": [0.9, 0.65, 0.06, 1],
                }
            )
    if occlusion in ("partial", "full"):
        target = objects["ball_0"]
        x = target["x"] * 0.60
        y = target["y"] * 0.60
        if occlusion == "partial":
            y += 0.05
        obstacles.append(
            {
                "name": "occluder",
                "x": x,
                "y": y,
                "radius": 0.065 if occlusion == "partial" else 0.11,
                "height": 0.35,
                "rgba": [0.9, 0.65, 0.06, 1],
            }
        )
    return {
        "bounds": [-0.6, 1.8, -1.2, 1.2],
        "objects": objects,
        "obstacles": obstacles,
        "seed": seed,
        "occlusion": occlusion,
        "clutter": clutter,
        "evaluation_only_initial_left_target": "ball_0"
        if "ball_0" in objects
        else None,
    }
