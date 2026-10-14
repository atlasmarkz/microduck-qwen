"""Seeded visual scenes. Ground truth is for physics, safety planning and evaluation only."""

import random


def make_scene(seed=42, occlusion="none"):
    if occlusion not in ("none", "partial", "full", "absent"):
        raise ValueError("unknown occlusion scenario")
    rng = random.Random(seed)
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
        "evaluation_only_initial_left_target": "ball_0"
        if "ball_0" in objects
        else None,
    }
