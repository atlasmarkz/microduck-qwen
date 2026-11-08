"""Can the robot physically leave the start room without touching a ball or a door frame? (model-independent check)

For each evaluated layout, rasterise the TRUE scene (walls with thickness, pillars, balls) at 2 cm, keep cells whose distance to
every obstacle surface is at least `--half-width` (the robot's measured half width 0.10 m plus sway), and flood-fill from the start.
A target ball whose surroundings are not reached is blocked: no contact-free route exists for any policy, so a failure there says
nothing about the model. The evaluation scores a contact with the scene as a failure, which is why a squeeze is not an option.

    python tools/feasibility.py runs/lang/cand-v2-99000 [--half-width 0.12]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CELL = 0.02


def reachable_balls(scene, half_width):
    x0, x1, y0, y1 = scene["bounds"]
    nx, ny = int(round((x1 - x0) / CELL)) + 1, int(round((y1 - y0) / CELL)) + 1
    solid = np.zeros((ny, nx), np.uint8)

    def px(x, y):
        return int(round((x - x0) / CELL)), int(round((y - y0) / CELL))

    for wall in scene["walls"]:
        cv2.line(solid, px(wall["x1"], wall["y1"]), px(wall["x2"], wall["y2"]), 1, max(1, int(round(wall["thickness"] / CELL))))
    for obstacle in scene["obstacles"]:
        cv2.circle(solid, px(obstacle["x"], obstacle["y"]), int(round(obstacle["radius"] / CELL)), 1, -1)
    for ball in scene["objects"].values():
        cv2.circle(solid, px(ball["x"], ball["y"]), int(round(ball["radius"] / CELL)), 1, -1)
    free = cv2.distanceTransform((1 - solid).astype(np.uint8), cv2.DIST_L2, 5) * CELL >= half_width
    _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    start = labels[px(0.0, 0.0)[1], px(0.0, 0.0)[0]]
    out = {}
    for name, ball in scene["objects"].items():
        # the robot only has to get within sight and 0.3 m of a ball: any free cell within 0.5 m of it in the start's region
        cx, cy = px(ball["x"], ball["y"])
        r = int(round(0.5 / CELL))
        window = labels[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        out[name] = bool(start != 0 and (window == start).any())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--half-width", type=float, default=0.12)
    args = parser.parse_args()
    for run in args.runs:
        for scene_file in sorted(run.glob("seed-*/simulator-0.scene.json")):
            scene = json.loads(scene_file.read_text(encoding="utf-8"))
            reach = reachable_balls(scene, args.half_width)
            colors = {}
            for name, ball in scene["objects"].items():
                colors.setdefault(ball["color"], []).append(reach[name])
            blocked = sorted(c for c, flags in colors.items() if not any(flags))
            print(f"{scene_file.parent.name}: balls reachable {sum(reach.values())}/{len(reach)}; colours with no reachable ball: {blocked or '-'}")


if __name__ == "__main__":
    main()
