"""Large multi-room worlds. Ground truth is for physics, safety planning and evaluation only.

A world is a rows x cols grid of square rooms joined by doorways. The start room is
moved to the origin, so the robot's home pose stays (0, 0) facing +x and the start
room keeps the familiar three-ball row ahead of the robot: every existing skill and
task still applies. Walls, other rooms and distractors add perceptual clutter and
planner load; they do not add skills.
"""

import math
import random

ROOM_SIZE = 3.6
DOOR_WIDTH = 1.1
WALL_THICKNESS = 0.10
WALL_HEIGHT = 0.40
WALL_RGBA = [0.55, 0.55, 0.60, 1]
PILLAR_RGBA = [0.45, 0.45, 0.50, 1]
BALL_COLORS = {
    "red": [0.9, 0.06, 0.04, 1],
    "blue": [0.04, 0.16, 0.95, 1],
    "green": [0.04, 0.8, 0.12, 1],
    "yellow": [0.95, 0.85, 0.05, 1],
}
# Curriculum levels: map size and clutter grow; the skill set does not change.
WORLD_LEVELS = [
    {"rooms": (1, 1), "density": 0.0},
    {"rooms": (1, 2), "density": 0.5},
    {"rooms": (2, 2), "density": 1.0},
    {"rooms": (3, 3), "density": 1.5},
]


def make_world(seed=42, rooms=(2, 2), density=1.0):
    rows, cols = rooms
    if not (1 <= rows <= 5 and 1 <= cols <= 5) or not 0 <= density <= 4:
        raise ValueError("world must be 1..5 x 1..5 rooms with density 0..4")
    rng = random.Random(seed)
    size = ROOM_SIZE
    start = (rng.randrange(rows), rng.randrange(cols))
    x0, y0 = -(start[1] + 0.5) * size, -(start[0] + 0.5) * size
    xmin, xmax, ymin, ymax = x0, x0 + cols * size, y0, y0 + rows * size
    center = lambda r, c: (x0 + (c + 0.5) * size, y0 + (r + 0.5) * size)

    # A random spanning tree of doorways makes every room reachable; extra doorways
    # (loops) appear with some probability.
    edges = [((r, c), (r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    edges += [((r, c), (r + 1, c)) for r in range(rows - 1) for c in range(cols)]
    rng.shuffle(edges)
    parent = {(r, c): (r, c) for r in range(rows) for c in range(cols)}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    doored = set()
    for a, b in edges:
        if find(a) != find(b):
            parent[find(a)] = find(b)
            doored.add((a, b))
        elif rng.random() < 0.35:
            doored.add((a, b))
    # The start room's +x neighbour is always open, so a ball in the next room can be
    # seen and approached through the doorway.
    forward = (start, (start[0], start[1] + 1))
    if cols > start[1] + 1:
        doored.add(forward)

    walls, doors = [], []

    def add_wall(x1, y1, x2, y2, planning=True):
        walls.append({"name": f"wall_{len(walls)}", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "thickness": WALL_THICKNESS, "height": WALL_HEIGHT, "rgba": WALL_RGBA,
                      "planning": planning})

    # The outer shell is visible and solid but not planned around: bounds already hold the robot in.
    for line in ((xmin, ymin, xmax, ymin), (xmin, ymax, xmax, ymax), (xmin, ymin, xmin, ymax), (xmax, ymin, xmax, ymax)):
        add_wall(*line, planning=False)
    for a, b in edges:
        vertical = a[0] == b[0]  # side-by-side rooms are split by a vertical wall
        if vertical:
            line_at, lo, hi = x0 + (a[1] + 1) * size, y0 + a[0] * size, y0 + (a[0] + 1) * size
        else:
            line_at, lo, hi = y0 + (a[0] + 1) * size, x0 + a[1] * size, x0 + (a[1] + 1) * size
        pieces = [(lo, hi)]
        if (a, b) in doored:
            if (a, b) == forward:
                centre = (lo + hi) / 2 + rng.uniform(-0.4, 0.4)
            else:
                centre = rng.uniform(lo + DOOR_WIDTH / 2 + 0.5, hi - DOOR_WIDTH / 2 - 0.5)
            pieces = [(lo, centre - DOOR_WIDTH / 2), (centre + DOOR_WIDTH / 2, hi)]
            doors.append({"x": line_at if vertical else centre, "y": centre if vertical else line_at,
                          "axis": "x" if vertical else "y", "width": DOOR_WIDTH,
                          "rooms": [list(a), list(b)], "forward": (a, b) == forward})
        for p, q in pieces:
            add_wall(*((line_at, p, line_at, q) if vertical else (p, line_at, q, line_at)))

    objects, obstacles = {}, []
    placed = [(0.0, 0.0, 0.45)]  # the robot start stays clear

    def room_of(x, y):
        return [min(rows - 1, max(0, int((y - y0) // size))), min(cols - 1, max(0, int((x - x0) // size)))]

    def add_ball(x, y, color):
        objects[f"ball_{len(objects)}"] = {
            "label": color + " ball", "color": color, "x": x, "y": y, "radius": 0.07, "collidable": True,
            "shape": "sphere", "rgba": BALL_COLORS[color], "path_margin": 0.035, "room": room_of(x, y)}
        placed.append((x, y, 0.07))

    def free_spot(r, c, radius, margin=0.6, tries=300):
        cx, cy = center(r, c)
        half = size / 2 - margin
        for _ in range(tries):
            x, y = rng.uniform(cx - half, cx + half), rng.uniform(cy - half, cy + half)
            if any(math.hypot(x - px, y - py) < radius + pr + 0.45 for px, py, pr in placed):
                continue
            if any(math.hypot(x - d["x"], y - d["y"]) < 0.95 for d in doors):
                continue
            return x, y
        return None

    # Start room: the usual three-ball row ahead of the robot (image-left is +y).
    names = list(BALL_COLORS)
    rng.shuffle(names)
    for index, y in enumerate((0.36, 0.0, -0.36)):
        add_ball(rng.uniform(0.9, 1.2), y + rng.uniform(-0.055, 0.055), names[index])
    # One ball in the next room, roughly along the view axis through the doorway.
    gate = next((d for d in doors if d["forward"]), None)
    if gate:
        add_ball(gate["x"] + rng.uniform(0.9, 1.3), max(-0.45, min(0.45, gate["y"] * 0.5 + rng.uniform(-0.15, 0.15))),
                 rng.choice(names[:3]))
    # The colour missing from the start row exists in exactly one other room, so "find the X
    # ball" is a real search task that needs doorways, not a nearby ball.
    others = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != start]
    if others:
        r, c = rng.choice(others)
        spot = free_spot(r, c, 0.07)
        if spot:
            add_ball(*spot, names[3])
    # Everywhere else: scattered balls and gray pillars, scaled by density.
    for r in range(rows):
        for c in range(cols):
            for _ in range(round(2 * density) + (1 if density and (r, c) != start else 0)):
                spot = free_spot(r, c, 0.07)
                if spot:
                    add_ball(*spot, rng.choice(names[:3]))
            for _ in range(round(1.5 * density)):
                spot = free_spot(r, c, 0.10)
                # Keep the start room's frontal area free for the three-ball row.
                if spot and not ((r, c) == start and math.hypot(*spot) < 1.6):
                    obstacles.append({"name": f"pillar_{len(obstacles)}", "x": spot[0], "y": spot[1],
                                      "radius": 0.10, "height": 0.30, "rgba": PILLAR_RGBA})
                    placed.append((spot[0], spot[1], 0.10))
    return {"bounds": [xmin, xmax, ymin, ymax], "objects": objects, "obstacles": obstacles,
            "walls": walls, "doors": doors, "rooms": [rows, cols], "start_room": list(start),
            "room_size": size, "grid": 0.10, "density": density, "seed": seed,
            "occlusion": "none", "clutter": False, "world": True,
            "evaluation_only_initial_left_target": "ball_0"}
