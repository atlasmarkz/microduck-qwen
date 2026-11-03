import json, sys
from pathlib import Path
import cv2, numpy as np
from microduck_qwen.mapping import OccupancyMap, CELL, UNKNOWN, FREE, OCCUPIED
out = Path(sys.argv[1]); mp = sorted((out / "run").glob("skill-*.map.npy"))[-1]
grid = np.load(mp); scene = json.loads((out / "scene.json").read_text())
m = OccupancyMap(); m.grid = grid
n = grid.shape[0]; half = n * CELL / 2
S = 60  # px per metre
lo, hi = -6.5, 6.5
W = int((hi - lo) * S)
img = np.full((W, W, 3), 40, np.uint8)
def px(x, y): return int((x - lo) * S), int((hi - y) * S)
xs = np.arange(n) * CELL - half + CELL / 2
for ix, iy in np.argwhere(grid != UNKNOWN):
    x, y = xs[ix], xs[iy]
    if lo < x < hi and lo < y < hi:
        c = (200, 200, 200) if grid[ix, iy] == FREE else (60, 60, 230)
        cv2.circle(img, px(x, y), 2, c, -1)
for ix_cells in m.frontier_cells():
    for ix, iy in ix_cells:
        cv2.circle(img, px(xs[ix], xs[iy]), 3, (0, 220, 255), -1)
for w in scene.get("walls", []):
    cv2.line(img, px(w["x1"], w["y1"]), px(w["x2"], w["y2"]), (0, 160, 0), 1)
for o in scene["objects"].values():
    col = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0), "yellow": (0, 255, 255)}[o["color"]]
    cv2.circle(img, px(o["x"], o["y"]), 6, col, 2)
cv2.circle(img, px(0, 0), 8, (255, 255, 255), 1)
cv2.imwrite(str(out / "map.png"), img)
print(out / "map.png", "free", int((grid == FREE).sum()), "occ", int((grid == OCCUPIED).sum()), "frontier clusters", len(m.frontier_cells()))
