"""Draw the depth-built map, frontiers and true walls/balls of one evaluated task.

    python tools/render_eval_map.py runs/lang/eff-98100/seed-98101/task-200 runs/lang/eff-98100/seed-98101/simulator-0.scene.json out.png
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

from microduck_qwen.mapping import OccupancyMap, CELL, UNKNOWN, FREE
task = Path(sys.argv[1]); scene_file = Path(sys.argv[2]); out = Path(sys.argv[3])
grid = np.load(sorted(task.glob('skill-*.map.npy'))[-1]); scene = json.loads(scene_file.read_text())
m = OccupancyMap(); m.grid = grid
n = grid.shape[0]; half = n * CELL / 2
S = 70; lo, hi = -6.0, 6.0
W = int((hi - lo) * S)
img = np.full((W, W, 3), 40, np.uint8)
px = lambda x, y: (int((x - lo) * S), int((hi - y) * S))
xs = np.arange(n) * CELL - half + CELL / 2
for ix, iy in np.argwhere(grid != UNKNOWN):
    x, y = xs[ix], xs[iy]
    if lo < x < hi and lo < y < hi:
        cv2.circle(img, px(x, y), 2, (200, 200, 200) if grid[ix, iy] == FREE else (60, 60, 230), -1)
for cells in m.frontier_cells():
    for ix, iy in cells:
        cv2.circle(img, px(xs[ix], xs[iy]), 3, (0, 220, 255), -1)
for w in scene.get('walls', []):
    cv2.line(img, px(w['x1'], w['y1']), px(w['x2'], w['y2']), (0, 170, 0), 2)
for o in scene['objects'].values():
    col = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}[o['color']]
    cv2.circle(img, px(o['x'], o['y']), 7, col, 2)
cv2.circle(img, px(0, 0), 10, (255, 255, 255), 2)
cv2.imwrite(str(out), img)
print(out, 'doors', [(round(d.get('x', 0), 2), round(d.get('y', 0), 2)) for d in scene.get('doors', [])][:8])
