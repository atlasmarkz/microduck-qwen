import json, glob, math, sys, os
d = sys.argv[1]
sc = json.load(open(os.path.join(d, 'scene.json'), encoding='utf-8'))
balls = [(o['x'], o['y'], o['radius'], o['color']) for o in sc['objects'].values()]
def wall_d(x, y):
    best = (9, None)
    for w in sc['walls']:
        ax, ay, bx, by = w['x1'], w['y1'], w['x2'], w['y2']; dx, dy = bx - ax, by - ay; L = dx * dx + dy * dy
        t = 0 if not L else max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / L))
        dist = math.hypot(x - (ax + t * dx), y - (ay + t * dy)) - w['thickness'] / 2
        if dist < best[0]: best = (dist, w['name'])
    return best
for f in sorted(glob.glob(os.path.join(d, 'run', 'skill-*.nav*.jsonl'))):
    L = [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
    if not L: continue
    x, y = L[-1]['position'][:2]
    nb = min((math.hypot(x - bx, y - by) - r, c, round(bx, 2), round(by, 2)) for bx, by, r, c in balls)
    wd = wall_d(x, y)
    print(os.path.basename(f), 'pos', round(x, 2), round(y, 2), 'clear', round(L[-1]['clearance_m'], 3), 'ball surf', round(nb[0], 2), nb[1:], 'wall surf', round(wd[0], 2), wd[1])
