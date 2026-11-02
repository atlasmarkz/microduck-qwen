"""Incremental 2D occupancy map built from RGB-D frames and the robot's own pose.

Perception only: free space is carved along depth rays and obstacles are marked where a
ray hits something above the floor. Pose and camera extrinsics are still the simulator's
(explicitly oracle localization); the scene's ground-truth obstacle list is never read.
Unknown space is never treated as free.
"""

import math

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
CELL = 0.10
FLOOR_HEIGHT = 0.03   # a hit below this is floor: free space, not an obstacle
MAX_HEIGHT = 0.80     # hits above this are ignored (ceiling-like returns)
# Matches the planner used by navigation.execute on the perceived map: robot radius 0.14,
# path margin 0.12 and the 5 cm cell radius, so waypoints are never inside its blocked zone.
INFLATE_M = 0.32


class OccupancyMap:
    def __init__(self, size_m=30.0, cell=CELL):
        self.cell = cell
        self.n = int(round(size_m / cell))
        self.origin = -size_m / 2
        self.grid = np.zeros((self.n, self.n), dtype=np.uint8)

    def index(self, x, y):
        return (np.floor((np.asarray(x) - self.origin) / self.cell).astype(int),
                np.floor((np.asarray(y) - self.origin) / self.cell).astype(int))

    def center(self, ix, iy):
        return (self.origin + (ix + 0.5) * self.cell, self.origin + (iy + 0.5) * self.cell)

    def _inside(self, ix, iy):
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)

    def _carve(self, camera, ends, stop_short=0.0, max_range=4.0):
        """Mark UNKNOWN cells FREE along the segments camera -> ends (never overwrites obstacles)."""
        if not len(ends):
            return
        steps = int(max_range / (self.cell / 2)) + 1
        along = np.linspace(0.0, 1.0, steps, endpoint=False)[None, :, None]
        vectors = ends - camera
        length = np.hypot(vectors[:, 0], vectors[:, 1])
        scale = np.where(length > 0, np.maximum(length - stop_short, 0) / np.maximum(length, 1e-9), 0.0)
        samples = (camera + vectors[:, None, :] * scale[:, None, None] * along).reshape(-1, 2)
        ix, iy = self.index(samples[:, 0], samples[:, 1])
        ok = self._inside(ix, iy)
        ix, iy = ix[ok], iy[ok]
        unknown = self.grid[ix, iy] == UNKNOWN
        self.grid[ix[unknown], iy[unknown]] = FREE

    def update_from_points(self, camera_xy, points, max_range=4.0, open_ends=None):
        """points: (N, 3) world hit points (NaN rows are ignored). open_ends: (M, 2) points where
        rays travelled to max_range without a return; they only carve free space.
        """
        points = np.asarray(points, dtype=float)
        camera = np.asarray(camera_xy, dtype=float)
        hit = np.isfinite(points).all(axis=1)
        self._carve(camera, points[hit, :2], stop_short=self.cell / 2, max_range=max_range)
        if open_ends is not None:
            self._carve(camera, np.asarray(open_ends, dtype=float), max_range=max_range)
        # Floor hits are free at the hit cell; obstacle hits are occupied (sticky).
        z = points[:, 2]
        floor = hit & (z < FLOOR_HEIGHT)
        solid = hit & (z >= FLOOR_HEIGHT) & (z <= MAX_HEIGHT)
        if floor.any():
            fx, fy = self.index(points[floor, 0], points[floor, 1])
            ok = self._inside(fx, fy)
            fx, fy = fx[ok], fy[ok]
            unknown = self.grid[fx, fy] == UNKNOWN
            self.grid[fx[unknown], fy[unknown]] = FREE
        if solid.any():
            # Nudge the mark a quarter cell past the surface so it lands in the obstacle's cell.
            direction = points[solid, :2] - camera
            norm = np.maximum(np.hypot(direction[:, 0], direction[:, 1]), 1e-9)[:, None]
            marks = points[solid, :2] + direction / norm * (self.cell / 4)
            ox, oy = self.index(marks[:, 0], marks[:, 1])
            ok = self._inside(ox, oy)
            self.grid[ox[ok], oy[ok]] = OCCUPIED

    def update_from_frame(self, frame, max_range=4.0, column_stride=6, row_stride=12):
        """Unproject a sparse grid of depth pixels exactly like vision.unproject, vectorised."""
        meta = frame.metadata
        height, width = frame.depth.shape
        rows = np.arange(row_stride // 2, height, row_stride)
        cols = np.arange(column_stride // 2, width, column_stride)
        v, u = np.meshgrid(rows, cols, indexing="ij")
        z = frame.depth[v, u].astype(float)
        valid = np.isfinite(z) & (z > 0.05)
        near = valid & (z <= max_range)
        rotation = np.asarray(meta["camera_rotation"])
        origin = np.asarray(meta["camera_position"])

        def to_world(depth):
            camera = np.stack([((u + 0.5) - meta["cx"]) / meta["fx"] * depth,
                               -((v + 0.5) - meta["cy"]) / meta["fy"] * depth, -depth], axis=-1)
            return origin + camera @ rotation.T

        world = to_world(np.where(near, z, 1.0))
        world[~near] = np.nan
        open_ends = to_world(np.full_like(z, max_range))[valid & ~near][:, :2]
        self.update_from_points(origin[:2], world.reshape(-1, 3), max_range, open_ends)

    def blocked(self, inflate_m=INFLATE_M):
        """Cells the robot centre may not enter: occupied (inflated) or never observed."""
        import cv2

        reach = int(math.ceil(inflate_m / self.cell))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
        grown = cv2.dilate((self.grid == OCCUPIED).astype(np.uint8), kernel) > 0
        return grown | (self.grid == UNKNOWN)

    def frontier_cells(self, min_cluster=3):
        """Free cells touching unknown space, in clusters of at least `min_cluster` cells."""
        import cv2

        free, unknown = self.grid == FREE, self.grid == UNKNOWN
        touching = np.zeros_like(unknown)
        touching[1:, :] |= unknown[:-1, :]
        touching[:-1, :] |= unknown[1:, :]
        touching[:, 1:] |= unknown[:, :-1]
        touching[:, :-1] |= unknown[:, 1:]
        mask = (free & touching).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.T.copy(), connectivity=8)
        keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_cluster]
        return [np.argwhere(labels.T == i) for i in keep]

    def plan_to_frontier(self, position, avoid=(), min_distance=0.4, avoid_radius=0.6, inflate_m=INFLATE_M):
        """Waypoints (world x, y) to the cheapest reachable frontier, or None.

        Dijkstra over known-free, uninflated-blocked cells; `avoid` lists world points whose
        neighbourhood is excluded so a frontier that failed once is not retried forever.
        """
        import heapq

        blocked = self.blocked(inflate_m)
        start = tuple(int(v) for v in self.index(position[0], position[1]))
        if not self._inside(*start):
            return None
        # The robot may already stand inside a wall's inflation: let it leave through known-free cells.
        near = (slice(max(start[0] - 3, 0), start[0] + 4), slice(max(start[1] - 3, 0), start[1] + 4))
        blocked[near] &= self.grid[near] != FREE
        targets = {}
        for cluster in self.frontier_cells():
            for cell in cluster:
                x, y = self.center(cell[0], cell[1])
                if any(math.hypot(x - ax, y - ay) < avoid_radius for ax, ay in avoid):
                    continue
                targets[(int(cell[0]), int(cell[1]))] = (x, y)
        if not targets:
            return None
        cost, parent, queue = {start: 0.0}, {}, [(0.0, start)]
        goal = None
        while queue:
            current_cost, current = heapq.heappop(queue)
            if current_cost > cost.get(current, math.inf):
                continue
            x, y = self.center(*current)
            if current in targets and math.hypot(x - position[0], y - position[1]) >= min_distance:
                goal = current
                break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    other = (current[0] + dx, current[1] + dy)
                    if not self._inside(*other) or blocked[other]:
                        continue
                    if dx and dy and (blocked[current[0] + dx, current[1]] or blocked[current[0], current[1] + dy]):
                        continue
                    new = current_cost + math.hypot(dx, dy) * self.cell
                    if new < cost.get(other, math.inf):
                        cost[other], parent[other] = new, current
                        heapq.heappush(queue, (new, other))
        if goal is None:
            return None
        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        points = [self.center(*c) for c in cells]
        # Waypoints every ~0.5 m, always ending on the frontier cell itself.
        spaced = [points[i] for i in range(5, len(points), 5)]
        if not spaced or spaced[-1] != points[-1]:
            spaced.append(points[-1])
        return spaced

    def as_scene(self, inflate_m=INFLATE_M):
        """A planning scene made only of perceived obstacles, for navigation.execute().

        `inflate_m` is the same clearance the frontier planner used, so a route it found is not
        then rejected as unreachable by the navigator's own (otherwise fixed) padding.
        """
        pad = max(0.03, inflate_m - 0.05 - 0.14)  # cell radius and robot radius are already counted
        cells = np.argwhere(self.grid == OCCUPIED)
        obstacles = []
        for index, (ix, iy) in enumerate(cells):
            x, y = self.center(ix, iy)
            obstacles.append({"name": f"cell_{index}", "x": float(x), "y": float(y), "radius": self.cell / 2,
                              "path_margin": pad})
        half = self.n * self.cell / 2
        return {"bounds": [-half, half, -half, half], "objects": {}, "obstacles": obstacles,
                "walls": [], "grid": self.cell}

    def explored_cells(self):
        return int((self.grid != UNKNOWN).sum())
