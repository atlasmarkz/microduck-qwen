"""Reproduce failing explore runs and dump map + true layout (diagnostic, not part of the package)."""
import json
import sys
from pathlib import Path

import numpy as np

from microduck_qwen.skills import SkillExecutor
from microduck_qwen.sim_runner import start_scene, stop_process
from microduck_qwen.vision import capture, propose_targets
from microduck_qwen.vision_world import WORLD_LEVELS

seed, out = int(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
level = WORLD_LEVELS[2]
process, log, info, _ = start_scene(out, seed, "none", 7940, collect_gt=True, world=level["rooms"], density=level["density"])
try:
    ex = SkillExecutor(info, out / "run")
    ex.run({"action": "return_home"})
    first = capture(info["port"])
    seen = {c["color"] for c in propose_targets(first) if c.get("kind", "ball") == "ball"}
    scene = json.loads(Path(info["scene_file"]).read_text())
    balls = [(o["color"], round(o["x"], 2), round(o["y"], 2), o["room"]) for o in scene["objects"].values()]
    hidden = sorted({b[0] for b in balls} - seen)
    print("SEED", seed, "balls", balls, "hidden", hidden, flush=True)
    for color in hidden * 2:
        ex.run({"action": "return_home"})
        r = ex.run({"action": "explore", "color": color})
        print("RESULT", {k: str(v)[:200] for k, v in r.items() if k in ("success", "error", "observation")}, flush=True)
        print("EXPLORE", color, {k: r.get(k) for k in ("found", "stopped", "explored_cells", "elapsed_s")},
              "iters", len(r.get("iterations", [])), "inv", [(i["color"], round(i["x"], 1), round(i["y"], 1)) for i in r.get("inventory", [])], flush=True)
        for it in r.get("iterations", []):
            print("   ", it, flush=True)
    (out / "scene.json").write_text(json.dumps(scene))
finally:
    stop_process(process)
