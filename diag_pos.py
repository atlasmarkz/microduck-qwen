import json, sys
from pathlib import Path
from microduck_qwen.skills import SkillExecutor
from microduck_qwen.sim_runner import start_scene, stop_process
from microduck_qwen.navigation import clearance, inside, nearest_obstacle
from microduck_qwen.vision_world import WORLD_LEVELS
seed, out = int(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
lv = WORLD_LEVELS[2]
p, log, info, _ = start_scene(out, seed, "none", 7940, collect_gt=True, world=lv["rooms"], density=lv["density"])
try:
    ex = SkillExecutor(info, out / "run")
    for k in range(2):
        st = ex.observer.read()
        pos = st["trunk"][:2]
        print("pos", [round(float(v), 2) for v in pos], "yaw", round(float(st["yaw"]), 2), "inside", inside(pos, ex.scene),
              "clear", round(clearance(pos, ex.scene), 3), "nearest", nearest_obstacle(pos, ex.scene), flush=True)
        r = ex.run({"action": "return_home"}); print("home", r["success"], r.get("error"), flush=True)
finally:
    stop_process(p)
