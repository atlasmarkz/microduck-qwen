"""Task outcome judged by simulator truth, independent of the rule teacher's per-step labels.

The strict score (`agent_eval`) counts an episode as correct only when every decision equals the teacher's: `done` where
the teacher wrote `stop` fails although the robot is exactly where it should be. This scorer asks only what physically
happened, using the MuJoCo truth the episode already records:

  * the robot's oracle pose before every decision (`memory.odometry_relative_home`, 1 cm resolution, plus the session
    home pose); the pose before a decision is the resting pose after the previous skill;
  * the true ball positions (`simulator-0.scene.json`) and, for "the leftmost ball in the photo", the MuJoCo segmentation
    of the first frame (`step-000-truth.json`);
  * scene contacts (`scene_contact_steps_delta`).

What the task requires is read from the task definition (`teacher_specs`: which colour, leftmost, move, return), never
from the teacher's step labels. Success: the model ended the task itself with `done` or `stop`; the required ball (any
ball of the colour, or the first frame's leftmost ball) was reached within the teacher's own resting-distance bound;
a required return ended within the return-home bound of home, after the reach; a required move matches the executor's
tolerance; and nothing in the scene was touched. Thresholds are the executor's own (see constants below).

Verdicts: `success`, `fail`, and three cuts where the episode stopped before the outcome was visible:
  * `cut_teacher`: the teacher could not label a later step (e.g. "no confidently visible blue ball"); the episode ended
    because of the labeller, not the robot;
  * `cut_deviation`: the model chose a different non-terminal skill and the strict evaluation stops right after it, so
    whether that route would have worked is unobserved;
  * `cut_infra`: the harness could not restore the task start.
The headline counts cuts as failures (lower bound) and reports the upper bound with cuts counted as successes.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .agent_data import APPROACH_TRUE_DISTANCE_M, ARRIVAL_TOLERANCE_M, COAST_ALLOWANCE_M
from .compare_runs import compare, is_invalid

HOME_STOP_M = 0.10            # return_home's point goal (SkillExecutor._return_to_pose)
HOME_TRUE_DISTANCE_M = HOME_STOP_M + ARRIVAL_TOLERANCE_M + COAST_ALLOWANCE_M
MOVE_TOLERANCE_M = 0.05       # SkillExecutor._bounded: |achieved - requested| <= 0.05 m
MOVE_LATERAL_M = 0.08         # ... and lateral drift < 0.08 m
TERMINALS = ("done", "stop")
CUTS = ("cut_teacher", "cut_deviation", "cut_infra")


def requirements(specs):
    """What the task asks for, from its definition: reach target, move, return home."""
    need = {"reach": None, "move": None, "home": False}
    for spec in specs:
        if "teacher_nearest" in spec or "teacher_color" in spec:
            need["reach"] = ("colour", spec.get("teacher_nearest") or spec.get("teacher_color"))
        elif "teacher_target" in spec:
            need["reach"] = ("leftmost", spec["teacher_target"])
        elif spec.get("action") == "move":
            need["move"] = spec
        elif spec.get("action") == "return_home":
            need["home"] = True
    return need


def world_pose(memory, home):
    odometry = memory["odometry_relative_home"]
    return (home["x"] + odometry["x"], home["y"] + odometry["y"], home["yaw"] + math.radians(odometry["yaw_deg"]))


def target_positions(need, scene, first_truth):
    balls = {name: obj for name, obj in scene["objects"].items() if obj.get("shape", "sphere") == "sphere"}
    kind, value = need["reach"]
    if kind == "colour":
        return [(o["x"], o["y"]) for o in balls.values() if o["color"] == value]
    seen = [(info["pixel"][0], name) for name, info in (first_truth or {}).get("objects", {}).items()
            if name in balls and info.get("visible_pixels", 0) > 0 and info.get("pixel")]
    if not seen:
        return []
    name = min(seen)[1] if value == "left" else max(seen)[1]
    return [(balls[name]["x"], balls[name]["y"])]


def score_episode(episode, scene, home, first_truth=None):
    """{"verdict": ..., "reason": ...} for one recorded independent-actor episode."""
    steps = [s for s in episode.get("steps", []) if s.get("memory")]
    error = str(episode.get("error") or "")
    if not steps or "cannot restore task start" in error:
        return {"verdict": "cut_infra", "reason": error or "no steps"}
    need = requirements(episode["teacher_specs"])
    poses = [world_pose(s["memory"], home) for s in steps]
    targets = target_positions(need, scene, first_truth) if need["reach"] else []
    if need["reach"] and not targets:
        return {"verdict": "cut_infra", "reason": "target not identifiable from truth"}

    def near_target(xy):
        return any(math.dist(xy, t) <= APPROACH_TRUE_DISTANCE_M for t in targets)

    reached_at = None
    # The pose before each decision is the resting pose after the previous skill. The last skill's own end pose only
    # matters when it is terminal (no motion), so poses[-1] covers it; a non-terminal last step is a fail or a cut anyway.
    for index, pose in enumerate(poses):
        if need["reach"] and near_target(pose[:2]):
            reached_at = index
            break
    last = steps[-1]
    action = (last.get("prediction") or {}).get("action")
    if action is None:
        return {"verdict": "fail", "reason": "invalid model output"}
    name = action["action"]
    if name not in TERMINALS + ("clarify", "unsupported"):
        if "no confidently visible" in error or "teacher" in error:
            return {"verdict": "cut_teacher", "reason": error}
        expected = (last.get("expected") or {}).get("action")
        if expected is not None and name != expected and "wrong task action" in error:
            return {"verdict": "cut_deviation", "reason": f"model {name}, teacher {expected}"}
        return {"verdict": "fail", "reason": error or "no terminal decision"}
    if name not in TERMINALS:
        return {"verdict": "fail", "reason": f"ended with {name}"}
    if episode.get("scene_contact_steps_delta", 0) != 0:
        return {"verdict": "fail", "reason": "touched the scene"}
    final = poses[-1]
    if need["reach"]:
        if reached_at is None:
            return {"verdict": "fail", "reason": "target never reached"}
        if not need["home"] and not near_target(final[:2]):
            return {"verdict": "fail", "reason": "left the target before ending"}
    if need["home"] and math.dist(final[:2], (home["x"], home["y"])) > HOME_TRUE_DISTANCE_M:
        return {"verdict": "fail", "reason": f"ended {math.dist(final[:2], (home['x'], home['y'])):.2f} m from home"}
    if need["move"]:
        start = poses[0]
        sign = 1 if need["move"]["direction"] == "forward" else -1
        dx, dy = final[0] - start[0], final[1] - start[1]
        along = sign * (dx * math.cos(start[2]) + dy * math.sin(start[2]))
        lateral = abs(-dx * math.sin(start[2]) + dy * math.cos(start[2]))
        if abs(along - need["move"]["distance_m"]) > MOVE_TOLERANCE_M or lateral >= MOVE_LATERAL_M:
            return {"verdict": "fail", "reason": f"moved {along:.2f} m (lateral {lateral:.2f})"}
    return {"verdict": "success", "reason": f"ended with {name}"}


def score_run(run_dir):
    """{(seed, task_id): {"strict": bool, "verdict": str, "reason": str, "instruction": str}} for valid tasks of one run."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    cache, out = {}, {}
    for result in report["results"]:
        if is_invalid(result):
            continue
        seed_dir = run_dir / f"seed-{result['seed']}"
        if seed_dir not in cache:
            simulator = json.loads((seed_dir / "simulator-0.json").read_text(encoding="utf-8"))
            scene = json.loads((seed_dir / "simulator-0.scene.json").read_text(encoding="utf-8"))
            cache[seed_dir] = (scene, simulator["home_pose"])
        scene, home = cache[seed_dir]
        task_dir = seed_dir / Path(result["episode"]).parent.name
        path = task_dir / "episode.json"
        if not path.exists():
            verdict = {"verdict": "cut_infra", "reason": str(result.get("error"))}
        else:
            episode = json.loads(path.read_text(encoding="utf-8"))
            truth_path = task_dir / "step-000-truth.json"
            first = json.loads(truth_path.read_text())["ground_truth"] if truth_path.exists() else None
            verdict = score_episode(episode, scene, home, first)
        out[(result["seed"], result["task_id"])] = dict(verdict, strict=bool(result["correct"]), instruction=result["instruction"])
    return out


def summary(scores):
    verdicts = Counter(v["verdict"] for v in scores.values())
    cuts = sum(verdicts[c] for c in CUTS)
    return {"tasks": len(scores), "strict": sum(v["strict"] for v in scores.values()), "truth": verdicts["success"],
            "truth_upper": verdicts["success"] + cuts, "verdicts": dict(verdicts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, nargs="+", help="run directory(ies), e.g. runs/lang/merged-v2-99000 runs/lang/merged-v2-99100")
    parser.add_argument("--b", type=Path, nargs="+", help="second run for a paired comparison on truth success (cuts count as failures)")
    parser.add_argument("--details", action="store_true", help="list every task where strict and truth disagree, and every non-success")
    args = parser.parse_args()
    runs = {"A": args.a} | ({"B": args.b} if args.b else {})
    scored = {}
    for label, directories in runs.items():
        scores = {}
        for directory in directories:
            scores.update(score_run(directory))
        scored[label] = scores
        s = summary(scores)
        print(f"{label}: strict {s['strict']}/{s['tasks']}  truth {s['truth']}/{s['tasks']}  "
              f"(upper bound with cuts {s['truth_upper']}/{s['tasks']})  {s['verdicts']}")
        if args.details:
            for key, v in sorted(scores.items()):
                if v["verdict"] != "success" or not v["strict"]:
                    print(f"  {key} strict={v['strict']!s:5} {v['verdict']:13} {v['reason'][:70]} | {v['instruction'][:40]}")
    if args.b:
        a = {k: v["verdict"] == "success" for k, v in scored["A"].items()}
        b = {k: v["verdict"] == "success" for k, v in scored["B"].items()}
        r = compare(a, b)
        print(f"truth, paired: A {r['a_correct']}/{r['pairs']}  B {r['b_correct']}/{r['pairs']}  B-only {r['only_b']}  "
              f"A-only {r['only_a']}  McNemar p={r['mcnemar_p']:.3f}  diff {r['diff_rate']:+.3f} "
              f"(95% {r['diff_ci95'][0]:+.3f}..{r['diff_ci95'][1]:+.3f})")


if __name__ == "__main__":
    main()
