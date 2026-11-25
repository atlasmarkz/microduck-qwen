"""Snapshot microduck_rl's live task registry into microduck_qwen/upstream_tasks.json.

Importing the registry needs mjlab, torch and warp, which the Colab notebook must not install, so
the notebook reads this snapshot instead. Run it in a throwaway environment built from upstream's
uv.lock (mjlab 1.3.0, warp-lang 1.12.0, torch 2.9.1, scipy) with the locked BAM source:

    python tools/export_upstream_tasks.py --rl <microduck_rl checkout> --bam <bam checkout> \
        --rl-revision <sha> --bam-revision <sha>
"""

import argparse
import inspect
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def name_of(func):
    return getattr(func, "__name__", type(func).__name__)


def summary(func):
    """First paragraph of the function's own docstring (upstream wording, not ours)."""
    text = inspect.getdoc(func) or inspect.getdoc(getattr(func, "__call__", None)) or ""
    return " ".join(text.split("\n\n")[0].split())[:240]


def robot_xml(spec_fn, constants):
    for name in re.findall(r"\b[A-Z_]+_XML\b", inspect.getsource(spec_fn)):
        return Path(getattr(constants, name)).name
    raise ValueError(f"no XML constant in {spec_fn}")


def command_ranges(cfg):
    ranges = getattr(cfg, "ranges", None)
    if ranges is None:
        return {}
    values = vars(ranges) if hasattr(ranges, "__dict__") else {"range": ranges}
    return {k: v for k, v in values.items() if v is not None}


def describe(task, env, rl, constants):
    stages = {}
    for term in env.curriculum.values():
        if "reward_name" in term.params and "weight_stages" in term.params:
            stages[term.params["reward_name"]] = term.params["weight_stages"]
    rewards = [{"name": name, "func": name_of(term.func), "weight": term.weight,
                "final": stages[name][-1]["weight"] if name in stages else term.weight} for name, term in env.rewards.items()]
    algorithm = rl.algorithm
    return {
        "id": task,
        "terrain": "rough" if "-Rough-" in task else "slope" if "Slope" in task else "flat",
        "robot_xml": robot_xml(env.scene.entities["robot"].spec_fn, constants),
        "other_entities": [name for name in env.scene.entities if name != "robot"],
        "episode_s": env.episode_length_s,
        "control_hz": round(1 / (env.decimation * env.sim.mujoco.timestep)),
        "rewards": rewards,
        "commands": {name: {"type": type(cfg).__name__, "ranges": command_ranges(cfg)} for name, cfg in env.commands.items()},
        "actor_obs": list(env.observations["actor"].terms),
        "critic_obs": list(env.observations["critic"].terms),
        "terminations": list(env.terminations),
        "events": list(env.events),
        "curriculum": list(env.curriculum),
        "ppo": {"experiment": rl.experiment_name, "max_iterations": rl.max_iterations,
                "steps_per_env": rl.num_steps_per_env, "hidden": list(rl.actor.hidden_dims),
                "learning_rate": algorithm.learning_rate, "gamma": algorithm.gamma, "lam": algorithm.lam,
                "entropy": algorithm.entropy_coef, "clip": algorithm.clip_param,
                "symmetry": getattr(algorithm, "symmetry_cfg", None) is not None},
    }


def fold_backlash(tasks):
    """Backlash twins differ only in the robot model; keep id and model, and prove the rest is equal."""
    by_id = {task["id"]: task for task in tasks}
    for task in tasks:
        base_id = task["id"].replace("-Backlash", "")
        if base_id == task["id"]:
            yield task
            continue
        base = by_id[base_id]
        for key in ("rewards", "commands", "actor_obs", "terminations", "episode_s", "terrain"):
            if task[key] != base[key]:
                raise ValueError(f"{task['id']} differs from {base_id} in {key}")
        yield {"id": task["id"], "base": base_id, "robot_xml": task["robot_xml"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--rl-revision", required=True)
    parser.add_argument("--bam-revision", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "microduck_qwen/upstream_tasks.json")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.rl / "src"), str(args.bam)]
    import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
    from importlib.metadata import version
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
    from mjlab_microduck.robot import microduck_constants

    names = [name for name in list_tasks() if "MicroDuck" in name]
    configs = {name: load_env_cfg(name) for name in names}
    tasks = list(fold_backlash([describe(n, configs[n], load_rl_cfg(n), microduck_constants) for n in names]))
    docs = {name_of(term.func): summary(term.func) for cfg in configs.values() for term in cfg.rewards.values()}
    args.out.write_text(json.dumps({
        "source": "pollen-robotics/microduck_rl", "revision": args.rl_revision,
        "bam_revision": args.bam_revision, "mjlab": version("mjlab"),
        "reward_docs": dict(sorted(docs.items())), "tasks": tasks,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
