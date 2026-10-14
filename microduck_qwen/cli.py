"""Preview or execute locally learned Chinese navigation goals."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import time

from . import ROOT
from .contract import compact
from .model import Planner
from .navigation import Observer, Robot, execute


def run_instruction(planner, instruction, preview=False, timeout=60):
    result = planner.predict(instruction)
    print(compact(result), flush=True)
    if preview:
        return result
    (ROOT / "runs").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    trace = ROOT / f"runs/trajectory-{stamp}.jsonl"
    with ExitStack() as resources:
        execution_lock = resources.enter_context((ROOT / "runs/navigation.lock").open("a"))
        try:
            fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("已有导航指令在执行；先用 stop 或 Ctrl+C 停止它。")
        try:
            info = json.loads((ROOT / "runs/simulator.json").read_text())
            robot = Robot(info["socket"])
            resources.callback(robot.close)
            observer = Observer(info["port"])
            resources.callback(observer.close)
            result["execution"] = execute(result["plan"], robot, observer, trace, timeout_s=timeout,
                                          cancel_file=ROOT/"runs/cancel_navigation")
        except BaseException as error:
            result["execution"] = {"success":False,"error":str(error) or type(error).__name__}
            raise
        finally:
            result["trajectory"] = str(trace)
            (ROOT / f"runs/command-{stamp}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(compact(result["execution"]),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=["plan","run","chat","stop"])
    parser.add_argument("instruction", nargs="?")
    parser.add_argument("--base",action="store_true")
    parser.add_argument("--adapter")
    parser.add_argument("--timeout",type=float,default=60)
    args = parser.parse_args()
    if args.command == "stop":
        (ROOT/"runs").mkdir(exist_ok=True)
        (ROOT/"runs/cancel_navigation").write_text(str(time.time_ns()))
        info = json.loads((ROOT/"runs/simulator.json").read_text())
        robot = Robot(info["socket"])
        try:
            print(robot.stop())
        finally:
            robot.close()
        return
    if args.command in ("plan","run") and not args.instruction:
        parser.error("需要提供自然语言指令")
    planner = Planner(base=args.base,adapter=args.adapter)
    if args.command == "chat":
        print("输入导航指令；exit 退出，Ctrl+C 中断并停止移动。")
        while True:
            try:
                text = input("你 > ").strip()
                if text in ("exit","quit","退出"):
                    break
                if text:
                    run_instruction(planner,text,timeout=args.timeout)
            except (EOFError,KeyboardInterrupt):
                break
            except Exception as error:
                print(f"未执行：{error}",flush=True)
    else:
        run_instruction(planner,args.instruction,preview=args.command=="plan",timeout=args.timeout)


if __name__ == "__main__":
    main()
