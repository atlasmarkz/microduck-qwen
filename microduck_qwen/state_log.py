"""Passive full-pose log of an evaluation episode, so third-person views can be rendered afterwards.

A second read-only observer connection, polled at 10 Hz: trunk pose and all joint positions. It never
sends a command, and a failed read is skipped rather than raised into the episode.
"""

import json
import threading
import time

from .navigation import Observer

PERIOD_S = 0.1


class StateLog:
    def __init__(self, port, path):
        self.port, self.path = port, path
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="state-log", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join(timeout=3)

    def _run(self):
        observer, last = None, None
        with self.path.open("w", encoding="utf-8") as out:
            while not self.stopped.is_set():
                started = time.monotonic()
                try:
                    if observer is None:
                        observer = Observer(self.port)
                    state = observer.read()
                    if state["sim_time"] != last:
                        last = state["sim_time"]
                        out.write(json.dumps({"t": round(last, 3), "xyz": [round(v, 4) for v in state["trunk"]],
                                              "quat": [round(v, 5) for v in state["imu"]["quat"]],
                                              "joints": [round(v, 4) for v in state["positions"]]}) + "\n")
                except (OSError, RuntimeError, ValueError, KeyError):
                    if observer is not None:
                        try:
                            observer.close()
                        except OSError:
                            pass
                    observer = None
                self.stopped.wait(max(0.0, PERIOD_S - (time.monotonic() - started)))
        if observer is not None:
            try:
                observer.close()
            except OSError:
                pass
