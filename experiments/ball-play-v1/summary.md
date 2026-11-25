# ball-play-v1: kick a found ball, or push it back to the start

Date 2026-11-25. Code `microduck_qwen/sim_ball.py`, `ball_play.py`, `colab_ball.py`; notebook cell 6 of
`notebooks/microduck-qwen-live-demo.ipynb`. Exploratory: no baseline, no repeat of identical runs, thresholds tuned on seed 98107.

## Setup

* Balls of one colour are free bodies: 15 g (the official kick ball), radius 0.07 m (official 0.035 m, kept so perception is
  unchanged), rolling friction 0.002 with condim 6. Without rolling resistance a pushed ball does not stop (20-30 m on an open floor).
* Walking and pushing: official `alpha_walking`. Kicking: official `ball_kick_left/right` for 0.5 s. No policy was trained.
* Finding and approaching use the camera. Lining up and pushing read the ball position from the simulator.
* The verb is matched by rule; the adapter knows neither skill. The way home is the harness skill `return_home`.

## Results

Model-free chain (harness `explore`/`approach`, then kick or push, then `return_home`), seeds 97001-97006, never used before, one run
each, blue ball, 2x2 world:

| | ball action succeeded | robot also back home |
|---|---|---|
| kick | 6/6, all on the first attempt; ball moved 0.67-0.99 m, 0.56-0.94 m/s | 6/6 |
| push home | 6/6; ball ended 0.17-0.50 m from the start | 5/6 |

* The push failure (97006): the ball came to rest 0.17 m from the start, so the start itself was blocked and `return_home` stopped
  for lack of clearance.
* Only 97001 is a long carry (5.41 m, 113 s, 43 repositionings). In the other five the ball was 0.95-1.12 m from the start, in view
  or in the same area, so 6/6 says little about carrying a ball between rooms. Seed 98107 (tuned on) carried it 4.13 m in 56 s.
* With Qwen on a Colab T4, seed 98107, one run each: push ended 0.51 m from the start (56 s of pushing, 139 s in total);
  kick moved the ball 0.69 m; both returned home.

Kick tolerance on an open floor (`alpha_walking` then kick, 7 cm ball): ball 0.10-0.13 m ahead and 0.02-0.065 m to the kicking side
gives 0.5-0.95 m/s; no fall in 60 trials. Pushing to random goals on an open floor: 16/16.

## Not shown

Perception-only ball tracking, balls next to walls, doorways in unfavourable directions, several balls of the chosen colour,
a real robot.
