# Research plan: where to put the improvement (harness, teacher SFT, outcome-reward RL)

Status: plan written after the 2026-11 review. Evidence behind it: `docs/improvement-evidence.md`, `experiments/lang-v1/summary.md`.

## Question

For a skill-level embodied agent, with the same budget, which layer should receive the improvement effort (harness, supervised
fine-tuning on teacher labels, or reinforcement learning on simulator-truth outcomes), and which gains are retained on layouts never
used for any decision? Conditional side question: can the agent recognise an instruction that is clear but physically impossible?
(Phase 0 result below says the data gives no support for this yet.)

## Evaluation protocol

* Split by layout seed into evolution (free to use), gate (accept/reject only) and sealed (run once, at least 200 valid tasks).
* Primary metric: task outcome success judged by simulator truth, independent of the rule teacher. Secondary: strict action-label
  agreement (kept for continuity), clarify/refuse correctness on impossible tasks, steps.
* Statistics: repeat identical configurations 3-5 times for a noise floor; paired exact McNemar plus bootstrap; report intervals only.
* Gate: the replay canary runs before any simulator run.
* Budget logging per arm: GPU hours, simulator episodes, manual edits.

## Arms

| Arm | Update target | Notes |
|---|---|---|
| A0 | none | frozen `lang-v2` plus current harness |
| A1 | harness | manual failure-driven loop, then an automated coding-agent loop with a fixed iteration budget |
| A2 | weights, teacher labels | vary data amount |
| A3 | weights, simulator-outcome reward | single-step GRPO on recorded states; reward needs a reward-hacking check (always-`stop` policy) first |
| A4 | A1 + A3 | additivity |

## Phases and decision rules

| Phase | Content | Rule |
|---|---|---|
| P0 | unreachable-target statistics, outcome-equivalent rescoring, records, splits, noise floor | if the 200-task interval half-width exceeds about 7 points, add tasks first |
| P1 | freeze A0; re-validate rows 5-11 of the evidence table one switch at a time on independent seeds; manual A1 | keep only changes with a gate-set gain |
| P2 | A2 data-amount curve | flat curve is recorded as "data is not the bottleneck" |
| P3 | A3 pilot | gain within noise on the gate set: record as a negative arm and stop |
| P4 | sealed run, once per arm | no tuning afterwards |
| P5 | public benchmark and a second base model (needs the teacher's platform) | required for a top-venue submission |
| P6 | writing, narrative chosen by results | see below |

Narratives: harness wins -> "cheap harness iteration beats retraining" (empirical paper with negative results); RL wins -> "simulator-truth
reward removes the teacher loop" (needs reward-hacking evidence); all within noise -> evaluation-methodology report, not a top-venue paper.

## Risks

Close prior art (Self-Harness, ASPIRE, ModularRSI); 8 GiB GPU limits the RL arm; sample size may bury all arm differences;
simulator-only and single robot. Not yet checked: prior work on skill-level outcome-reward RL and on feasibility-aware refusal.

## Phase 0 results so far (seeds 99000/99100, 16 layouts)

1. **Unreachable targets: none.** `tools/feasibility.py` finds every ball reachable on all 16 layouts at half width 0.12 m
   (15-16 of 15-16 balls per layout). The 13 failures of `merged-v2` therefore are not caused by a physically blocked target in the
   sense this check measures; it does not cover time-budget limits or narrow gaps tighter than 0.12 m. The feasibility-aware side
   question has no support from the current failures; it would need deliberately constructed impossible tasks.
2. **Outcome-equivalent rescoring** (`tools/equivalent_score.py`; counts `done` for an expected `stop` after clean motion; strict stays the headline):

   | Run | Strict | Outcome-equivalent |
   |---|---|---|
   | cand-v2 | 61/74 (82.4%) | 65/74 (87.8%) |
   | cand-v4 | 60/74 (81.1%) | 64/74 (86.5%) |
   | merged-v2 | 61/74 (82.4%) | 65/74 (87.8%) |

   The gaps between runs are unchanged by the rescoring. The equivalence is still defined by the same rule teacher's expectation,
   so it does not remove the teacher dependence; the independent simulator-truth success scorer is still to be built.
3. Still open in P0: layout-seed registry and three-way split, noise-floor repeats, HITCH row in `docs/related-work.md`, lang-v4 canary table.
