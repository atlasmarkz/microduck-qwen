# Improvement evidence table and the RSI question

Framework: the evidence levels and the five dimensions come from the Awesome RSI scope notes
(https://github.com/lobehub/awesome-rsi). I read that list's terminology, its embodied, harness, self-verification and safety sections;
I did not read the listed papers. All numbers below are copied from `experiments/*/summary.md`; none are new measurements.

Evidence levels used here:
* **none**: no measurable change, or a change that cannot be told from run-to-run noise (about 10 of 73 tasks flip between identical runs).
* **task gain (dev)**: better on the tasks the change was developed against; not shown on independent tasks.
* **retained gain**: a persistent change that helps on tasks/layouts never used for any decision, paired test reported.
* **improver gain**: the mechanism that produces improvements became better. Nothing in this project reaches this level.

## Table

| # | Change | Update target | Feedback source | Retained artifact | Validation | Result | Level |
|---|---|---|---|---|---|---|---|
| 1 | self-evolve v1 (1 round, own tasks) | weights | own perception, odometry, collision; self-posed tasks | adapter self-v2 | 6 held-out tasks, accept gate | 0.83 -> 0.67, rejected | none (n = 6) |
| 2 | self-evolve v3 (3 rounds) | weights | same | adapters, all rejected | 18 held-out tasks | 15/18 -> 10, 14, 13; unchanged baseline ranged 11-15; control retrain 14/18 | none |
| 3 | self-evolve v4 (3 rounds, hindsight relabel) | weights | same plus hindsight terminal correction | `self-v4-r2` | 36 held-out tasks, 5-point slack | 29/36 -> 25, 28, 31 | none (noise floor on 36 tasks not re-measured; round 1 accepted only by the slack) |
| 4 | lang-v1 expert-corrected retrain | weights | rule teacher corrects rollouts | `lang-v1` | unseen layouts 98000/98100 | 42/80 -> 52/80, intervals overlap; explore 10/48 -> 20/48 | task gain (dev) |
| 5 | faithful map (no free space carved through walls) | executor code | failure analysis | code | same 80 tasks, paired | 52 -> 64, p = 0.012 | task gain (dev) |
| 6 | lang-v2 retrain | weights | teacher | `lang-v2` | same 80 tasks, paired | 64 -> 63, p = 1.0 | none |
| 7 | pad floor, four clearance tiers | executor code | failure analysis | code | same 80 tasks, paired | 64 -> 69, p = 0.27 | none |
| 8 | lang-v3 (move-only phrasing data) | weights | teacher | not adopted | 80 tasks plus replay canary | 69 -> 66; explore 41 -> 34; canary flagged `explore->approach` 44 -> 41 | negative; the gate worked |
| 9 | explore/approach recoveries | executor code | literature plus failure analysis | code | 73 valid tasks, on/off switch | 60 -> 61, p = 1.0; 4 same-layout saves | none |
| 10 | narrowest clearance tier (0.20 m) | executor code | one failing layout | code | 73 valid tasks | 61 -> 64, p = 0.45; tier never exercised | none (one-layout mechanism check only) |
| 11 | settle-on-target before explore returns | executor code | two failing tasks | code | 73 valid tasks | 64 -> 64, p = 1.0; found and fixed a bug in the step | none overall; mechanism shown on 2 layouts |
| 12 | executor feedback line after move/turn | prompt/executor text | failure analysis (cause found in `agent_prompt`) | code | replay canary plus 32 never-used layouts, one task type | canary `move->done` 41/48 -> 48/48; simulator 26/32 -> 32/32, p = 0.031 | **retained gain** (one task type) |
| 13 | longer stop feedback after approach | prompt text | failure analysis | removed | replay canary | 626 -> 608 | negative |
| 14 | lang-v4 data (English clarify) | weights | teacher | not adopted | canary plus fresh seeds 99000/99100 | canary English clarify 138/138; end-to-end 61/74 vs 60/74, p = 1.0 | none |
| 15 | far re-acquisition | executor code | failure analysis | code | 12 task pairs on 2 layouts | 4/12 -> 8/12, p = 0.125 | none (too few pairs; mechanism works when it fires) |
| 16 | harden balls, retry cap, start-of-approach re-acquire | executor code | failures on seeds 99000/99100 | code (merged, 172 tests pass) | paired re-run `merged-v2` running | not available yet | pending |
| 17 | action legality check plus model-scored legal fallback | harness (decision filter) | prompt contract and training-data state classes | code, flag off by default | offline replay of 3908 recorded states, both arms from one run | `start->explore` 828/836 -> 834/836; 6 decisions only with it, 0 only without; 0 teacher labels flagged | none yet (offline only, 5 distinct layout/task pairs; simulator effect not measured) |

Gaps this table exposes:
* Only row 12 is a retained gain, and it covers one task type ("move forward 20 cm").
* Rows 5-7, 9-11 and 15 were developed on layouts that were later reused for decisions. The fresh seeds 99000/99100 measured the whole stack once (61/74 = 82.4%), not each change.
* The replay canary (rows 8, 12, 13) is the only mechanism that caught regressions before an expensive run. It is an evaluation tool, not an improver.
* The honest overall number on never-used layouts is 82%, below the 84-88% measured on development layouts.

## Is RSI necessary for this project?

No, and I would not use the label.

1. **The definition is not met.** Recursive self-improvement needs the improvement-producing mechanism to be improved. Here the improver is a rule teacher plus hand-written fixes plus me reading failure taxonomies. The model does not generate, select or verify its own improvements in any way that was shown to work.
2. **The one genuinely self-driven loop did not work.** Rows 1-3 (self-posed tasks, self-verification, self-generated training rows) produced no demonstrated gain over seven rounds; most rounds were rejected by the gate or were inside the noise. The gains in this project came from teacher-corrected data (row 4, weakly) and from executor-side changes diagnosed from failures (rows 5 and 12).
3. **The vocabulary is borrowed, the evidence is not.** The list itself says not every entry demonstrates RSI in the strict sense. Calling this project "self-evolving" would invite the reviewer question "where is the improver gain?", and the honest answer is that there is none.
4. **What is worth keeping from the RSI framing**: the three evidence levels and the five-dimension table (above). They discipline the claims. The project is better described as failure-driven iterative improvement with a release gate.
5. **Where a self-improvement angle could become real.** The manual loop (failure taxonomy -> hypothesis -> code change -> replay canary -> paired test) is what systems like ASPIRE automate with a coding agent. Automating it would be a defensible research question, but it requires an improver-gain experiment under a fixed budget, which is a different and larger project than the current one.

## Consequence for the paper

* Claim "what kind of update works": compare update targets under one protocol (self-generated weights data, teacher-corrected weights data, executor code, prompt text). The current evidence already points one way (rows 1-3 vs 5, 12), but it needs the same protocol, independent layouts and more tasks.
* Report negative results (rows 2, 3, 6, 8, 13, 14) as findings.
* Move retained-gain validation of rows 5-11 onto independent seeds, one change at a time, before any claim.
