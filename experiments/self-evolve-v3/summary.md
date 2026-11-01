# self-evolve-v3 (3 rounds) and diagnostics

Setup: base `agent-explore-v1`, 3 worlds/round (level 2), 4 min free exploration each, 150 training steps,
held-out = 6 layouts x (2 task ids + 1 explore task) = 18 rollouts. Escape manoeuvre (`escape_cramped`) active.

| round | posed | verified | self rows | held-out after | decision |
|---|---|---|---|---|---|
| start | | | | 15/18 | |
| 0 | 12 | 7 | 19 | 10/18 | rejected |
| 1 | 10 | 7 | 18 | 14/18 | rejected |
| 2 | 12 | 9 | 26 | 13/18 | rejected |

## Diagnostics (unchanged `agent-explore-v1`, same 18-task protocol)
- seed 98000: 14, 15, 15 (three runs; same layouts, only sim/sampling nondeterminism)
- seed 98100: 13; seed 98200: 11  (layout-to-layout spread: 11..15)
- Control, base data only, 150 more steps: 14/18 (98000), 13/18 (98100): indistinguishable from the baseline,
  so retraining itself is not what hurts.

## Reading
- Rounds 1 and 2 (14, 13) are inside the baseline range: no evidence either way.
- Round 0 (10/18 on seed 98000, base 14-15 on the same layouts) is the only result outside the range; with one
  evaluation it is suggestive of harmful rows, not proven.
- The 18-task held-out set cannot resolve changes smaller than about 4 tasks. Self rows are ~6% of the
  training set (19 of 299) and training loss is already ~2e-6, so the update is tiny.
- Self-verified data is dominated by colour_approach; explore_find rarely verifies.
Next: larger/repeated held-out evaluation, task-type balanced self data, include failure corrections.
