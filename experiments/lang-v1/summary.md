# lang-v1: wide phrase bank, expert-corrected retrain

Starting adapter `self-v4-r2`; retrained 300 steps on `data/lang-v1` (corrected rollouts on worlds + colour tasks, phrase bank).
Evaluation: unseen layouts (seeds 98000/98100, 8 each), unseen phrasing components (test split of the phrase bank), world level 2.

| adapter | seed 98000 | seed 98100 | total | Wilson 95% |
|---|---|---|---|---|
| self-v4-r2 | 22/40 | 20/40 | 42/80 | ~0.42-0.63 |
| lang-v1 | 27/40 | 25/40 | 52/80 | ~0.54-0.74 |

By task kind (both seeds pooled): plain approach tasks 16/16 + 16/16 for both adapters (no regression);
explore-in-another-room phrasings 10/48 -> 20/48.

Caveats: intervals overlap, the explore gain is the only moving part, and half of explore tasks still fail
(mostly physical/navigation, not parsing: action_type_ok 36->38 of 40). Not a claim of general language robustness.

## Explore robustness fix (same adapter, same 80 tasks)

Failure analysis of lang-v1 showed most explore failures were the skill itself stopping with "no reachable frontier left"
(18 of 28 failures), not wrong model output. Causes found: frontiers added to the avoid list after one navigation refusal,
the frontier planner's clearance differing from the navigator's padding (goal "unreachable"), and exploring from a cramped pose.
Fix: three-tier frontier planning (normal clearance, forget avoid list, 0.25 m clearance), the same clearance passed to the
navigator, `escape_cramped` before explore. Diagnostic: `diag_explore.py`.

| | seed 98000 | seed 98100 | total |
|---|---|---|---|
| lang-v1 before fix | 27/40 | 25/40 | 52/80 |
| lang-v1 after fix | 31/40 | 21/40 | 52/80 |

Explore tasks 20/48 -> 21/48; "no reachable frontier" failures 18 -> 9. No net gain: the failures moved. New losses are
between-task `return_home` failures (2 -> 6: clearance stop or 65-87 s navigation timeouts), 2 `clarify` outputs, and runs where
explore succeeded but the later approach/closing step failed. One diagnostic seed (98003) went 0/3 -> 2/2 on explore alone.
Noise between identical runs is about +/-4 tasks of 40, so neither direction is established.
