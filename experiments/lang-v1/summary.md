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

## return_home retry + harness retry (same adapter, same 80 tasks)

`return_home` now backs away (`escape_cramped`) and re-plans on a clearance stop or leg timeout (up to 3 tries); the evaluator retries
an episode whose between-task reset failed and counts persistent ones separately (`infra_failures`; the 98000 run predates that
evaluator change, the 98100 run has it).

| | seed 98000 | seed 98100 | total | explore tasks |
|---|---|---|---|---|
| before fixes | 27/40 | 25/40 | 52/80 | 20/48 |
| explore fix | 31/40 | 21/40 | 52/80 | 21/48 |
| + return_home retry | 30/40 | 30/40 | 60/80 (75%) | 28/48 |

Between-task reset failures 6 -> 0; remaining failures are mostly "no reachable frontier left" (10) and wrong physical results (7).
60/80 vs 52/80 has overlapping 95% intervals and run-to-run spread is about +/-4 per 40, so this is a consistent trend over three
runs of the same adapter, not a significance claim. No retraining was involved: all gains are execution-layer robustness.

## Target memory and surviving safety stops (same adapter, same 80 tasks)

| | seed 98000 | seed 98100 | total | explore tasks |
|---|---|---|---|---|
| + return_home retry | 30/40 | 30/40 | 60/80 | 28/48 |
| + target memory (partly seen balls) | 29/40 | 32/40 | 61/80 | 29/48 |
| + continue after mid-route safety stop | 31/40 | 30/40 | 61/80 | 29/48 |

Target memory fired twice in 80 tasks; no measurable effect. Surviving mid-route safety stops fixed the reproduced layout
(seed 98005, 0 -> found) but "no reachable frontier left" stayed at 10-11 of the ~19 failures across the 80 tasks, so that was not
the main cause overall. The remaining failures are not yet explained; 6-7 are wrong action types by the model (retraining territory).

## Faithful map: no free-space carving through walls (same adapter, same 80 tasks)

Rendering a failing run's map showed the cause of "no reachable frontier left": the camera sits above the 0.4 m walls, so rays with
no return (they pass over a wall or the world edge) were carved as free ground, which leaked free space outside the world and put the
nearest "frontiers" where nothing could be reached. Rays without a return are no longer carved. Consequences handled: ball cells get
less inflation than walls (the navigator keeps 3.5 cm around balls but 20 cm around walls), the navigator gets 5 cm less padding than
the frontier planner (its lattice is half a cell off the map's), explore allows 12 iterations / 300 s, and `escape_cramped` + safety-stop
survival remain. The evaluator's retry bug (re-creating an existing episode directory) was fixed on the way.

| | seed 98000 | seed 98100 | total | explore tasks | "no reachable frontier" |
|---|---|---|---|---|---|
| + safety-stop survival | 31/40 | 30/40 | 61/80 | 29/48 | 11 |
| + faithful map | 34/40 | 30/40 | 64/80 (80%) | 32/48 (67%) | 0 |

Remaining 16 failures: 6 wrong action type by the model, 4 wrong physical outcome, 2 iteration/time budget, 2 between-task reset failures,
2 other. Mean explore time stays ~130 s. Run-to-run spread is about +/-4 per 40, so 64 vs 61 is within noise on its own; the removal of the
"no reachable frontier" class (11 -> 0) is the mechanism-level evidence. Geometry near the doorway behind the three-ball start row is
tight; some runs still hit safety stops there.

## lang-v2: corrected-data retrain on top of the faithful map

Expert-corrected aggregation with `lang-v1` as actor (world scenes seeds 69000-69007, colour tasks 69500-69505), 300 steps from `lang-v1`
(`data/lang-v2`, 3074 training rows). Same 80 held-out tasks and same code as the 64/80 run.

| | seed 98000 | seed 98100 | total | explore tasks | wrong action type |
|---|---|---|---|---|---|
| lang-v1 | 34/40 | 30/40 | 64/80 | 32/48 | 6 |
| lang-v2 | 29/40 | 34/40 | 63/80 | 32/48 | 1 |

The retrain did what it targeted (wrong-action-type failures 6 -> 1) but the total did not move: the freed tasks now fail on
"iteration/time budget used" (2 -> 7) and "no confidently visible ball" (1 -> 3), i.e. exploration that does not finish inside 12 iterations /
300 s. Seed 98000 was re-run once after a simulator start failure (partial 10/10 discarded).
The next bottleneck is exploration efficiency, not the language model.
