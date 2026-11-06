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
navigator, `escape_cramped` before explore. Diagnostic: `tools/diag_explore.py`.

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

## Navigator pad floor and four clearance tiers (adapter lang-v2, same 80 tasks)

Failure analysis of the 10 budget-exhausted explores showed 7 were the same tight spot hit repeatedly: the navigator guard stops at 1.5 cm
clearance, and paths planned with <5 cm pad (pad floor was 1 cm, ball relief on top) tripped it on ordinary tracking wobble. Pad floor is now
5 cm (3 cm only on the last-resort 0.22 m tier); a 0.26 m tier sits before it.

| | seed 98000 | seed 98100 | total | explore tasks | mean explore time |
|---|---|---|---|---|---|
| lang-v1, faithful map | 34/40 | 30/40 | 64/80 | 32/48 | 130 s |
| lang-v2, faithful map | 29/40 | 34/40 | 63/80 | 32/48 | 129 s |
| lang-v2, pad floor | 37/40 | 32/40 | 69/80 (86%) | 41/48 (85%) | 117 s |

Budget-exhausted failures 7 -> 0. New weakness: task 4 ("move forward 20 cm") fell to 12/16 because after the move the model says `unsupported`
instead of `done` in 4 episodes (lang-v2 had 15/16, lang-v1 16/16); the corrected-data retrain added "unsupported" examples and this is the
likely side effect (not verified). Other failures: 5 wrong-type incl. those 4, 4 "no reachable frontier" (the tight-door layout 98101), 2 physical.
Spread between identical runs is about +/-4 per 40; 69 vs 63-64 is outside that, but it is two seed groups of one world level, simulator only.

## lang-v3: move-only phrasing data on top of lang-v2 (same code, same 80 tasks)

Targeted fix for "after `move`, the model says `unsupported`": two more move-only phrasing variants in the training tasks (移动/挪动 wording; none equals a
valid/test phrase), corrected-data aggregation with `lang-v2` as actor (7 varied scenes, seed offset 8700, tasks 0 1 2 4 5 6 7 8), 300 steps from `lang-v2`.

| | seed 98000 | seed 98100 | total | "往前移动20厘米" | left-ball + return | explore tasks |
|---|---|---|---|---|---|---|
| lang-v2 (pad floor) | 37/40 | 32/40 | 69/80 | 12/16 | 16/16 | 41/48 |
| lang-v3 | 36/40 | 30/40 | 66/80 | **16/16** | 16/16 | 34/48 |

The targeted failure is gone (12/16 -> 16/16), but that task has one held-out phrase and the new training phrasings are close to it, so the gain is
probably optimistic. Explore tasks dropped 41 -> 34: 2 `invalid actor JSON` (format failures, new), 2 wrong action type, 3 "no confidently visible ball",
3 layout 98101 (tight doorway, unchanged), others physical. Total 66 vs 69 is inside the +/-4 per 40 run-to-run spread, but the explore drop (-7 of 48) is
larger than that and the new format failures point at the retrain, not the executor. Decision: lang-v3 is not adopted; lang-v2 stays the default.
Not done: a variant that keeps the explore data share fixed while adding move-only rows, or a lower learning rate / fewer steps.

## Correction: which earlier gains are statistically supported (paired McNemar, same 80 tasks)

Earlier sections compared totals and called 69 vs 63-64 "outside the +/-4 noise". A paired test (`compare_runs`) looks only at the
tasks that one run solved and the other did not, which is where the information is:

| comparison | A | B | only B solved | only A solved | McNemar p | difference (95% bootstrap) |
|---|---|---|---|---|---|---|
| lang-v1 before fixes -> faithful map | 52/80 | 64/80 | 16 | 4 | 0.012 | +15 pts (+4..+26) |
| lang-v1 before fixes -> current (pad floor, lang-v2) | 52/80 | 69/80 | 22 | 5 | 0.002 | +21 pts (+9..+33) |
| faithful map -> lang-v2 retrain | 64/80 | 63/80 | 8 | 9 | 1.0 | -1 pt (-11..+9) |
| faithful map -> + pad floor | 64/80 | 69/80 | 9 | 4 | 0.27 | +6 pts (-3..+15) |
| lang-v2 same code -> + pad floor | 63/80 | 69/80 | 11 | 5 | 0.21 | +8 pts (-3..+18) |
| lang-v2 -> lang-v3 | 69/80 | 66/80 | 7 | 10 | 0.63 | -4 pts (-14..+6) |

The overall climb from 52 to 69 is real, and the faithful map (no free space carved through walls) is the step that carries it. The later
steps (retrain, pad floor) are not separable from luck at 80 tasks; the pad floor may help but this sample cannot show it.

## Recovery for `explore` and `approach` (adapter lang-v2, same 80 tasks, run `recov2`)

Two execution-layer changes taken from the literature (see `docs/related-work.md`):
* `explore`: when no frontier is reachable at any clearance, walk to the reachable cell nearest the closest frontier and look again
  (`mapping.vantage_targets`, `Explorer.recover`, at most 2 per call). Idea: SEER keeps a door "to be confirmed" until visited;
  mode-switching frontier search / AERR-Nav leave a deadlock through a recovery mode.
* `approach`: on a clearance stop, a lost target or a timeout, back off, face where the target was, look again and re-bind *the same ball*
  by position (`skills.rebind_near`), then retry; up to 2 recoveries (RePLan, EMAC+: recover instead of ending the episode; MG-Nav and
  detection-driven goal refinement: re-acquire the target before the last leg).

| | seed 98000 | seed 98100 | total | explore tasks | "往前移动20厘米" |
|---|---|---|---|---|---|
| lang-v2 (before) | 37/40 | 32/40 | 69/80 | 41/48 | 12/16 |
| lang-v2 + recoveries | 39/40 | 30/40 | 69/80 | 40/48 | 13/16 |

Paired against the previous run: 5 tasks only the new run solved, 5 only the old one, p = 1.0. **No measurable change in the total.**
(The first attempt at this evaluation was killed part-way by the computer going to sleep and was rerun from scratch as `recov2`;
the partial run was discarded. A keep-awake request is now made before long evaluations.)

What the mechanisms did:
* `explore` recovery fired in 6 tasks and the ball was found afterwards in 1 (seed 98005, task 201). In the 5 others, including all of layout
  98101, nothing new was revealed: that doorway is passable only with about 5 mm of margin, so the recovery cannot create a path.
* `approach` recovery fired in 3 tasks: it saved 1 (seed 98006, a clearance stop at close range) and failed to re-acquire in 2 (seed 98107,
  ball 3.5-3.8 m away: after backing off and turning, the ball is not confidently detectable at that range). Those two failed the way the
  old run's single "target lost" failure did, so the recovery is useful at close range and does nothing at long range.
* Robot footprint check: the collision geoms reach about 0.10 m from the trunk at rest while the navigator assumes 0.14 m (gait sway and
  tracking error). The tight-doorway failures are therefore a choice of safety margin, not an impossibility; the margin was left unchanged.

Not done (next candidates): gate `_reacquire` by distance and walk toward the ball when it is farther than ~2 m instead of backing off;
explore with an information-gain / utility score instead of nearest frontier (mean explore time is still ~125 s).

## Regression canary: replaying visited states (adapters lang-v1 / lang-v2 / lang-v3)

The 21-row held-out split cannot see the regressions (it holds early single-room scenes; `move->done` has 2 rows and every adapter passed
both). `agent_canary --replay` instead re-decides every step recorded by the `eff` rollouts (226 states: the numbered image the model saw,
its instruction and memory, the teacher's action) with each adapter, counting an action-type match:

| state class | lang-v1 | lang-v2 | lang-v3 |
|---|---|---|---|
| `move->done` | 16/16 | **12/16** | 16/16 |
| `explore->approach` | 44/44 | 44/44 | **41/44** |
| `approach->stop` | 7/12 | 11/12 | 11/12 |
| `start->explore` | 46/48 | 48/48 | 48/48 |
| all 226 | 219 | 221 | 222 |

It flags exactly the two regressions that cost a two-hour rollout each to find: lang-v2's `move->done` (all 4 wrong rows are
`{"action":"unsupported"}`) and lang-v3's `explore->approach` (3 rows with an invalid approach action). It needs about 10 minutes per adapter
and no simulator: `bash scripts/canary_chain.sh replay "--replay runs/lang/eff-98000 runs/lang/eff-98100" ADAPTER_0 ADAPTER_1 ...`.
Limits: states come from lang-v2's own rollouts, so a regression in a state that policy never visits is invisible; type match ignores
wrong parameters.

## Evaluation tooling added

`compare_runs` (paired McNemar + bootstrap), `agent_canary` (above), `tools/failure_taxonomy.py`, `agent_eval --phrase-variant 1..3` (three
more held-out wordings per task, checked by a test to be disjoint from every training and test wording), `scripts/eval_world.sh`.
Two seed groups and one wording of the "move forward" task were the weakest part of the measurement; more groups (98200, 98300) and the
alternative wordings have not been run yet.
