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

## Expanded evaluation: new layouts and a clean recovery ablation (adapter lang-v2)

New layouts never used for any earlier decision: seeds 98200 and 98300, 8 layouts each, same 5 task types per layout. The harness
produced 40 tasks for group 98200 and 37 for group 98300 (77 per arm; I did not investigate why 98300 has 3 fewer).
The only difference between the two arms is the environment variable `MICRODUCK_RECOVERY` (0 = explore and approach recoveries off,
1 = on), so this is a clean switch, unlike the earlier before/after comparison.

**Tasks the harness cannot label are excluded.** The "leftmost ball" teacher cannot label a task when the layout shows several yellow
balls or cannot see the leftmost one confidently (`teacher leftmost target is ambiguous` / `not confidently visible`). Those are not model
outcomes: 4 of the 77 tasks per arm (layouts 98200, 98205, 98302 and 98307, task 3 in each), identical in both arms, are excluded,
leaving 73. Raw report totals (off 32/40 and 28/37, on 33/38 and 28/37) were written by runs that did not all apply this rule and are
not used below.

| arm | valid tasks | correct | explore | "move forward 20 cm" | left ball + return |
|---|---|---|---|---|---|
| recovery off | 73 | 60 (82%) | 36/45 | 12/16 | 12/12 |
| recovery on  | 73 | 61 (84%) | 36/45 | 13/16 | 12/12 |

Paired test on the same 73 tasks: 7 solved only with recovery, 6 only without, McNemar p = 1.0, difference +1 point
(95% bootstrap -8 to +11). **No measurable effect of the recoveries on the total.**

What actually happened:
* Recovery fired in 4 tasks (approach) and 1 (explore). Approach recovery fired in seed 98201 tasks 200, 201, 202 and in 98303 task 200;
  the approach then succeeded in all four. In 98201 the recovery-off arm failed all three of those tasks, so these are real saves, but
  all three are one layout (one geometry, not three independent samples).
* The other 10 tasks that differ between the arms (98204 x3, 98207, 98203, 98205, 98303, 98306, 98307 x2) had no recovery firing
  in either arm. They are run-to-run noise from identical code: about 10 of 73 tasks flip between two runs, which matches the earlier
  +/-4 per 40 estimate and explains why a +1 difference cannot be separated from zero.
* Failures with recovery on are still mostly `explore` ending without a confirmed ball (outcome check failed) and "target lost during
  approach and not re-acquired" (2). The long-range target-loss case is not helped by the current recovery (see the previous section).

**Held-out wording, same seed (98200, non-explore tasks, invalid tasks excluded):** the original wording 12/14 (both arms); three alternative
held-out phrasings 13/14, 13/14, 14/14. The model is not brittle to these three rewordings (n = 14 each, so this cannot detect a gap
smaller than about 20 points).

**Conclusions.** (1) 82-84% on new layouts is close to the earlier 86% (69/80), so there is no sign of a large drop on layouts the fixes
were not designed against (the intervals overlap widely; this does not prove there was no tuning to the old layouts). (2) The recoveries are cheap safety nets but their effect on the total is below what 73 tasks can resolve;
the only evidence for them is the 3 same-layout saves. (3) The remaining explore failures are not addressed by retrying;
the next candidates are still the safety-margin tier for knife-edge doorways and distance-gated re-acquisition.

## Anatomy of the 12 failures on the new layouts (recovery on, 73 valid tasks)

Classified from each episode's step log (`tools/failure_taxonomy.py`, which now has a `done` for `stop` category):

| category | tasks | what happened |
|---|---|---|
| `move` then `unsupported` | 3 | "move forward 20 cm": the move succeeds, then the model emits `unsupported` instead of `done` |
| `done` where the teacher expects `stop` | 3 | "approach, then stop": explore and approach both succeed, then the model says `done`; the strict score expects `stop` |
| approach failed | 3 | target lost during approach and not re-acquired (2), target no longer safely visible (1) |
| explore ended, ball not confidently visible | 2 | explore reports the colour found, but no fully visible ball is available for approach |
| `clarify` on an English instruction | 1 | |

The second row was filed under "physical outcome wrong" before: the evaluator's error string ("failed physical skill or wrong task action")
does not distinguish the two, and the motion was fine in all three. `stop` and `done` both end the episode with the robot already standing
still after a successful approach, so the physical outcome is identical; only the labelled skill differs. The model emits `done` for `stop`
in 3 of 11 such tasks with recovery on, 1 of 12 with recovery off and 0 of 12 in the earlier `recov2` run, so this is mostly sampling
noise around a weak preference, not a systematic fault.

Two ways to score the same 73 tasks:
* strict (the exact labelled skill): 61/73 (84%), the number used everywhere above;
* outcome-equivalent (`done` accepted for `stop` after a verified approach): 64/73 (88%).
The strict number stays the headline; the second only shows how much of the gap is skill-label strictness rather than robot behaviour.

## Narrowest clearance tier (0.20 m), evaluated on the same new layouts

Motivation: on layout 98101 the knife-edge doorway left no reachable frontier at any clearance down to 0.22 m. Replaying the saved map
offline found a path at 0.20 m and below. The robot's collision geoms reach only ~0.10 m from the trunk at rest while the navigator assumes
0.14 m plus a 5 cm map cell, so 0.20 m still leaves a margin over the geometry. New last tier in `explore.clearance_tiers`: inflation 0.20 m
with a 1 cm navigator pad, tried only when every earlier tier found nothing (`MICRODUCK_NARROW=0` removes it).

Single-layout check (98101, hidden green ball, one run each, a layout this change was designed on): without the tier "no reachable frontier
left" after 3 steps (3353 cells); with it the robot crossed the doorway and found the ball (5141 cells, 182 s).

Evaluation on seeds 98200 and 98300 (73 valid tasks, recovery on, layouts not used to design the tier), against the recovery-on run above:

| arm | correct | explore | "move forward 20 cm" | left ball + return |
|---|---|---|---|---|
| without the tier | 61/73 | 36/45 | 13/16 | 12/12 |
| with the tier    | 64/73 | 36/45 | 16/16 | 12/12 |

Paired: 5 tasks only with the tier, 2 only without, McNemar p = 0.45. Three of the five gains are "move forward 20 cm" (the model's `move` then
`unsupported` slip, which depends on sampling, not on exploration). **The explore family is identical, 36/45, and no explore call in either arm
ended with "no reachable frontier left"**, so these layouts never exercise the tier. The +3 is run-to-run noise; the evaluation shows
that the tier does no harm to explore (36/45 both), not that it helps. Its only positive evidence is the single 98101 check above.
Walks that failed mid-route: 4 with the tier, 2 without (small counts, not interpreted).

Next: evaluate on layouts with knife-edge doorways, which means finding such layouts first (the generator does not label them).

## Settle on the target before explore returns (diagnostic, layouts 98205 and 98307)

Two evaluated tasks ended with "explore found the colour, then no fully visible ball to approach". The post-explore frames showed why: the
ball sat on the image border (x 619/640 in one, not detected at all in the other), so the next frame did not show it fully visible.
New step in `explore` (`MICRODUCK_SETTLE`, now default on): face the nearest ball, take a fresh frame, and require it fully visible with at
least 50 px to both sides; if it cannot be confirmed the position becomes a suspect and the search continues.

Same layout, same hidden colour (blue), the frame the next skill would see right after explore (one run each; the tasks were chosen
because they failed, so this shows the mechanism, not a rate):

| layout | without the step | with the step |
|---|---|---|
| 98205 | ball box x 578..640 (cut by the right edge) | x 502..581 (clear, fully visible) |
| 98307 | ball box x 0..15 (on the left edge) | x 368..395 (near the centre, fully visible) |

Not yet measured: effect on the full evaluation (regressions, time added by the turns). That run is next.

## Settle step, full evaluation (seeds 98200 and 98300, recovery and the narrow tier on)

| arm | correct | explore | "move forward 20 cm" | left ball + return |
|---|---|---|---|---|
| narrow tier, no settle | 64/73 | 36/45 | 16/16 | 12/12 |
| + settle (run `ext-settle`) | 64/73 | 38/45 | 14/16 | 12/12 |

Paired: 5 tasks only with settle, 5 only without, McNemar p = 1.0. The total did not move; the pieces did:
* Gains in explore: the two tasks the step was built for (98205 and 98307, task 201) plus three more (98201/202, 98302/201, 98305/202), which
  may be the same edge-of-view effect or noise.
* Explore losses, three: 98200/200 (the model wrote `stop` where `done` was expected, label noise), and 98206/201 and 98300/201, which were a
  real bug in the step: its pre-turn `healthy()` check raised the body's "clearance too small" safety stop and that ended the whole explore
  although the ball was already found.
* "Move forward 20 cm" lost two tasks to the model's `move` then `unsupported` slip (sampling noise, unrelated to explore).

Fix: `Explorer.settle` now returns the unsettled sighting on a body safety stop and re-raises only cancel, fall and stale-physics.
Re-running just the two affected layouts with the fix (`scripts/eval_layouts.sh`, runs `fixchk-98206` and `fixchk-98300`, one run each):
98206 task 201 and 98300 task 201 both pass now; 98206 task 200 fails for an unrelated reason (the model emitted `stop` where `return_home`
was expected). These are single runs of tasks chosen because they failed: they show the bug is gone, not a rate. The corrected code has
not had a full 73-task evaluation.

## Executor feedback after move/turn (the `move` then `unsupported` slip)

Cause found in `agent_prompt`: approach, explore, return_home and bypass_return each end with an "executor feedback" line telling the model
what was verified and what to do next; move and turn did not, so after a completed "move forward 20 cm" the model saw nothing saying the
instruction was satisfied and sometimes wrote `unsupported`. Added a line for move/turn (output `done` when nothing else is left; use
`unsupported` only if an unimplemented skill such as kicking remains). No retraining; the same adapter (lang-v2).

* Replay canary (638 recorded states): `move->done` 41/48 -> 48/48; every other state class identical.
* Simulator, task "move forward 20 cm" only, 32 layouts never used before (98400..98731), same adapter, flag off vs on:
  26/32 -> 32/32, 6 tasks only with the feedback, 0 only without, McNemar p = 0.031, +19 points (95% bootstrap +6..+34).
  All 6 failures without it were `move` then `unsupported`; none remain with it.

Tried and rejected, same offline canary: a longer feedback after approach that spells out when to write `stop`, `return_home` or `done`.
It fixed `approach->stop` (28 -> 34 of 34) but pushed 20 states that should go home to `stop` (`approach->return_home` 74 -> 54), net
626 -> 608. A frozen LoRA adapter is not robust to a reworded prompt it never saw; the `stop`/`done` confusion needs training data, not
prompt text. Code removed.

## Correction: suspect walks never happened; and the far re-acquisition that depends on the same bound

Found while checking why a new re-acquisition step did nothing: the navigation contract caps a point goal's stop distance at 0.6 m
(`contract._number(..., 0.1, 0.6)`), but `Explorer.check_suspect` asked for 0.9 m and the new re-acquisition for 1.4 m. The plan was rejected,
the `ValueError` was caught as "not reached", and nothing moved. Over every recorded run, **25 of 25 suspect walks failed on this bound**, so the
"revisit partly seen balls of the target colour (ApexNav)" claimed earlier for `explore` was never actually executed; it only looked at the
view it already had. Both distances are now 0.6 m (`SUSPECT_STOP_M`, `FAR_REACQUIRE_STOP_M`), a test builds every internal walking plan against the
contract, and a failed far walk is recorded in the recovery list instead of being swallowed. Every number measured before this fix was produced
without working suspect walks, including the settle step's fallback (an unconfirmed ball becomes a suspect, which was then not visited).

Far re-acquisition (when the target is lost while farther than 1.8 m from where it was, walk along the planned route to 0.6 m of that
spot before looking again, instead of backing off blind): A/B on the two layouts with the hardest approaches (98302, where a ball 3 m away
slips behind a wall edge, and 98204), explore tasks only, two repeats per arm, flag off vs on, 12 task pairs:
4/12 -> 8/12 correct; 4 tasks only with it, 0 only without, exact McNemar p = 0.125 (too few pairs to be significant). The walk started in 3 of the
12 runs with it on, and the approach succeeded after it every time (one of the three is still scored wrong by the `stop`/`done` label slip).
Read as: the mechanism works when it fires, the layouts are noisy, and the size of the effect on a full evaluation is still unmeasured.

## Candidate evaluation on fresh seeds 99000 / 99100 (lang-v2 vs lang-v4, all execution fixes)

- lang-v2: 61/74 valid (82.4%); lang-v4: 60/74 (81.1%). Paired: only-v2 5, only-v4 4, McNemar p = 1.0, difference -1.4 points (95% CI -9.5..+6.8). No evidence that v4 helps; keep lang-v2.
- Fresh seeds are lower than the dev layouts (84-88%), so the dev numbers were optimistic.
- Failure anatomy (v2): 4 `done` where `stop` expected, 4 explore budget used, 2 clarify misfires, 1 approach target not safely visible.
- Merged dev-explore into main (retry cap per frontier, hardened ball relief, start-of-approach re-acquire); 172 tests pass. Not yet re-evaluated.
