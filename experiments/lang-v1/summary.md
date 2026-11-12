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

## Merged execution fixes on the same fresh layouts (lang-v2, seeds 99000 / 99100, run `merged-v2`)

Harden balls after a clearance stop, cap retries per frontier, re-acquire at the start of approach (commit ad26f71), against `cand-v2` on the same 74 valid tasks.

* 61/74 vs 61/74; 4 tasks only with the fixes, 4 only without; McNemar p = 1.0; difference 0.0 (95% bootstrap -8.1..+8.1). **No measurable effect on the total.**
* Failure mix is nearly unchanged: 4 `done` for `stop`, 4 explore budget used, 1 explore clearance stop, 1 wrong action type. One failing episode recorded no steps at all (`infra: reset`, a harness failure that is scored as a miss); `tools/failure_taxonomy.py` crashed on it and now reports it.
* The fixes were built for specific failures (the 99005 explore loop, approach not safely visible at the start). Whether those particular tasks changed was not inspected here; the paired total says only that gains and losses cancel.

## Action legality check (harness, no retraining; `MICRODUCK_LEGALITY=1`, default off)

`microduck_qwen/legality.py` rejects a decoded skill that contradicts the skill contract in the current state, then lets the
model choose again among legal types: every type is scored by the model's log-probability of the prefix `{"action":"<type>`,
and the output is regenerated greedily with the best legal prefix forced. Rules come from `agent_prompt` and the state classes
of the training data, not from task labels: no `done` before a verified skill; no `return_home` without a home; with a home,
no `clarify` after a verified skill, and none at the start when the instruction names a target colour (every `clarify` row in
the training data is a target-less instruction such as "去那边"); no repeat of the verified explore/approach of the same colour
or of a return. Not rules: `stop` vs `done` (both legal terminals) and `unsupported` after move/turn (legitimate when an
unimplemented skill remains).

Rule check before any model run: 0 of the teacher labels in every dataset under `data/` and 0 of the 3908 recorded teacher
labels in `runs/lang` are flagged. Of the 3908 recorded model decisions (all adapters), exactly 24 are flagged: 22 `clarify` at
the start of an explore task and 2 unparseable outputs; the other recorded errors (63 `stop`/`done` swaps after approach,
24 `unsupported` after move) are not contract violations and are untouched.

Offline replay, `lang-v2`, all 3908 recorded states of `runs/lang` (one run with the check on; a legal decision passes through
unchanged and every intervention keeps the original output, so the same run gives both arms; `tools/legality_report.py`):

| replay set | off | on | only on | only off | interventions |
|---|---|---|---|---|---|
| canary dev set (6 runs, 638 states) | 626 | 629 | 3 | 0 | 3, all fixed |
| fresh seeds 99000/99100 (cand-v2, cand-v4, merged-v2; 597 states) | 581 | 583 | 2 | 0 | 2, all fixed |
| all of `runs/lang` | 3857 | 3863 | 6 | 0 | 8: 6 fixed, 2 still wrong |

* The off arm reproduces the earlier canary exactly (626/638), so the replay is deterministic.
* Every intervention is `start->explore` where `lang-v2` wrote `clarify`; the class goes 828/836 -> 834/836. No other class changes; nothing correct is broken.
* The 2 still wrong are the same task (layout 98302, task 202, "please the blue ball is in another room, go next to it, then return to your initial position") replayed from two runs: the best legal type was `return_home` (-3.5 and -4.4) just ahead of `explore` (-4.4 and -4.8). The same task from a third run was fixed.
* The prefix score ranks the model's own illegal choice first in 8/8 interventions, so the scoring is consistent with what greedy decoding produced.
* Cost: 0 on legal decisions; an intervention takes 8-13 s (11 scoring passes plus one generation).
* The 8 interventions come from 5 distinct layout/task pairs (98302/202 appears four times), so this is not 8 independent cases.

Reading: a guard with no false positives on everything recorded, fixing the clarify misfire class offline. It addresses at most
the 2 clarify misfires seen in the fresh-seed failures (of 13), so the expected end-to-end effect is about 1-3 tasks out of 74, inside run-to-run
noise; not measured in the simulator. Not done: the obvious next rule ("`return_home` at the start, while the robot is still at home, is
pointless") would fix the remaining 2, but it was found on these same states, so it needs states never used for the decision.

### Legality check in the simulator (fresh seeds 99000 / 99100, run `legal-v2`, paired with `merged-v2`)

Same adapter (lang-v2), executor code and layouts as `merged-v2`; only `MICRODUCK_LEGALITY=1` differs (recorded as `"legality": true`
in both run reports; `scripts/pipelines/eval-legality.sh`).

* 61/74 vs 62/74; 3 tasks only without, 4 only with; McNemar p = 1.0; difference +1.4 points (95% -5.4..+8.1).
* **The check never fired**: 0 interventions in all 74 episodes. lang-v2 made no illegal decision on these layouts, so both arms
  ran the same decision path and the 7 flips are run-to-run noise (physical execution: 2 "no confidently visible ball to approach",
  4 failed physical skill or wrong action after execution, 1 task-start restore failure). The +1 says nothing about the check.
* Useful by-product: an A/A repeat on the same 74 tasks, 7 flips (9.5%), consistent with the ~10 of 73 seen earlier. One identical
  repeat cannot separate a 1-3 task effect.
* The offline replay had found the check's 2 fresh-seed interventions in `cand-v2-99000`, made by the same adapter on the same
  layouts; in this run those states did not recur (the explore start frames differ between runs).

Reading: the legality check is a zero-cost guard on these layouts (it adds latency only when it fires, and it did not), with an offline
gain on recorded misfires and no measurable end-to-end effect. Keep it available behind the flag; whether to turn it on by default
is a policy choice, not a result.

## Simulator-truth outcome scorer (`python -m microduck_qwen.truth_score RUN ... [--b RUN ...] [--details]`)

Scores each task by what physically happened, without the teacher's per-step labels: the robot's oracle pose before every decision,
the true ball positions, the first frame's segmentation (for "leftmost"), and scene contacts. The task definition (`teacher_specs`) says
only what is required (colour or leftmost ball, move, return). Success: the model ended the task itself with `done` or `stop`; the
required ball was reached within the teacher's own resting bound (0.375 m); a required return ended within 0.175 m of home (return_home's
0.10 m goal + 0.025 arrival + 0.05 coast) after the reach; a required move is within the executor's 0.05 m / 0.08 m lateral tolerance;
no scene contact. Episodes the strict evaluation cut before the outcome was visible are reported separately: `cut_teacher` (the teacher
could not label a later step), `cut_deviation` (a different non-terminal skill, after which the strict run stops) and `cut_infra`; the
headline counts them as failures and an upper bound counts them as successes. Same valid-task filter as `compare_runs`.

Validation on every run with a report in `runs/lang` (1588 valid tasks, 25 run groups):
* **0 of the 1263 strict passes fail the truth check**, so the scorer agrees with the existing physical verification wherever the
  teacher agreed.
* 69 strict failures are truth successes, every one a terminal label swap with the robot where it should be (49 ended with `done`,
  20 with `stop`). The rescored totals are 1332 vs 1263.
* No paired conclusion changes:

| comparison | strict | truth |
|---|---|---|
| cand-v2 vs cand-v4 (fresh seeds) | 61 vs 60, p = 1.0 | 66 vs 65, p = 1.0 |
| cand-v2 vs merged-v2 | 61 vs 61, p = 1.0 | 66 vs 66, p = 1.0 |
| merged-v2 vs legal-v2 | 61 vs 62, p = 1.0 | 66 vs 68, p = 0.73 (5 vs 3 discordant; the check never fired) |
| move feedback off vs on (32 layouts) | 26 vs 32, p = 0.031 | 26 vs 32, p = 0.031 |
| new vs fix | 52 vs 52, p = 1.0 | 54 vs 56, p = 0.79 |
| base vs eff | 42 vs 69, p < 0.001, +34 points | 52 vs 70, p = 0.001, +22.5 points |

Readings:
* The largest historical gain survives but is a third smaller: part of base -> eff was the model learning the teacher's `stop`/`done`
  convention, not better physical outcomes.
* On the fresh seeds the model's remaining failures are execution, not decisions. In merged-v2 all 7 truth failures had the expected
  skill chosen (5 explore budget or clearance stops, 1 approach target not bound, 1 safety stop after explore); legal-v2's 4 are
  explore budget. Layout 99104 fails all three explore tasks in both runs.
* `merged-v2` and `legal-v2` ran the same decision path, so their 8 truth flips are an A/A noise estimate (8 of 74 = 11%).
* Two `legal-v2` tasks are `cut_teacher`: explore found the ball, then the teacher could not label the approach ("no confidently visible
  blue ball"), and the strict score counted the labeller's refusal as a model failure.
* Not covered: what a deviating model would have done after the strict run stopped it (`cut_deviation`; none in these runs, the
  clarify misfires end in `clarify`, which is a truth failure). Measuring that needs a free-running evaluation mode that keeps executing
  the model's own actions after a deviation.

## Perceived-map escape for the explore stuck loop (`MICRODUCK_MAP_ESCAPE=1`, run `mapesc-v2`)

Cause: the navigator's guard checks clearance against the perceived map (5 cm cells, so a door jamb is drawn about 10 cm too close)
and stops below 0.015 m, while `escape_cramped` measured the true scene and only acted below 0.05 m. At the recorded stuck points the
true clearance was 0.09-0.11 m and the perceived one 0.015-0.016 m, so the robot never backed away and every later walk failed at
the guard from the same spot. Fix (flag, off by default): after a failed walk and before each explore step or recovery, also run
`escape_cramped` against the perceived map; it stops early if backing away would bring the true clearance under the guard threshold.
Two new unit tests cover the perceived-only obstacle and the real obstacle behind the robot.

Paired evaluation, fresh seeds 99000/99100, same code, adapter and seeds as `merged-v2` / `legal-v2`; only the flag differs.

| comparison (74 tasks) | strict | truth |
|---|---|---|
| merged-v2 vs mapesc-v2 | 61 vs 67, 3 vs 9 discordant, p = 0.15 | 66 vs 71, 3 vs 8 discordant, p = 0.23 (95% -0.01..+0.16) |
| legal-v2 vs mapesc-v2 | 62 vs 67, 2 vs 7, p = 0.18 | 68 vs 71, 2 vs 5, p = 0.45 |

Mechanism, over the 44 explore calls of each run:

| run | explore found the ball | stuck (>= 3 failed walks in a row from one spot) | map escape fired |
|---|---|---|---|
| merged-v2 | 39/44 | 4 (0 found) | - |
| legal-v2 | 41/45 | 5 (1 found) | - |
| mapesc-v2 | 43/44 | 0 | 4 calls, all 4 found the ball |

Attribution of the flips:
* The escape fired in 99005 tasks 201/202 and 99104 tasks 200/202, and all four are truth successes. 99104 failed all three explore
  tasks in both earlier runs (0/6) and now passes all three (task 201 without the escape firing, so a different route). 99005 tasks
  201/202 had already passed in `legal-v2` without the escape, so those two flips are not attributable to it.
* None of the three losses touched the escape: 99005/200 and 99006/202 are true-scene clearance safety stops in explore (the error
  that also hit 99103/201 in `merged-v2`), 99007/201 is a model `clarify`.

Reading: the stuck pattern is gone (4-5 calls -> 0) and the one layout that failed every explore task in both earlier runs now passes,
but the overall gain is not significant, it is the size of the A/A noise (8 truth flips between `merged-v2` and `legal-v2`), and the
fix was diagnosed from failures on these same seeds (layouts 99104 and 99005). It is a dev-task mechanism result, not a retained gain.
Confirming it needs layouts never used for any decision. Remaining truth failures: 2 explore safety stops (true clearance), 1 clarify.

### Confirmation on never-used layouts (seeds 99200/99300, runs `mapoff-v2` / `mapesc-v2`)

Both arms run in one pipeline (`scripts/pipelines/eval-map-escape-new.sh`), interleaved per seed group; the flag is the only
difference. lang-v2 training data stops at seed 97202 and no earlier evaluation used these seeds. 80 valid tasks.

| | strict | truth | explore found | stuck explore calls | map escape fired |
|---|---|---|---|---|---|
| off (`mapoff-v2`) | 72/80 | 74/80 (cuts 3) | 45/46 | 0 | - |
| on (`mapesc-v2`) | 72/80 | 74/80 (cuts 2) | 45/45 | 0 | 0 |
| paired | 3 vs 3 discordant, p = 1.0 | 2 vs 2, p = 1.0 (95% -0.05..+0.05) | | | |

Reading: **not confirmed, and not testable on these layouts**. The stuck loop never occurred in either arm (0 of 91 explore calls),
so the escape never fired and the two arms ran the same path; the 4 truth flips are noise (e.g. 99206/201 and 99207/202 swap a
true-clearance safety stop between arms, neither near an escape). The condition it fixes is layout-specific: 4-5 of 44 explore calls
on 99000/99100, concentrated on two layouts, and 0 of 91 on 16 new layouts. It also cost nothing here (74 vs 74). Keeping it behind
the flag; a retained-gain claim needs layouts where the stuck point exists, e.g. a larger sample of new seeds.
Remaining truth failures on the new seeds, both arms: English "please the green ball is in another room" ends with `clarify` (2 per
arm), true-clearance safety stops (1 per arm), and the teacher refuses to label approach after explore found a green ball (2-3 cuts).

## Frozen base with the same harness (run `frozen`, seeds 99200/99300)

Question: how much of the success comes from the LoRA adapter, with the harness held fixed? `agent_eval` without `--adapter`
(frozen Qwen3-VL-4B-Instruct), same arguments, prompt, validator and executor as `mapoff-v2`; map escape and legality off in both.
Pipeline `scripts/pipelines/eval-frozen-base.sh`. Same 80 tasks on the same 16 never-used layouts. This measures the adapter, not
the harness: a harness ablation would need a primitive-only action set (no `explore`/`approach` executor).

| | strict | truth | truth outcomes |
|---|---|---|---|
| frozen base | 19/80 | 27/80 | 27 success, 49 cut at a deviation, 4 fail |
| lang-v2 LoRA (`mapoff-v2`) | 72/80 | 74/80 | 74 success, 3 teacher cut, 3 fail |
| paired | 3 vs 56 discordant, p < 1e-12, +66 pp (95% +54..+78) | 3 vs 50, p < 1e-12, +59 pp (95% +46..+71) | |

By family (strict): explore 18/48 vs 40/48, task 3 1/16 vs 16/16, task 4 0/16 vs 16/16. Output format is never the problem
(`format_ok` holds in all 80 frozen tasks; the JSON contract in the prompt is enough).

Where the frozen base fails (first wrong step, 61 failures):

| previous skill | expected | frozen output | tasks |
|---|---|---|---|
| `return_home` | `done` | `explore` again | 17 |
| `move`/`turn` | `done` | the same `move` again | 16 |
| `return_home` | `done` | `approach` again | 14 |
| `approach` | `done` | `stop` | 6 |
| (first step) | `explore` | `approach` a ball not in view | 2 |
| other (`stop` vs `done`, target, physical) | | | 6 |

53 of the 61 failures are the same thing: every earlier skill was the right one and executed successfully, and the model does not
end the task. It ignores the explicit executor feedback line ("移动或转向已成功完成……其余要求都已满足则输出done", row 12 of
`docs/improvement-evidence.md`) and repeats the last part of the plan. LoRA has 0 such failures.

Reading: with the harness fixed, the adapter adds +59 pp truth on never-used layouts. Almost all of it is one learned behaviour,
recognising that the instruction is finished and emitting `done`; skill choice and target binding are already right in the frozen
model in all but about 8 tasks. If termination were handled (counting the 53 as passes) the frozen base would be about 72/80, the
same as LoRA, but that is a counterfactual: the truth scorer cannot see what the repeated skill would have done to the outcome, so it
is not a measured number. The cheap test is a frozen-base re-run with a stronger termination cue or a rule-based stop, which would
say whether this gap needs training at all.

### Frozen base plus a termination rule (run `frozterm`, seeds 99200/99300)

The rule (`MICRODUCK_TERMINATION=1`, `microduck_qwen/legality.py`, off by default) comes from the lang-v2 training data, not from
the evaluation labels: after a verified `return_home`/`bypass_return` only `done` or `unsupported` is legal (455 of 455 training
states end with `done`), and repeating the identical verified move/turn is illegal (never in the training data). A violating
decision is replaced by the model's own best-scoring legal action, as in the legality filter (row 17); no rule picks the action.
Offline, before the run: it flags 47 of the frozen model's 59 first wrong steps and none of the LoRA decisions or teacher labels on
these seeds. Pipeline `scripts/pipelines/eval-frozen-termination.sh`; same arguments as `frozen`.

| | strict | truth | truth outcomes |
|---|---|---|---|
| frozen base (`frozen`) | 19/80 | 27/80 | 27 success, 49 cut at a deviation, 4 fail |
| frozen base + termination rule (`frozterm`) | 60/80 | 68/80 | 68 success, 6 deviation cuts, 5 teacher cuts, 1 fail |
| lang-v2 LoRA (`mapoff-v2`) | 72/80 | 74/80 | 74 success, 3 teacher cuts, 3 fail |
| rule vs frozen, paired | 43 vs 2, p < 1e-10 | 44 vs 3, p < 1e-10 | |
| LoRA vs rule, paired | 17 vs 5, p = 0.017, +15 pp (95% +4..+25) | 10 vs 4, p = 0.18, +7.5 pp (95% -1..+16) | |

The rule fired 47 times in 220 steps: 43 regenerations chose `done` (27 after a return, 16 after a repeated move/turn), 4 were
unparseable outputs (the filter also handles those; regenerated as `search`/`explore`). About 10 s extra per firing (two extra
forward passes).

Remaining frozen-base errors (first wrong step, 14 tasks): `stop` instead of `done` after an approach (7; the truth scorer counts
these as successes when the robot did stop at the ball, which is why strict and truth differ), "靠近照片里最左侧的球，之后返回起点"
(task 3, 4 layouts) answered with `search`/`explore` instead of `approach` with `selector=leftmost`, `approach` for a ball not in
view where the teacher explores (2), and 1 `done` vs `stop`. Plus 5 teacher cuts (no confidently visible green ball after explore)
and 2 English `clarify`.

Reading: a two-line contract rule closes 41 of the 47 truth points between the frozen base and the adapter (27 -> 68 vs 74); the
remaining truth gap (+7.5 pp) is not significant on 80 tasks. On strict scoring the adapter is still significantly better
(+15 pp), mostly because it learned the teacher's `stop`/`done` wording after an approach and the leftmost-selector task. So most of
the measured adapter gain is a termination behaviour that does not need training. Caveat: the rule was chosen after looking at the
frozen-base failures on these same seeds. It is a behaviour class taken from training data, not tuned to these layouts, but by this
project's own standard it is a dev-seed result; confirming it needs never-used seeds (and a LoRA + rule arm, expected to change
nothing since the rule flagged none of its decisions offline).
