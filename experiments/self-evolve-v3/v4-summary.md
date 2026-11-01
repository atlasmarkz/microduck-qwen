# self-evolve-v4 (3 rounds, 36-task held-out, hindsight terminal correction)

Changes vs v3: held-out 12 layouts x 3 tasks = 36; hindsight relabel of a wrong closing done/stop; task-kind
balance; self rows repeated x3; extra attempts on explore tasks; accept tolerance 5 points.

| round | posed | verified (hindsight) | rows after balance | held-out | decision |
|---|---|---|---|---|---|
| start | | | | 29/36 (81%, 65-90%) | |
| 0 | 12 | 7 (4) | 21 | 25/36 (69%) | rejected |
| 1 | 11 | 8 (1) | 21 | 28/36 (78%) | accepted (within slack) |
| 2 | 11 | 8 (2) | 18 | 31/36 (86%, 71-94%) | accepted |

Final adapter: `adapters/self-v4-r2`. 29/36 -> 31/36 is +2 tasks with overlapping 95% intervals, so this is
not evidence of improvement; the noise floor on 36 tasks was not re-measured. Hindsight rows were a minority
(7 of 28 tasks that produced rows). Round 1 was accepted at 28 vs 29 only because of the 5-point slack.
