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
