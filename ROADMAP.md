# SpinCore Roadmap — active state 2026-09-20

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS / PRESERVED**.
- HU400 structural gate — **PASS**.
- HU400 broad fresh-refit gate — **PASS**.
- HU400 100-iteration online pilot — **PASS**.
- HU400 refresh to 8000 — **MIXED**.
- AveragePolicy at 8000 vs Stage B JAMMER — **RESOLVED IMPROVEMENT**.
- AveragePolicy weak-baseline tradeoff — **NOT DETECTED**.
- current behavior at 8000 vs Stage B JAMMER — **RESOLVED IMPROVEMENT**.
- current behavior at 8000 vs Stage B PASSIVE_CALLER — **RESOLVED REGRESSION**.
- root training beyond 8000 — **PAUSED**.
- holdout — **SEALED**.
- next — **7600 -> 8000 current-behavior first-divergence localization**.

## Why training stops at 8000

The deployed AveragePolicy now moves in the desired direction:

- JAMMER `+1.902`, CI95 entirely positive;
- no resolved PASSIVE_CALLER or UNIFORM_LEGAL regression.

But the training behavior develops a new PASSIVE_CALLER weakness:

- `-2.921`, CI95 `[-5.541,-0.302]`.

Because current behavior drives future trajectory collection, this cannot be ignored before additional roots.

## Next read-only gate

Direct iteration 7600 -> 8000 first-divergence attribution across the already-used forensic HU population.

Primary:
- PASSIVE_CALLER.

Classify the first sampled action divergence into:
- PREFLOP_ROOT;
- PREFLOP_FACING_ALL_IN;
- PREFLOP_OTHER;
- FLOP;
- TURN;
- RIVER.

No roots. No optimizer steps. No holdout.

## Immediate action

Run `bash tools/run_lt2_hu_7600_8000_behavior_first_divergence.sh`.

Do not continue training.
