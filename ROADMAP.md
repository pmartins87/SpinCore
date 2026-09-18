# SpinCore Roadmap — active state 2026-09-18

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Weak-baseline regression — **CONFIRMED**.
- AveragePolicy extra-fit — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED**.
- HU-preflop target variance — **HIDDEN/CHANCE DOMINANT**.
- K4 board averaging — **VALID ESTIMATOR IMPROVEMENT**.
- Jammer K4 causal gate — **NOT MET**.
- Cross-street future-chance audit — **PASS; FUTURE-CHANCE NOT FAILURE-SPECIFIC**.
- Stage-A/B target-drift matrix — **PASS; NO UNIVERSAL TARGET-DRIFT EXPLANATION**.
- HU policy-chain audit — **PASS; MULTI-MECHANISM**.
- Jammer current-behavior first divergence — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Policy-chain verdict

### Jammer

Deployed AveragePolicy:
- B-A `-1.682`, CI95 `[-2.767,-0.597]`.

Current Advantage-induced behavior:
- B-A `-8.430`, CI95 `[-12.515,-4.345]`.

Aggregation-chain delta:
- `+6.748`, CI95 `[+2.521,+10.975]`.

The strong defect is already upstream in current behavior. AveragePolicy buffers it.

Together with stationary Jammer facing-all-in targets, this shifts the causal search toward model action ranking / regret matching rather than target drift.

### PassiveCaller

AveragePolicy B-A is resolved negative, while current behavior B-A is unresolved positive.

This is a separate aggregation/history candidate, not the immediate primary gate.

### UniformLegal

No resolved A/B mechanism.

## Next gate

Run `tools/run_lt2_hu_behavior_first_divergence.sh`.

Purpose:
localize where the resolved `-8.43` chips/hand Jammer current-behavior loss first appears.

Groups:
1. NO_DIVERGENCE;
2. PREFLOP_ROOT;
3. PREFLOP_FACING_ALL_IN;
4. PREFLOP_OTHER;
5. FLOP;
6. TURN;
7. RIVER.

## Decision after behavior localization

If Jammer loss concentrates in PREFLOP_FACING_ALL_IN:
- audit broad non-selected reference action gaps, model action gaps, RM support and fallback regime there.

If it localizes elsewhere:
- follow that exact state class.

No training before this localization.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_hu_behavior_first_divergence.sh`.

Stop at PASS or first error. Do not train.
