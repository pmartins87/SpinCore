# SpinCore Current Work

Date: 2026-09-18
Status: **LT2 STAGE B PASS — K4 ESTIMATOR BENEFIT CONFIRMED BUT JAMMER CAUSAL GATE NOT MET — CROSS-STREET FUTURE-CHANCE AUDIT NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_JAMMER_COMMON_REFERENCE_V2_1_RESULT_20260918.md`
- `docs/LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_20260918.md`
- `docs/LT2_STAGE_A_B_FIRST_DIVERGENCE_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed deployed-policy regressions

HU Jammer:
- Stage-B-minus-Stage-A `-1.682`, CI `[-2.767,-0.597]`;
- FACING_ALL_IN first-divergence contribution `-1.126`, CI `[-2.049,-0.202]`.

PassiveCaller:
- Stage-B-minus-Stage-A `-1.261`, CI `[-2.377,-0.145]`;
- resolved FLOP contribution `-0.672`, CI `[-1.299,-0.046]`.

UniformLegal:
- overall B-A unresolved;
- resolved TURN subgroup contribution `-0.552`, CI `[-0.970,-0.134]`.

The Stage-B deterioration is multi-mechanism until proved otherwise.

## V2.1 full-JSON review

The common-reference V2.1 passed mechanically and mathematically.

Action-gap invariant:
- max canonical Stage-A/B fixed-deal difference `2.98e-08`.

K4 estimator effect on 24 selected Jammer-facing anchors:
- K4-K1 target MSE `-0.026189`, CI `[-0.030418,-0.021961]`;
- K4-K1 reference-best-action regret `-19.27` chips, CI `[-30.07,-8.47]`.

This estimator improvement is resolved.

But Stage-B model/policy degradation on the selected anchors is not:
- AveragePolicy regret B-A `+0.454`, CI `[-2.447,+3.355]`;
- Advantage regret B-A `+16.734`, CI `[-6.365,+39.833]`.

Therefore the predefined causal gate for training K4 is **not met**.

## Outcome-equivalence correction

V2.1 selected transitions:
- `0->1`: 11;
- `1->0`: 2;
- `1->9`: 7;
- `9->1`: 4.

The 11 CHECK_CALL<->ALL_IN transitions all had identical reference values and zero forensic terminal B-A delta.

After an opponent jam, they are one benchmark class:

`CONTINUE = {CHECK_CALL, ALL_IN}`.

Thus 45.8% of the selected V2.1 anchors were not responsible for Jammer EV loss.

On the 13 outcome-relevant FOLD-vs-CONTINUE anchors:
- K4-K1 MSE remains resolved at approximately `-0.02560`, CI `[-0.03178,-0.01943]`;
- K4-K1 regret approximately `-16.77`, CI `[-34.54,+1.00]` is unresolved;
- Stage-B-minus-Stage-A Advantage regret approximately `+29.17`, CI `[-11.86,+70.20]` is unresolved;
- AveragePolicy regret approximately `+0.71`, CI `[-3.92,+5.33]` is unresolved.

## Strategic decision

Do **not** train K4 now.

Continuing to drill only Jammer would risk exactly the benchmark overfitting we wanted to avoid.

The next question is broader:

**Does future-chance target variance also characterize the resolved PassiveCaller FLOP and UniformLegal TURN regression contexts?**

If yes, prefer a generalized future-chance estimator by street rather than a HU-preflop/Jammer-specific patch.

## Active gate

Run:

```bash
bash tools/run_lt2_cross_street_future_chance.sh
```

The audit uses the already-seen forensic seeds only and compares FAILURE vs CONTROL states in:
- Jammer preflop FOLD-vs-CONTINUE;
- PassiveCaller FLOP;
- UniformLegal TURN.

It holds the actual opponent hand and visible board fixed and resamples only unrevealed future cards.

Reference:
- 8 future boards;
- 4 repeats/board;
- exact1.

Candidate:
- independent 8-board stream;
- exact0;
- K1 vs K4.

Holdout seeds `20261001..20261006` remain untouched.

DeepCrusher remains deferred.

## Immediate user action

Pull `main` and run `bash tools/run_lt2_cross_street_future_chance.sh`.

Wait for `LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_PASS` or the first error.

Then send `SpinCore_LT2_cross_street_future_chance.json`.

Do not start any training.
