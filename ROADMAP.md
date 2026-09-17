# SpinCore Roadmap — active state 2026-09-17

This file tracks the active legacy-first functional training path. Historical snapshots remain preserved in Git history and validation/docs; they do not override the current plan.

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**: 31 root workers, 8 parent Torch threads, vectorized batching.
- LT2 Stage A — **PASS**: iteration 3000 / 1.8M roots.
- Concurrent-fit production parity — **PASS**: exact semantic parity.
- LT2 Stage B — **PASS**: iteration 7500 / 4.5M roots.
- LT2 Stage B resource/postvalidation — **PASS**: zero swap; finalized checkpoint valid; all four 2M reservoirs saturated/replacement.
- Stage A -> Stage B paired weak-baseline review — **COMPLETE: NO DETECTABLE IMPROVEMENT OR REGRESSION**.
- Stage A -> Stage B policy-drift review — **COMPLETE: MATERIAL POLICY MOVEMENT, ESPECIALLY HU POST-STREET-0**.
- Stage A -> Stage B contemporary checkpoint cross-play — **NEXT**.
- DeepCrusher faithful oracle — **BUILD IN PARALLEL**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_LEARNING_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`

## Completed milestones

### LT0 / LT1 / Stage A

LT0 proved the repaired pipeline. LT1 established the production-shaped large-reservoir line. Stage A reached 1.8M roots; 3H AveragePolicy had crossed 2M while HU remained 820,667. Weak-baseline learning was positive overall/3H and noisy/flat HU.

### LT2 Stage B

4.5M roots / iteration 7500. Preserve:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`

SHA256:

`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`

Final policy seen counts:

- 3H 5,549,800;
- HU 2,072,704.

All four 2M memories are in replacement regime. Resource gate passed with 0 swap and min observed WSL MemAvailable 7.473 GiB.

## Weak-baseline checkpoint delta

Stage A and Stage B were evaluated on the same 1000 empirical scenarios/deals against uniform-legal, passive-caller and jammer families. All nine Stage-B-minus-Stage-A 95% CIs included zero.

Conclusion: no statistically distinguishable gain or regression under the weak fixed-opponent sentinel. This is not an exploitability or GTO result.

## Policy-drift result

The two finalized policies were then queried on 11,040 identical states from 3000 checkpoint-independent probe scenarios.

Overall movement:

- mean TV 0.041395;
- median TV 0.032119;
- p95 TV 0.108574;
- 26.77% of decisions TV >= 0.05;
- 12.10% argmax disagreement.

HU has a heavier tail than 3H: HU p95 TV 0.147864 and 12.34% of HU decisions exceed TV 0.10.

The movement is strongly concentrated after street 0. HU street 1/2/3 mean TV is approximately 0.095 / 0.115 / 0.122, with argmax disagreement approximately 31% / 34% / 25%.

Therefore the weak-baseline flatness cannot be treated as policy stagnation. The AveragePolicy moved materially while the weak opponents did not resolve an EV difference.

## Immediate decision gate — contemporary checkpoint cross-play

Do **not** auto-extend beyond iteration 7500.

Run `tools/run_lt2_checkpoint_crossplay.sh`.

The primary comparison holds the opponent environment fixed to the exact same deterministic 50/50 Stage-A/Stage-B mixture and changes only the hero checkpoint. It reports paired Stage-B-minus-Stage-A chip EV for ALL, 3H and HU.

Additional diagnostics:

- direct seat-balanced HU Stage B vs Stage A;
- 3H invasion: B singleton vs A/A and A singleton vs B/B.

Interpretation:

- coherent Stage-B advantage -> weak baselines were insensitive; current training line remains plausibly productive and can be considered for another bounded block after DeepCrusher/cross-play review;
- no relative advantage despite material drift -> evidence of strategic cycling/neutral movement under the current regime; investigate training dynamics before more roots;
- mixed/borderline result -> increase only the targeted read-only evaluation sample, not training compute.

Cross-play remains relative evidence only. Faithful DeepCrusher R8 v22 is still required for canonical strength claims.

## Product strength path

Future product evidence must include faithful DeepCrusher paired chip-EV, HU/3H separately, then blind/stack/position breakdowns and later full-tournament performance.

## Immediate action

```bash
bash tools/run_lt2_checkpoint_crossplay.sh
```

Wait for `LT2_CHECKPOINT_CROSSPLAY_PASS`, send `SpinCore_LT2A_to_LT2B_crossplay.json`, and do not start further training first.
