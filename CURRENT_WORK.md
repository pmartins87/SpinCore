# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE CONFIRMED — STORED-TARGET FIT BURDEN LARGE — HELD-OUT BUDGET SWEEP NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_TRAINING_DYNAMICS_FIT_RESULT_20260917.md`
- `docs/LT2_TRAINING_DYNAMICS_FIT_AUDIT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500 until the held-out fit-budget sweep is reviewed.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Established strength result

The statistically powered 30k weak-baseline gate met its predeclared precision target. Stage B is clearly positive versus uniform legal in 3H/HU and versus passive caller in 3H, unresolved versus passive HU and jammer 3H, and negative versus jammer HU:

- HU Jammer Stage B raw chip EV: `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta: `-1.682`; simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Thus the extra Stage-A -> Stage-B continuation measurably worsened this specific weak-opponent HU cell. More roots are not admitted blindly.

## Fit audit result

The read-only 25k-per-memory audit completed successfully.

Weighted Stage-B stored-target metrics:

- 3H Advantage: fit fraction vs zero `9.98%`, induced-policy TV `0.5969`, argmax agreement `33.56%`;
- HU Advantage: fit fraction vs zero `14.17%`, induced-policy TV `0.6058`, argmax agreement `26.40%`;
- 3H AveragePolicy: uniform-to-target gap closed `12.98%`, KL `0.6323`, TV `0.4286`, argmax `51.46%`;
- HU AveragePolicy: uniform-to-target gap closed `16.27%`, KL `0.6387`, TV `0.4318`, argmax `49.20%`.

Stage A is broadly similar. Stage-B HU AveragePolicy is modestly worse than Stage A on several fit metrics, but there is no isolated catastrophic Stage-B fit transition.

Important: these raw residuals do **not** prove optimizer underfitting. Advantage targets contain external-sampling variance and AveragePolicy contains historical targets from many iterations. Some residual error is irreducible. The correct next test is therefore a fixed held-out optimizer-budget curve.

## Mechanics clarified

Advantage is reset every iteration. The active production model then gets only `100 x 1024 = 102,400` sample draws before it generates that iteration's behavior policy. With a 2M reservoir, a simple independent-draw coverage approximation is about 99.8k unique items, roughly 5.0% of the reservoir. This is a scale diagnostic, not a threshold.

AveragePolicy receives 4000 steps **per milestone finalization**, not 4000 lifetime total. Counters show 8000 cumulative policy steps at Stage A and 12000 at Stage B.

## Immediate causal experiment

Run:

```bash
bash tools/run_lt2_stage_b_fit_budget_sweep.sh
```

It uses Stage B only, no roots, and never overwrites the source checkpoint. A deterministic 25k holdout is excluded from optimization sampling.

Advantage budgets: `0,25,50,100,200,400,800,1600` cumulative steps from a fresh deterministic reset. The canonical 100-step point is included.

AveragePolicy budgets: `0,1000,2000,4000,8000` additional steps from the stored Stage-B policy+optimizer. The canonical +4000 finalization increment is included.

There is no arbitrary PASS score. We will inspect the held-out learning curves: if error keeps falling materially beyond production budgets, optimizer budget is implicated; if it plateaus while fit remains poor, capacity/representation/target variance becomes the next suspect.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_stage_b_fit_budget_sweep.sh`. Wait for `LT2_STAGE_B_FIT_BUDGET_SWEEP_PASS`, then send `SpinCore_LT2_stage_b_fit_budget_sweep.json`. Do not resume root training first.