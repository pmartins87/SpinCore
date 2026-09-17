# SpinCore Roadmap — active state 2026-09-17

This file tracks the active legacy-first functional training path. Historical engineering snapshots remain preserved in Git history and validation/docs; they do not override the current training plan.

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**: 31 root workers, 8 parent Torch threads, vectorized batching.
- LT2 Stage A — **PASS**: iteration 3000 / 1.8M roots total.
- LT2 Stage A resource gate — **PASS**: no swap; first AveragePolicy saturation transition reached in 3H.
- Weak-baseline learning curve through Stage A — **POSITIVE OVERALL/3H; HU NOISY/FLAT**.
- Concurrent-fit microbenchmark — **PASS**: 1.312188x fit speedup with exact same-thread fit parity.
- Concurrent-fit full-iteration concept parity — **PASS**.
- Concurrent-fit production-function parity — **PASS**: exact semantic parity.
- LT2 Stage B — **PASS**: iteration 7500 / 4.5M roots total.
- LT2 Stage B resource/postvalidation gate — **PASS**: zero swap, finalized checkpoint valid, all four 2M memories in saturation/replacement regime.
- LT2 Stage B paired learning review — **NEXT**.
- DeepCrusher faithful oracle — **BUILD IN PARALLEL**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`
- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`

## Learning milestones

### LT0 — calibration

Purpose: prove the repaired pipeline trains, saves, resumes and plays complete 3H/HU hands. It is not a final-strength run.

### LT1 — production-shaped milestone

Result: 2000 iterations / 1.2M roots completed and preserved.

### LT2 Stage A — first saturation gate

Result: 3000 iterations / 1.8M roots completed. 3H AveragePolicy crossed 2M while HU remained at 820,667. Resource gate passed with no swap.

Weak-baseline trend through Stage A was positive overall and especially 3H. HU remained flat/noisy and was explicitly deferred until HU policy saturation.

### LT2 Stage B — all-reservoir saturation gate

Result: **PASS** at iteration 7500 / 4.5M roots.

Final checkpoint:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`

SHA256:

`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`

Final roots:

- 3H: 2,452,500;
- HU: 2,047,500;
- total: 4,500,000.

Final AveragePolicy seen counts:

- 3H: 5,549,800;
- HU: 2,072,704.

Thus both policy reservoirs crossed 2M; together with the already-saturated Advantage reservoirs, all four memories are now in replacement regime.

Resource/postvalidation result:

- `LT2_STAGE_B_POSTVALIDATION_PASS`;
- checkpoint 2.623474 GiB;
- finalized save 72.938 s;
- min WSL MemAvailable 7.473 GiB;
- max swap used 0 GiB;
- preserved Stage A source unchanged.

The historical `KeyError: 'roots'` was a post-run launcher validation bug, not a training failure. It is corrected and closed.

## Immediate learning gate

Do **not** auto-extend beyond iteration 7500.

Run `tools/run_lt2_stage_b_learning_review.sh`.

This read-only gate compares Stage A (1.8M) and Stage B (4.5M) on the same 1000 fixed-seed empirical scenarios/deals and weak opponent families. It additionally preserves row-level evidence to compute a direct paired 95% CI for the Stage B-minus-Stage A checkpoint delta.

Primary questions:

- did overall and 3H improvement continue after 1.8M roots?
- did HU improve after its AveragePolicy reservoir finally crossed 2M?
- is any apparent checkpoint change statistically distinguishable from evaluation noise under the fixed weak-baseline diagnostic?

Weak baselines remain regression/learning sentinels only. They do not establish GTO strength.

## Later LT2 / LT3

If Stage B learning evidence remains healthy, continue from the exact Stage B checkpoint through a larger bounded block. Do not choose the next root target until the paired Stage A -> Stage B review is complete.

If HU materially regresses or the learning curve stalls, investigate training dynamics/architecture before committing another multi-million-root block.

## Product strength path

Future product evidence must include:

- faithful DeepCrusher R8 v22 direct paired chip-EV;
- HU and 3H separately;
- stack/blind/position breakdowns;
- later full Spin & Go tournament win rate after continuous tournament progression is frozen.

The DeepCrusher oracle must reproduce the frozen OpenPPL strategy faithfully; do not substitute a simplified imitation for canonical claims.

## Immediate action

```bash
bash tools/run_lt2_stage_b_learning_review.sh
```

Wait for `LT2_STAGE_B_LEARNING_REVIEW_EVAL_PASS`, send the generated Stage A report, Stage B report and checkpoint-delta JSON, and do not start further training first.
