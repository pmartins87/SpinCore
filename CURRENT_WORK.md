# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — ALL FOUR 2M RESERVOIRS SATURATED/REPLACEMENT — LEARNING REVIEW NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`
- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

The active learning line is continuous LT1 -> LT2. Do not restart any completed stage and do not continue beyond iteration 7500 until the Stage A -> Stage B learning review is complete.

## Preserved checkpoints

LT2 Stage A source:

- iteration 3000 / 1.8M roots;
- `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B milestone:

- iteration 7500 / 4.5M roots;
- `/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Both checkpoints are preserved. Stage B is the current continuation state if further training is later authorized.

## LT2 Stage B — PASS

The expensive computation completed successfully and the independent postvalidation also passed:

`LT2_STAGE_B_POSTVALIDATION_PASS`

Validated milestone:

- target iteration 7500;
- 4,500,000 total roots;
- 3H roots 2,452,500;
- HU roots 2,047,500;
- 31 root workers;
- 8 parent Torch threads;
- vectorized batching;
- admitted production `concurrent_fit` path;
- checkpoint every 250 iterations;
- finalized AveragePolicy fit and checkpoint present;
- preserved Stage A source unchanged.

Final policy sample counts:

- 3H AveragePolicy seen: 5,549,800;
- HU AveragePolicy seen: 2,072,704.

The HU AveragePolicy reservoir therefore crossed its 2M capacity during Stage B. Both Advantage reservoirs and both AveragePolicy reservoirs are now in saturation/replacement regime.

## Resource result

Postvalidation evidence:

- final checkpoint size 2.623474 GiB;
- finalized save 72.938 s;
- minimum observed WSL `MemAvailable` 7.473 GiB;
- maximum observed swap used 0 GiB;
- process max RSS from `/usr/bin/time`: 20,638,976 KiB;
- kernel swaps: 0.

Resource status remains healthy. Memory consumption increased versus Stage A, as expected after the HU policy memory filled, but there was still substantial available memory and no swap.

Two wall-time sources disagree and must remain separate rather than averaged:

- trainer report `wall_seconds`: 41,375.819 s;
- `/usr/bin/time` elapsed: 10:50:23.

This is a telemetry discrepancy only; it does not affect the milestone state or checkpoint integrity.

## Historical wrapper error — CLOSED

The original Stage B launcher raised `KeyError: 'roots'` after training because its post-run validator read `report["final"]` instead of `report["final"]["domains"]`.

The computation had already completed, finalized and written the report/checkpoint. The launcher was corrected and `tools/audit_completed_lt2_stage_b.sh` independently validated the completed run. Do not rerun Stage B because of that closed wrapper defect.

## Strategic question now

Before Stage B, weak-baseline evidence improved overall and in 3H, while HU was flat/noisy from LT1 -> LT2-A. At Stage A the HU AveragePolicy reservoir was only 820,667/2M, so the flat HU result was not treated as a ceiling.

Stage B has now crossed HU policy saturation. The next decision must therefore be based on learning evidence, especially HU, rather than more blind compute.

## Immediate finite gate — Stage A -> Stage B paired learning review

Run:

`tools/run_lt2_stage_b_learning_review.sh`

This is **read-only evaluation, not training**. It evaluates the exact Stage A and Stage B policies on the same 1000 fixed-seed empirical scenarios/deals/opponent families and preserves row-level results so a direct paired checkpoint-delta CI can be computed.

It produces:

- `SpinCore_LT2A_learning_report.json`;
- `SpinCore_LT2B_learning_report.json`;
- `SpinCore_LT2A_to_LT2B_checkpoint_delta.json`.

The checkpoint-delta CI is the missing statistic from the earlier learning-curve review. It directly addresses whether 4.5M roots improved or regressed 3H/HU against uniform-legal, passive-caller and jammer families.

## Stop condition

Do not continue training beyond iteration 7500 until the paired learning review is inspected.

If Stage B shows continued meaningful improvement with healthy resources, authorize a larger continuation from the exact Stage B checkpoint. If HU materially regresses or the learning curve stalls, investigate architecture/training dynamics before spending another multi-million-root block.

DeepCrusher remains the future product-strength reference; weak baselines are learning/regression sentinels, not the final acceptance target.

## Immediate user action

Pull current `main` and run:

```bash
bash tools/run_lt2_stage_b_learning_review.sh
```

Wait for `LT2_STAGE_B_LEARNING_REVIEW_EVAL_PASS`, then send the three JSON files copied to Windows Downloads. Do not start any additional training while this evaluation is pending.
