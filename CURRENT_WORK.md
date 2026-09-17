# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B COMPUTE COMPLETE AT 4.5M ROOTS — TRAINER PASS — AUDIT-ONLY POSTVALIDATION NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT2_CONCURRENT_FIT_PARITY_RESULT_20260916.md`
- `docs/LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

The active learning line is LT1 -> LT2. Do not restart any completed stage and do not continue beyond iteration 7500 until Stage B evidence is reviewed.

## Preserved source

LT2 Stage A remains preserved at iteration 3000 / 1.8M roots:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`

SHA256:

`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`

## LT2 Stage B compute — COMPLETED

The canonical 31-worker Stage B run reached iteration 7500 and the trainer itself exited successfully after final AveragePolicy fitting and the finalized checkpoint write.

Completed run directory:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911`

Observed completion evidence from the trainer:

- iteration 7500/7500 completed;
- final checkpoint written;
- final checkpoint size about 2.623474 GiB;
- finalized checkpoint write about 72.94 s;
- trainer printed `PASS report=... checkpoint=...`;
- `/usr/bin/time` exit status 0;
- elapsed wall time 10:50:23;
- max resident set about 20.64 GB decimal / 19.68 GiB;
- kernel-reported swaps 0;
- final 3H roots 2,452,500;
- final HU roots 2,047,500;
- total roots 4,500,000;
- 3H AveragePolicy `strategy_seen=5,549,800`;
- HU AveragePolicy `strategy_seen=2,072,704`, so the HU 2M reservoir threshold was crossed.

This establishes that the expensive training computation itself completed. It must **not** be rerun because of the wrapper error below.

## Post-validation wrapper defect

After the successful trainer exit, `tools/run_long_training_lt2_stage_b.sh` failed in its small report-validation snippet with:

`KeyError: 'roots'`

Root cause: `compact_report()` stores final domain records under `report["final"]["domains"]`, but the launcher incorrectly summed `report["final"].values()` as though domain dictionaries were directly under `final`.

This error happened **after** training, finalization, report creation, finalized checkpoint write, and source-preservation check. It is a wrapper validation bug, not a training failure.

The launcher is corrected for future use to read `data["final"]["domains"]` and to copy evidence to Windows before post-validation.

## Audit-only recovery — REQUIRED NEXT

Do not rerun Stage B. Run only:

`tools/audit_completed_lt2_stage_b.sh`

The audit targets the already-completed run, validates report schema/config/execution mode/root totals/finalized checkpoint metrics, confirms both policy reservoirs crossed 2M, parses the memory log, checks the preserved Stage A source SHA256, computes the completed Stage B checkpoint SHA256, and copies evidence to Windows Downloads.

Required result:

`LT2_STAGE_B_POSTVALIDATION_PASS`

After that, send `SpinCore_LT2_STAGE_B_report.json` and `SpinCore_LT2_STAGE_B_memory.log` for the resource and learning review. Do not continue training past iteration 7500 before that review.

## Execution profile used for Stage B

- 31 root workers;
- 8 parent Torch threads;
- worker numerical threads 1;
- vectorized batches;
- production `concurrent_fit` iteration mode;
- checkpoint every 250 iterations;
- one-minute WSL RAM/swap telemetry.

The worker-count benchmark was intentionally skipped by user choice; 31 workers remain the canonical topology for this completed stage.

## Immediate user action

Pull current `main` and run:

```bash
bash tools/audit_completed_lt2_stage_b.sh
```

This does **not** train anything and must finish far faster than Stage B. Send the terminal output plus `SpinCore_LT2_STAGE_B_report.json` and `SpinCore_LT2_STAGE_B_memory.log`. Do not rerun the 10h50 Stage B computation and do not start any further training yet.
