# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE A PASS — 1.8M ROOTS — PRODUCTION CONCURRENT-FIT PARITY PASS — LT2 STAGE B READY WITH 31 WORKERS**

## Active source of truth

Read before new compute:

- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT2_CONCURRENT_FIT_PARITY_RESULT_20260916.md`
- `docs/LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

The active learning line is LT1 -> LT2. Preserve the finalized LT2 Stage A checkpoint and continue from it; do not restart LT0/LT1.

## Preserved source

LT2 Stage A is finalized at iteration 3000 / 1.8M roots:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`

SHA256:

`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`

This checkpoint remains read-only for Stage B.

## Evidence through Stage A

Stage A used 31 root workers, 8 parent Torch threads and vectorized batching. It completed with no swap use, minimum WSL MemAvailable about 10.67 GiB, process max RSS about 16.3 GiB, and a 2.236 GiB final checkpoint. Both Advantage reservoirs and the 3H AveragePolicy reservoir are saturated; HU AveragePolicy is 820,667/2M.

The weak-baseline learning curve remains positive overall and especially 3H. HU is still noisy/flat and is the principal strategic signal to re-check after HU AveragePolicy reaches capacity.

## Production concurrent fit — ADMITTED

The repeated fit benchmark gave a 1.312188x speedup for 8-thread concurrent 3H/HU Advantage fitting versus sequential 8-thread fitting, with exact same-thread model/loss/RNG parity.

The production iteration gate passed exact semantic parity at iteration 3001:

- `semantic_parity=true`;
- `source_unchanged=true`;
- `first_difference=null`;
- reference wall 11.3933 s;
- production concurrent candidate wall 11.0136 s.

Production `--iteration-mode concurrent_fit` is admitted. Sequential remains the default compatibility path.

## Worker-count decision

A short worker-count benchmark was prepared to test whether 16/20/24/28 workers could outperform the proven 31-worker topology. The user explicitly declined this extra benchmark and chose to keep **31 workers** for Stage B.

Therefore the worker-count tuning gate is closed without execution. Do not block Stage B on it and do not change the canonical worker count away from 31 without a future explicit decision.

## Checkpoint cadence

The canonical Stage B launcher remains at checkpoint every 250 iterations. This preserves the already-reviewed restart-risk/serialization tradeoff. A wider cadence such as 500 could save only several additional minutes, but it is not required for Stage B and is not being changed as part of the worker-count decision.

## LT2 Stage B — READY

Canonical launcher:

`tools/run_long_training_lt2_stage_b.sh`

Stage B contract:

- source: preserved iteration-3000 LT2 Stage A checkpoint;
- target: iteration 7500;
- +4500 iterations / +2.7M roots;
- 4.5M roots total;
- **31 root workers**;
- 8 parent Torch threads;
- worker numerical threads 1;
- vectorized batches;
- production `concurrent_fit` iteration mode;
- checkpoint every 250 iterations;
- one-minute WSL RAM/swap telemetry;
- isolated Stage B run directory;
- final AveragePolicy fit and finalized checkpoint required.

Why iteration 7500: the Stage-A HU policy-sample rate projects HU AveragePolicy reaching the 2M reservoir capacity around iteration 7.3k. Iteration 7500 gives a bounded margin to observe the all-four-reservoir saturated/replacement regime.

## Stop condition after Stage B

After `LT2_STAGE_B_PASS`, do not extend farther automatically. Review final HU policy saturation, memory/swap, checkpoint size/save time, actual throughput, 3H learning, HU learning, and DeepCrusher comparison if the faithful oracle is ready.

## Immediate user action

Pull current `main` and run:

```bash
bash tools/run_long_training_lt2_stage_b.sh
```

Let it run until `LT2_STAGE_B_PASS` or `LT2_STAGE_B_FAIL`. On PASS, send `SpinCore_LT2_STAGE_B_report.json` and `SpinCore_LT2_STAGE_B_memory.log` from Windows Downloads. On FAIL, do not restart automatically; send the terminal output and generated logs.
