# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE A PASS — 1.8M ROOTS — PRODUCTION CONCURRENT-FIT PARITY PASS — LT2 STAGE B READY**

## Active source of truth

Read before new compute:

- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT2_CONCURRENT_FIT_PARITY_RESULT_20260916.md`
- `docs/LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

The active learning line is LT1 -> LT2. Preserve the finalized LT2 Stage A checkpoint and continue from it; do not restart LT0/LT1.

## Preserved checkpoints

- LT0 calibration: 200 iterations / 120k roots, `runs/lean_first_training/20260915_131133/checkpoint.pt`.
- LT1: 2000 iterations / 1.2M roots, `runs/long_training_lt1/20260915_181249/checkpoint.pt`, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.
- LT2 Stage A: 3000 iterations / **1.8M roots total**, `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

The Stage A checkpoint is the read-only source for Stage B and must remain preserved.

## LT2 Stage A — PASS

Stage A added 600k roots with 31 root workers, 8 parent Torch threads and vectorized batching.

Resource result:

- trainer wall scope about 2h55m47s;
- minimum observed WSL `MemAvailable` about 10.67 GiB;
- maximum swap used 0 GiB;
- process max RSS about 16.3 GiB;
- final checkpoint size 2.235843 GiB;
- finalized save about 54.5 s.

Reservoir state at iteration 3000:

- 3H Advantage saturated at 2M capacity;
- HU Advantage saturated at 2M capacity;
- 3H AveragePolicy seen 2,216,393 and in replacement regime;
- HU AveragePolicy seen 820,667 and still below capacity.

Checkpoint growth slowed sharply after 3H policy saturation, so there is no runaway serialization/memory signal.

## Learning evidence through 1.8M roots

The fixed-seed weak-baseline diagnostic remains positive overall and especially 3H:

- uniform overall cEV: +10.836 -> +15.873 -> +17.015;
- uniform 3H cEV: +2.407 -> +8.898 -> +12.769;
- passive-caller overall cEV: -2.402 -> -0.553 -> -0.064;
- jammer overall cEV: -5.498 -> -2.719 -> -1.298;
- jammer 3H cEV: +2.295 -> +4.045 -> +7.422.

HU is the tracked warning. LT1 -> LT2-A HU point estimates were flat/slightly worse across the weak families, but intervals are wide and the HU policy reservoir is only 820,667/2M. This is not yet evidence of a ceiling.

## Concurrent-fit optimization — ADMITTED FOR PRODUCTION

The repeated fit microbenchmark showed that overlapping the independent 3H/HU Advantage optimizer loops at 8 Torch threads reduced median combined fit wall from 7.1461 s to 5.4460 s:

- fit speedup: 1.312188x;
- fit-wall reduction: about 23.79%;
- exact same-thread model hashes, final losses and batch RNG states.

The full-iteration concept gate then passed exact semantic parity at iteration 3001.

The corrected **production** gate also passed:

`LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_PASS`

Evidence:

- `semantic_parity=true`;
- `source_unchanged=true`;
- `first_difference=null`;
- reference wall 11.3933 s;
- production candidate wall 11.0136 s;
- one-shot whole-iteration speedup 1.03448x.

Reference and candidate matched exactly for models, optimizers, sampler/domain RNGs, counters, reservoir RNGs, added Advantage/Policy sample streams and all non-timing report semantics. See `docs/LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`.

The production trainer now exposes `--iteration-mode concurrent_fit`. `sequential` remains the default path.

## LT2 Stage B — READY

Canonical launcher:

`tools/run_long_training_lt2_stage_b.sh`

Stage B contract:

- source: exact preserved iteration-3000 checkpoint above;
- target: iteration **7500**;
- +4500 iterations;
- +2.7M roots;
- **4.5M roots total**;
- 31 root workers;
- 8 parent Torch threads;
- worker numerical threads 1;
- vectorized batches;
- production `concurrent_fit` iteration mode;
- checkpoint every **250 iterations**;
- one-minute WSL RAM/swap telemetry;
- isolated Stage B run directory;
- final AveragePolicy fit and finalized checkpoint required.

The 250-iteration cadence reduces serialization overhead relative to Stage A while limiting maximum restart loss to one 250-iteration interval. Stage A source is never overwritten.

Why iteration 7500: the Stage-A HU policy-sample rate projects HU AveragePolicy reaching the 2M reservoir capacity around iteration 7.3k. Iteration 7500 gives a bounded margin to observe the all-four-reservoir saturated/replacement regime.

## Stop condition after Stage B

After `LT2_STAGE_B_PASS`, do **not** extend farther automatically. Review:

- final HU policy seen/length and whether 2M was crossed;
- memory/swap with all four reservoirs saturated or near saturated;
- checkpoint size/save time and growth;
- actual throughput with concurrent fit;
- 3H learning continuation;
- HU learning continuation/regression using a fresh diagnostic;
- DeepCrusher direct benchmark if the faithful oracle is ready.

Only then decide the next multi-million-root block.

## DeepCrusher

The faithful DeepCrusher R8 v22 oracle remains a parallel engineering track and future product-strength reference. Weak baselines are regression/learning sentinels, not the final acceptance target.

## Immediate user action

Pull `main` and run exactly:

```bash
bash tools/run_long_training_lt2_stage_b.sh
```

Let Stage B run until it prints either `LT2_STAGE_B_PASS` or `LT2_STAGE_B_FAIL`. Do not stop it merely because Task Manager CPU utilization oscillates; the measured execution profile is intentional. On PASS, send `SpinCore_LT2_STAGE_B_report.json` and `SpinCore_LT2_STAGE_B_memory.log` from Windows Downloads before doing any further training. On FAIL, do not restart automatically; send the terminal output and generated logs.
