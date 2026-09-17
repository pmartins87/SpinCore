# SpinCore Current Work

Date: 2026-09-16
Status: **LT2 STAGE A PASS — 1.8M ROOTS — LEARNING CURVE POSITIVE OVERALL/3H — CONCURRENT FIT CONCEPT PARITY PASS — FINAL PRODUCTION PARITY GATE NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT2_CONCURRENT_FIT_PARITY_RESULT_20260916.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT1_COMPLETION_REVIEW_20260916.md`
- `docs/LT1_FIT_BENCHMARK_RESULT_20260916.md`

The active learning line is LT1 -> LT2. Preserve the finalized LT2 Stage A checkpoint and continue from it; do not restart LT0/LT1.

## Preserved checkpoints

- LT0 calibration: 200 iterations / 120k roots, `runs/lean_first_training/20260915_131133/checkpoint.pt`.
- LT1: 2000 iterations / 1.2M roots, `runs/long_training_lt1/20260915_181249/checkpoint.pt`, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.
- LT2 Stage A: 3000 iterations / 1.8M roots, `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

## LT2 Stage A result

Stage A added 600k roots with 31 root workers, 8 parent Torch threads and vectorized batching. It completed successfully with no swap use. Minimum observed WSL MemAvailable was about 10.67 GiB; process max RSS about 16.3 GiB; final checkpoint size 2.235843 GiB. The 3H AveragePolicy reservoir crossed the 2M capacity; HU ended at 820,667/2M. Checkpoint growth slowed sharply after 3H policy saturation, so there is no runaway serialization/memory signal.

The fixed-seed weak-baseline learning curve remained positive overall and especially 3H from 120k -> 1.2M -> 1.8M roots. HU remained noisy/flat from 1.2M to 1.8M and must be tracked at the next meaningful checkpoint; it is not yet a demonstrated plateau because the HU policy reservoir is far from saturation.

## Concurrent-fit optimization result

The read-only fit microbenchmark compared sequential/concurrent 3H+HU Advantage fitting at 4 and 8 threads, three repeats each. Same-thread model hashes, final losses and batch RNG states matched exactly. The best admitted case was 8-thread concurrent:

- sequential-8 median combined fit wall: 7.1461 s;
- concurrent-8 median fit wall: 5.4460 s;
- speedup: 1.312188x;
- fit-wall reduction: about 23.79%.

A full iteration-3001 concept gate then compared canonical sequential execution with the staged concurrent candidate. Exact semantic parity passed for model/optimizer state, sampler/domain RNGs, counters, reservoir RNGs, added-sample streams and non-timing report fields. Source checkpoint remained unchanged. One-shot whole-iteration timing was 11.4179 s reference versus 11.1143 s candidate; this single timing is diagnostic only, while the repeated fit benchmark is the timing evidence.

Because the concept validator contained its own staged candidate, production integration is not yet authorized solely by that result.

## Production concurrent function added

The production implementation now exists at:

- `python/spincore/lean_concurrent_iteration.py`
- `tools/validate_lt2_production_concurrent_iteration.py`
- `tools/validate_lt2_production_concurrent_iteration.sh`

It preserves canonical shared-sampler order by pre-sampling root and policy episodes in the original call order, collecting both root batches with their pre-fit domain models, resetting models sequentially, overlapping only the two independent Advantage optimizer loops, and replaying policy trajectories in canonical domain order.

## Immediate finite gate

Run exactly once:

```bash
bash tools/validate_lt2_production_concurrent_iteration.sh
```

Required result:

```text
LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_PASS
```

This is a read-only gate against the preserved iteration-3000 checkpoint. Do not start Stage B until the report is reviewed.

## Stage B after production parity

If the production parity gate passes, continue the same LT2 learning state toward approximately iteration 7500 (+4500 iterations / +2.7M roots, 4.5M roots total). At the observed HU strategy-sample rate, the HU AveragePolicy reservoir should cross 2M near iteration 7.3k. That is the next natural bounded milestone for full four-reservoir saturation and a new 3H/HU learning-curve review.

The Stage B launcher must preserve the iteration-3000 checkpoint, use 31 root workers, 8 Torch threads, vectorized batches, the admitted production concurrent-fit path, memory telemetry and resumable checkpoints. Do not launch it manually before the parity gate result is reviewed.

## DeepCrusher

The faithful DeepCrusher R8 v22 oracle remains a parallel engineering track and future product-strength reference. Weak baselines are regression/learning sentinels, not the final acceptance target.

## Immediate user action

Pull current `main`, run `bash tools/validate_lt2_production_concurrent_iteration.sh`, and send the generated `report.json`. **Do not start LT2 Stage B yet.**
