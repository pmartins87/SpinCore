# LT1 completion review — 2026-09-16

Status: **LT1 execution complete; checkpoint preserved. The targeted neural-fit optimization and bounded physical resume/throughput gate have now passed. LT2 Stage A is authorized as the next finite continuation.**

See `LT1_FIT_BENCHMARK_RESULT_20260916.md` for the post-completion physical benchmark and selected runtime profile.

## Evidence
User-supplied LT1 report.json SHA256: `3cd21150451f70be5d3504df5b881e243b97b394adba71114d4405dafa390ee6`.
Run: `/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249`.
Terminal reports `LT1_SCALE_VALIDATION_PASS` and exit status 0.

2,000 consecutive iterations; 600 roots/iteration; seed 20260922.
3H: 654,000 roots, 115,262,111 nodes, 20,598,662 advantage samples seen, 1,485,285 strategy samples seen.
HU: 546,000 roots, 79,329,378 nodes, 16,312,588 advantage samples seen, 544,490 strategy samples seen.
All recorded advantage losses finite. Final policy losses: 1.11331534 (3H), 1.10055304 (HU). These are fit diagnostics, not poker-strength evidence.

The preserved finalized LT1 checkpoint was later independently loaded by the physical fit benchmark and verified as:

```text
source_iteration = 2000
source_sha256 = beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337
```

The benchmark verified the source SHA before/after and reported `source_unchanged=true`.

## Complete timing attribution
`report.wall_seconds = 24,961.87672` (6h56m01.9s).

- Advantage fit: 17,742.20762 s (71.08%).
- Trees: 5,042.20995 s (20.20%).
- Sampled policy collection: 876.64775 s (3.51%).
- 20 periodic checkpoints: 868.22447 s (3.48%).
- Final policy fit: 346.06717 s (1.39%).
- Remaining measured wall time: 86.51976 s (0.35%).

The LT1 trainer version used for this run measured wall_seconds before its final finalized=True checkpoint. That final save metric was printed but not appended to `report.checkpoint_metrics`. Hence the LT1 report excludes final-save cost; 53.30093 s / 1.97414 GiB is the iteration-2000 pre-finalization checkpoint, not independently measured final-save cost. The trainer was subsequently corrected so future reports include the finalized checkpoint metric and explicit wall scope.

Terminal `/usr/bin/time` reports 6:39:00, shorter than the trainer's wall interval; this remains an unresolved clock/measurement inconsistency. Do not average the two or promise LT2 runtime from them.
Reported CPU 904% is about 28.25% of 32 logical processors, conditional on that timing source.
Maximum RSS 13,358,660 KiB is not an aggregate worker-pool peak. Prior `free -h`: ~30 GiB WSL total, ~19 GiB available, swap zero. No current evidence that increasing WSL allocation fixes the neural-fit bottleneck.

## Sampling and reservoir interpretation
3H blind counts: 10/20 339088; 15/30 195071; 20/40 91134; 30/60 24514; 40/80 3604; 50/100 511; 60/120 78.
HU: 10/20 49224; 15/30 135335; 20/40 162153; 30/60 123438; 40/80 57339; 50/100 15394; 60/120 2667; 80/160 350; 100/200 100.
Current `legacy_scenario_data.json` assigns ZERO empirical 3H weight to 80/160 and 100/200. Their absence in 3H is consistent with that model, not a discovered sampling fault. Any rare-state oversampling is a separate explicit strategic change.

Reservoir capacity is 2M per memory per domain. Under current reservoir semantics, both advantage memories reached capacity; policy memories ended at 1,485,285 / 544,490 samples and did not reach capacity. LT1 therefore did NOT establish the fully occupied 8M-slot memory/checkpoint peak. Preserve capacity; monitor growth during LT2.

## Neural-fit gate — subsequently closed

The finite follow-up prescribed by the original completion review has been executed.

Physical benchmark evidence:

```text
benchmark report SHA256 = c63596d505c24fea51e260237ce33495f1d02f0c23efd2cbc222388a034957b1
benchmark code commit = 6d9901366a9281a584e3f03703b6bb4efaf364b9
baseline = 8 threads + reference batching = 7.663904482004 s
selected = 8 threads + vectorized batching = 6.5824409390043 s
fit_speedup = 1.1642952140430884x
```

Same-thread reference/vectorized final losses and model hashes matched. A disposable resumed iteration 2001 executed with the selected fit profile and 31 root workers, produced exactly 600 roots (327 3H + 273 HU), and left the LT1 source unchanged. This satisfies the bounded continuation gate. No repeated thread matrix is required.

## Decision and finite next step

Preserve LT1 unchanged as the continuation source. Do not start another fresh campaign and do not jump directly to tens of millions of roots.

The next finite gate is **LT2 Stage A**:

1. Copy the exact LT1 checkpoint into a separate `runs/long_training_lt2/<timestamp>/` directory.
2. Continue from iteration 2000 through iteration 3000 only: 1,000 additional iterations / 600,000 roots.
3. Use 31 root workers, 8 parent Torch threads and `batch_mode=vectorized`.
4. Checkpoint every 100 iterations and record one-minute WSL memory/swap telemetry.
5. Stop after iteration 3000 and review `report.json`, `memory.log`, checkpoint size/save time and reservoir state before any larger continuation.
6. Do not require a DeepCrusher strength verdict to authorize this infrastructure/memory stage; that remains later learning-curve work.

Canonical launcher: `tools/run_long_training_lt2_stage_a.sh`.

The reason for the bounded Stage A is specific: the 3H policy memory should reach its 2M capacity during this block at approximately the LT1 sample rate, giving the first direct observation of policy-reservoir saturation and its memory/checkpoint cost. If that transition is healthy, continue the same LT2 checkpoint in larger blocks; if not, address the measured memory/serialization bottleneck before expanding compute.
