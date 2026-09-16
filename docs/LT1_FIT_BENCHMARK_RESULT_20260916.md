# LT1 physical fit benchmark result — 2026-09-16

Status: **PASS — physical Ryzen gate closed; LT2 Stage A authorized from the preserved LT1 checkpoint.**

## Evidence identity

User-supplied benchmark report SHA256: `c63596d505c24fea51e260237ce33495f1d02f0c23efd2cbc222388a034957b1`.

Benchmark schema: `LT1_FIT_BENCHMARK_V1`.

Source LT1 checkpoint:

```text
/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt
source_iteration = 2000
source_sha256 = beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337
```

Benchmark code commit:

```text
6d9901366a9281a584e3f03703b6bb4efaf364b9
```

Runtime:

```text
torch = 2.13.0+cpu
roots_per_iteration = 600
reservoir_capacity = 2,000,000 per memory/domain
advantage_steps = 100 per domain/iteration
batch_size = 1024
root workers on resume check = 31
```

## Physical result

Baseline was the previously selected LT1 fit configuration:

```text
8 Torch threads + reference batch construction
fit_seconds = 7.663904482004
```

Winner:

```text
8 Torch threads + vectorized batch construction
fit_seconds = 6.5824409390043
fit_speedup = 1.1642952140430884x
fit-time reduction = 14.11%
```

The frozen 5% admission threshold is therefore exceeded by a wide margin. No rerun of the thread/batch matrix is required.

The full vectorized fit ranking was:

| Torch threads | Fit seconds, 3H+HU |
|---:|---:|
| 8 | **6.58244** |
| 16 | 8.47159 |
| 4 | 9.11097 |
| 2 | 9.16187 |
| 1 | 14.88826 |

This confirms that increasing Torch threads above 8 is counterproductive for the current LT1 workload on this Ryzen.

## Where the gain came from

For 3H at 8 threads, batch assembly fell from `0.594913 s` to `0.166501 s`; for HU it fell from `0.595156 s` to `0.163558 s`. Optimizer time changed only modestly. The optimization therefore removed Python-side batch-construction overhead rather than changing the learning algorithm.

Same-thread reference/vectorized comparisons produced identical final losses and identical model SHA256 values in both domains. This is the intended semantic parity check for the optimization. Cross-thread bitwise equality is not required or claimed.

## Resume integration check

The benchmark restored the original LT1 model/RNG/counters and executed one disposable resumed iteration, iteration 2001, using the selected 8-thread/vectorized fit and the production 31-worker root profile.

```text
3H roots = 327
HU roots = 273
total roots = 600
3H execution_mode = parallel_31x1
HU execution_mode = parallel_31x1
source_unchanged = true
benchmark wall_seconds = 113.7423954229962
status = LT1_FIT_BENCHMARK_PASS
```

The disposable iteration was not saved or promoted. The preserved LT1 checkpoint remains the continuation source.

## Decision

Freeze the following execution profile for the next continuation stage unless later evidence changes it:

```text
root workers = 31
worker Torch/OpenMP/BLAS threads = 1
parent Torch threads = 8
batch_mode = vectorized
```

Do not repeat the 1/2/4/8/16-thread matrix merely to seek a larger win. The current optimization gate is closed.

The projected whole-run benefit is smaller than the 16.43% fit throughput increase because trees, policy sampling, checkpointing and final policy fitting remain. Using LT1's measured 71.08% advantage-fit share, a simple Amdahl-style estimate is roughly a 10% reduction in comparable total wall time. This is planning guidance only, not a runtime promise; LT1 still has a documented clock discrepancy.

## LT2 Stage A — finite next gate

Do **not** jump directly from this PASS to tens of millions of roots. The next stage is a bounded continuation from the preserved LT1 state into a separate run directory:

```text
additional iterations = 1,000
additional roots = 600,000
start = iteration 2001
end = iteration 3000
checkpoint cadence = 100 iterations
31 root workers
8 parent Torch threads
vectorized batches
```

Why 1,000 iterations: LT1 ended with the 3H policy reservoir at 1,485,285 samples out of 2,000,000, while the HU policy reservoir was 544,490. At the observed LT1 rates, another 1,000 iterations should cause the 3H policy reservoir to reach capacity during the stage while leaving HU below capacity. This deliberately measures the first policy-memory saturation transition before authorizing a much longer extension. The exact crossing point is an estimate, not a gate.

Stage A acceptance review after completion must inspect:

- successful continuation through iteration 3000 with finite losses;
- source LT1 checkpoint still byte-identical;
- WSL memory availability and swap telemetry during the run;
- checkpoint size/save-time behavior after 3H policy memory fills;
- actual fit/tree/policy-sampling throughput under vectorized batches;
- reservoir counts/capacities and any evidence of serialization or memory pressure.

If resource behavior is healthy, continue the same LT2 checkpoint into larger resumable blocks. If memory/checkpoint behavior becomes the dominant bottleneck, compact reservoir/checkpoint representation is preferred over shrinking the 2M reservoir merely for convenience.
