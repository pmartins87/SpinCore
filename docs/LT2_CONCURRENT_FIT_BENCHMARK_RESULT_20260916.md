# SpinCore LT2 concurrent-fit benchmark result — 2026-09-16

Status: **MICROBENCHMARK PASS — full-iteration semantic parity still required before Stage B**

Source checkpoint: finalized LT2 Stage A iteration 3000 at `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

The bounded read-only benchmark compared sequential and concurrent 3H/HU Advantage optimizer loops at 4 and 8 parent Torch threads, three repeats per case. Resets remained sequential and deterministic. Same-thread sequential/concurrent parity required exact equality of final model hashes, last losses and per-domain batch RNG states.

Measured medians:

- 4-thread sequential: `10.008498176 s` combined fit wall;
- 4-thread concurrent: `5.941759533 s`;
- 8-thread sequential baseline: `7.146130735 s`;
- 8-thread concurrent: `5.445964403 s`.

The selected concurrent case is 8 threads. Relative to the 8-thread sequential production baseline, fit speedup is `1.312188293x`, equivalent to about a 23.79% fit-wall reduction. The source checkpoint remained unchanged and the benchmark emitted `LT2_CONCURRENT_FIT_BENCHMARK_PASS` with `worth_full_iteration_integration=true`.

Important limitation: this benchmark isolated only the optimizer loops. It does **not** by itself prove that reordering a complete iteration preserves the canonical scenario sampler stream, root traversal inputs, policy-trajectory inputs, reservoirs, optimizer states and RNG states.

Therefore Stage B is not yet authorized. The next finite gate is one exact full-iteration parity test at iteration 3001. The candidate pre-samples root and policy episodes in the exact canonical sampler-call order, collects both domains with the pre-fit models, resets both Advantage networks sequentially, overlaps only the independent optimizer loops, and then replays policy trajectories in canonical domain order. The gate compares non-timing iteration semantics, model/optimizer hashes, sampler/batch/reservoir RNG states, counters, reservoir sizes/seen counts, and streaming hashes of every sample added during the iteration.

Canonical parity wrapper: `tools/validate_lt2_concurrent_iteration.sh`.

Do not start LT2 Stage B until `LT2_CONCURRENT_ITERATION_PARITY_PASS` is reviewed.