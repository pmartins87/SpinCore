# SpinCore LT2 production concurrent-fit parity result — 2026-09-17

Status: **PASS — production concurrent-fit iteration path is semantically equivalent to canonical sequential iteration 3001**

## Source

Preserved LT2 Stage A checkpoint:

- iteration: 3000;
- path: `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`;
- SHA256: `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

The source remained unchanged during the gate.

## Gate

The corrected production gate ran iteration 3001 twice from the same source:

1. canonical `run_iteration`;
2. production `run_iteration_concurrent_fit` from `python/spincore/lean_concurrent_iteration.py`.

Execution profile:

- root workers: 31;
- parent Torch threads: 8;
- batch mode: vectorized;
- candidate change: only the independent 3H/HU Advantage optimizer loops overlap.

The validator removes only explicitly non-semantic timing telemetry before equality comparison. Every training-state field remains strict.

## Result

`LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_PASS`

- `semantic_parity=true`;
- `source_unchanged=true`;
- `first_difference=null`;
- reference wall: 11.393278807 s;
- candidate wall: 11.013555343 s;
- one-shot whole-iteration speedup: 1.034477828x.

The whole-iteration timing is diagnostic only. The repeated fit microbenchmark remains the stronger throughput measurement and showed 1.312188x fit speedup / about 23.79% fit-wall reduction for concurrent 8-thread fitting.

## Exact semantic evidence

Reference and production candidate matched exactly for:

- sampler RNG state;
- 3H/HU Advantage model hashes;
- 3H/HU policy model hashes;
- Advantage optimizer hashes;
- policy optimizer hashes;
- domain batch RNG states;
- counters;
- Advantage reservoir seen/length/RNG state;
- AveragePolicy reservoir seen/length/RNG state;
- every added Advantage sample stream by count and SHA256;
- every added AveragePolicy sample stream by count and SHA256;
- non-timing iteration report semantics.

Iteration-3001 sample-stream equality included:

- 3H Advantage: 5,698 samples, SHA256 `4888a585df785ae03ba93cda8c5522f8bf6a2fc174a3aa66d1fca45ad06593c9`;
- 3H policy: 745 samples, SHA256 `dae901c69d92c2bdbe14e0eff3d86e1470e8df854761b1a32d44668a358ded40`;
- HU Advantage: 10,904 samples, SHA256 `3efb43023bf195814459eea87a2f62e6b6c8345ad21724a7e352d3af1bc0323c`;
- HU policy: 260 samples, SHA256 `475fdb2b4e2fc89712a8f7ffabf6af6301d7f4622e34dc377024eb981138bc27`.

Because the full semantic state is identical after one iteration, repeated use preserves equivalence inductively: each subsequent iteration starts from the same state and consumes the same canonical sampler/RNG streams under the validated production transformation.

## Production admission

The concurrent-fit execution path is admitted for LT2 Stage B.

The authoritative trainer now exposes:

`--iteration-mode concurrent_fit`

The sequential path remains the default and is preserved for portability/regression comparison.

## Next bounded milestone

LT2 Stage B continues the preserved iteration-3000 state through iteration 7500:

- +4,500 iterations;
- +2.7M roots;
- 4.5M roots total;
- 31 root workers;
- 8 parent Torch threads;
- vectorized batches;
- production `concurrent_fit` iteration mode;
- checkpoint cadence 250 iterations;
- one-minute WSL memory/swap telemetry;
- isolated Stage B run directory; Stage A source remains read-only.

Canonical launcher:

`tools/run_long_training_lt2_stage_b.sh`

After Stage B completes, stop before extending farther. Review full four-reservoir saturation, memory/swap, checkpoint cost, actual throughput and a new 3H/HU strength diagnostic.
