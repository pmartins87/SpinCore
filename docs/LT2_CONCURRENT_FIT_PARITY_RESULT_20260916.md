# SpinCore LT2 concurrent-fit result — 2026-09-16

Status: **CONCEPT PARITY PASS — PRODUCTION FUNCTION ADDED — ONE FINAL PRODUCTION PARITY GATE REQUIRED BEFORE STAGE B**

## Read-only fit microbenchmark

Source: finalized LT2 Stage A checkpoint at iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

The bounded 3H/HU Advantage-fit benchmark compared sequential and concurrent domain fits at 4 and 8 parent Torch threads, three repeats per case.  Same-thread sequential/concurrent model hashes, final losses and per-domain batch RNG states matched exactly.  The source checkpoint remained unchanged.

Measured medians:

- 4-thread sequential: 10.0085 s combined fit wall;
- 4-thread concurrent: 5.9418 s;
- 8-thread sequential: 7.1461 s;
- 8-thread concurrent: 5.4460 s.

The selected comparison was 8-thread sequential versus 8-thread concurrent:

- fit speedup: **1.312188x**;
- fit-wall reduction: about **23.79%**;
- benchmark status: `LT2_CONCURRENT_FIT_BENCHMARK_PASS`;
- `worth_full_iteration_integration=true`.

## Full iteration concept parity

A second read-only gate ran iteration 3001 twice from the same finalized iteration-3000 checkpoint: canonical sequential execution and the staged concurrent-fit candidate.

Exact semantic parity passed for:

- Advantage model hashes in both domains;
- policy model hashes in both domains;
- Advantage and policy optimizer hashes;
- per-domain batch RNG states;
- counters;
- reservoir lengths/seen counts and reservoir RNG states;
- sampler RNG state;
- every Advantage and policy sample added during the iteration, by count and streaming SHA-256;
- the complete non-timing iteration report.

Observed sample-stream equality included:

- 3H Advantage: 5,698 samples, SHA256 `4888a585df785ae03ba93cda8c5522f8bf6a2fc174a3aa66d1fca45ad06593c9`;
- 3H policy: 745 samples, SHA256 `dae901c69d92c2bdbe14e0eff3d86e1470e8df854761b1a32d44668a358ded40`;
- HU Advantage: 10,904 samples, SHA256 `3efb43023bf195814459eea87a2f62e6b6c8345ad21724a7e352d3af1bc0323c`;
- HU policy: 260 samples, SHA256 `475fdb2b4e2fc89712a8f7ffabf6af6301d7f4622e34dc377024eb981138bc27`.

The concept gate reported:

- canonical wall: 11.4179 s;
- candidate wall: 11.1143 s;
- one-shot whole-iteration speedup: 1.0273x;
- `semantic_parity=true`;
- `source_unchanged=true`;
- status `LT2_CONCURRENT_ITERATION_PARITY_PASS`.

The one-shot whole-iteration timing is diagnostic only.  The stronger repeated timing evidence is the dedicated fit benchmark.  The full-iteration gate's primary purpose was semantic parity, and it passed.

## Production integration decision

The concurrent path is worth integrating because the repeated fit benchmark shows a material 23.8% reduction in the dominant fit phase and the staged full-iteration candidate is semantically exact.  However, the concept validator contained its own candidate implementation.  Before Stage B, the actual production function must itself be compared against canonical `run_iteration`.

Production implementation added:

- `python/spincore/lean_concurrent_iteration.py`
- `tools/validate_lt2_production_concurrent_iteration.py`
- `tools/validate_lt2_production_concurrent_iteration.sh`

The production function preserves the canonical shared-sampler stream by pre-sampling root and policy episodes in the original domain/call order, collecting both domain root batches with their pre-fit models, resetting models sequentially, overlapping only the independent optimizer loops, then replaying policy trajectories in canonical domain order.

## Gate before Stage B

Run exactly one final read-only production parity gate:

```bash
bash tools/validate_lt2_production_concurrent_iteration.sh
```

Required result:

```text
LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_PASS
```

Do not launch Stage B until that result is reviewed.  If it passes, the next bounded training stage remains continuation of the same iteration-3000 checkpoint toward approximately iteration 7500 so the HU AveragePolicy reservoir crosses the 2M capacity.
