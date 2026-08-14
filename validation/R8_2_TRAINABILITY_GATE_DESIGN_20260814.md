# SpinCore R8.2 — absolute production trainability gate

Date: 2026-08-14  
Status: **PRECOMMITTED INFRASTRUCTURE — PHYSICAL MEASUREMENT NOT RUN / NOT PASS**  

`READY FOR OFFICIAL TRAINING = NO`  
`READY FOR TABLES = NO`

## Purpose

The existing R8.2 calibration chooses the fastest concurrency that preserves the accepted production semantics exactly. That is necessary but not sufficient: the fastest correct configuration can still be operationally useless if the complete official training plan would require many months or years.

This supplemental gate makes trainability an explicit production requirement without relaxing any strategic-quality requirement.

Machine-readable contract:

```text
validation/R8_TRAINABILITY_TIME_BUDGET_CONTRACT_20260814.json
```

Pure projection implementation:

```text
python/spincore/production_trainability.py
SPINCORE_R8_PRODUCTION_TRAINABILITY_V1
```

Regression:

```text
python_tests/test_r8_production_trainability.py
```

## Frozen wall-clock requirement

The complete official core-policy training path from the start of R8.3 through completed R8.4 and the R8.5 frozen HU+3H policy artifacts must fit inside **90 wall-clock days** for every selected production profile and every required algorithm-seed stream.

A **20% operational reserve is inside those 90 days**. The nominal conservative upper-bound projection therefore has an implied budget of 75 days:

```text
75 nominal days × 1.20 = 90 days
```

The reserve is not permission to run for 108 days. Ninety days remains the hard ceiling.

R9 strategic auditing, R10 OpenHoldem integration, R11 opponent-data/exploitation work and R12 operational homologation are not counted as neural/CFR training time. They remain separate mandatory gates.

## What is timed

A timing sample is not traversal-only and not inference-only. It is one **complete durable production iteration from accepted checkpoint to the next accepted checkpoint**, including the work required to advance that stream under the frozen production algorithm:

```text
traversal/action/chance work
+ reservoir work
+ required neural fitting
+ serialization/checkpoint/transaction cost
```

At least three complete-iteration timings are required for every stream in the final frozen plan. The projection uses the **slowest valid repeated timing for each stream**, not its mean.

The measurement must be made on the intended production Ryzen host, or on another explicitly frozen production compute topology that receives its own semantically exact calibration. GitHub-hosted runner elapsed time is not acceptable evidence for this gate.

## Complete workload rule

The projection input must exactly cover the final scheduler plan:

```text
all selected production profiles
× TRUE_HEADS_UP + THREE_HANDED
× all required algorithm seeds/streams
× every frozen production iteration
```

The set of measured stream IDs must equal the planned stream-ID set exactly. Missing or extra streams fail closed.

Every timing row must come from the concurrency selected by the semantically exact R8.2 calibration. A faster concurrency whose integrated production state differs from the serial reference remains ineligible regardless of its projected finish date.

## Conservative scheduling projection

Each `(profile, domain, algorithm_seed)` stream remains serial because of the accepted persistent live RNG execution-order contract. The estimator computes each stream's serial duration using its slowest repeated full-iteration timing, then applies deterministic longest-processing-time-first packing while treating each entire stream as pinned to one worker.

This `LPT_STREAM_PINNED_CONSERVATIVE_UPPER_BOUND` is intentionally at least as restrictive as the actual scheduler, which may move a stream between workers at iteration boundaries. It therefore does not assume unsafe intra-stream/root parallelism or optimistic perfect packing.

The gate passes only when:

```text
all measurements are semantically exact and error-free
AND
LPT projected nominal upper bound × 1.20 <= 90 days
```

## What may and may not be optimized

Engineering speedups are encouraged when they preserve the exact accepted semantics: C++/Python hotspot work, batching/vectorization, memory/layout improvements, process reuse, semantically exact thread tuning and concurrency across independent streams.

The following are **not** runtime optimizations and cannot be used merely to make the clock fit:

```text
fewer required seeds
fewer required roots/iterations
fewer optimizer steps
smaller required ensemble
removing HU or 3H
removing a selected production profile
changing uncertainty/behavior semantics
unsafe intra-stream parallelism
```

If the selected architecture does not fit after valid engineering optimization, official R8 training remains BLOCKED. A new strategic abstraction study must be designed and frozen before observing its outputs; the project may not silently choose a runner-up after results are known.

## Relation to R7.5.3 / R7.5.4

R7.5.3 has now selected `C0_V1_FROZEN_CONTROL`, keeping the compact 126-byte observation and 152,438-parameter model rather than promoting a richer representation that failed the frozen evidence cascade. This is favorable for trainability, but it is not sufficient proof of the 90-day gate.

R7.5.4 already records `nodes_per_root`, `seconds_per_root`, peak memory, samples/root and effective unique branches. Those metrics remain strategic/engineering evidence and favor simpler trees when strategic candidates are equivalent. The new absolute gate adds the missing requirement: even the selected action abstraction is not R8-production-eligible until its complete exact-profile Ryzen projection satisfies the 90-day ceiling.

## Current status

No physical trainability PASS is claimed now because the exact R8.0 selected-state production profile and official physical R8.2 calibration are still prerequisites. The method and threshold are frozen **before** those results are observed.

`READY FOR OFFICIAL TRAINING = NO`  
`READY FOR TABLES = NO`
