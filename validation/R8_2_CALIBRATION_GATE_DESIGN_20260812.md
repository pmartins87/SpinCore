# SpinCore R8.2 — production calibration gate precommit

Status: **DESIGN/INFRASTRUCTURE ONLY — NOT R8.2 PASS**

READY FOR TABLES: **NO**

## Purpose

Choose the safe concurrency used for production training on the Ryzen host without changing the accepted learning semantics merely to increase CPU utilization.

## Hard prerequisites for an official R8.2 calibration

An official calibration run is forbidden until all of the following are true:

1. finite R7.4 final gate is PASS and explicitly authorizes R8;
2. the exact R8.0 GGPoker production profile being calibrated is materialized from bound first-party evidence;
3. R8.1 integrated production infrastructure remains PASS;
4. the learning profile, strategy domain, algorithm seeds and production profile identities are frozen for the calibration.

The existence of this design and its code does not satisfy any of those prerequisites.

## Calibration unit

Parallelism is only across genuinely independent production streams. A stream is identified by the accepted production identity including production profile, strategy domain and algorithm seed. Execution inside one stream remains serial in the order required by the persistent live `batch_rng` contract.

Naive root-level parallelism inside a stream is not a calibration candidate.

## Serial reference

For the exact set of independent streams participating in calibration, execute the reference schedule with concurrency 1 and persist the authoritative semantic state digest of each stream after the fixed calibration workload. The digest must derive from the accepted integrated transaction/checkpoint state, not from logs or performance counters.

## Candidate trials

Test a finite set of positive concurrency levels appropriate for the physical Ryzen host. Every trial must execute the same frozen per-stream workload as the serial reference and report:

- concurrency;
- elapsed wall-clock seconds;
- completed work units;
- the authoritative semantic digest for every participating stream;
- error/OOM status;
- peak memory and CPU utilization as telemetry when available.

## Fail-closed eligibility

A trial is eligible only if:

- it completes without error or OOM; and
- its complete stream->semantic-digest mapping is exactly equal to the serial reference mapping.

A faster run that changes even one stream digest is rejected. Missing or extra streams are also a semantic mismatch.

CPU utilization is **not** an acceptance gate. There is no minimum CPU percentage and no minimum speedup requirement.

## Selection

Among eligible trials, select the highest measured throughput (`completed_work_units / elapsed_seconds`). On an exact throughput tie, select the lower concurrency to reduce operational complexity and resource pressure.

If no trial is eligible, R8.2 fails closed and production training remains blocked.

## Implementation

The pure fail-closed selector is implemented in `python/spincore/production_calibration.py` with schema `SPINCORE_R8_PRODUCTION_CALIBRATION_V1`. Regression coverage is in `python_tests/test_r8_production_calibration.py`.

The selector deliberately emits `ready_for_official_training=false` and `ready_for_tables=false`; only a future official calibration orchestrator, after all hard prerequisites are independently proved, may consume its result to authorize the chosen concurrency for R8.3/R8.4.

## Frozen interpretation

This gate optimizes **semantic throughput**, not headline CPU utilization. It does not relax any R7 gate, does not change the selected R7.3 behavior, does not close the R7.3 exact-reproducibility debt, and does not authorize table use.
