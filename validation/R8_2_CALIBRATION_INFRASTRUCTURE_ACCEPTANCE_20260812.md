# SpinCore R8.2 — calibration infrastructure acceptance

Status: **PASS — INFRASTRUCTURE / PRECOMMIT ONLY**

R8.2 official calibration status: **NOT RUN / NOT PASS**

READY FOR OFFICIAL TRAINING: **NO**

READY FOR TABLES: **NO**

## Accepted implementation

- gate design: `validation/R8_2_CALIBRATION_GATE_DESIGN_20260812.md`
- selector: `python/spincore/production_calibration.py`
- schema: `SPINCORE_R8_PRODUCTION_CALIBRATION_V1`
- regression: `python_tests/test_r8_production_calibration.py`
- authoritative regression run: `31657726682`
- regression result: **PASS**

## Proven semantics

The selector fails closed unless a candidate concurrency produces exactly the same complete stream-to-semantic-digest mapping as the serial reference and completes without an error. A faster trial with any changed, missing or extra stream state is ineligible.

The authoritative digest is not caller-defined. It is the validated `generation_id` of `SPINCORE_R8_PRODUCTION_TRANSACTION_V1`, which binds the production stream identity and hashes of the stream/model/RNG checkpoint, scheduler checkpoint and central Algorithm-R checkpoint. Calibration rejects transactions lacking semantic-consistency validation, mismatched generation identity, malformed generation identity or duplicate stream identity.

Among eligible trials, the highest semantic throughput wins. Exact throughput ties select the lower concurrency. CPU utilization and memory are telemetry only; no arbitrary CPU target or minimum speedup can override semantic equivalence.

Parallelism remains limited to genuinely independent production streams. This infrastructure does not authorize root-level parallelism inside one `(profile, domain, algorithm_seed)` stream and does not change the persistent live `batch_rng` execution-order contract.

## Still required before R8.2 can become PASS

1. finite R7.4 final PASS authorizing R8;
2. exact R8.0 GGPoker production profile materialized from bound first-party selected-state evidence;
3. R8.1 production infrastructure remains accepted;
4. physical calibration on the intended Ryzen host using the frozen production identities/workload;
5. persisted calibration report with at least one semantically exact error-free trial selected by the frozen rule.

This record neither starts production training nor closes the R7.3 exact-reproducibility release debt.
