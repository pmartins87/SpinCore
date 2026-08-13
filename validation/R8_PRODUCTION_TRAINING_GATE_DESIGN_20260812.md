# R8 Production Training — finite gate precommit 2026-08-12

`READY FOR TABLES = NO`.

This document freezes the **shape and sequencing** of R8 before any official production result is observed. It does not authorize official R8 training before `validation/R7_4_FINAL_GATE.json` reports `r7_4_pass = true` and `r7_4_ready_to_advance_to_r8 = true`.

The deferred R7.3 exact historical reproducibility issue remains release debt. R8 engineering may proceed under the explicitly provisional strategy-quality path, but that debt is not PASS and must close or be formally dispositioned before final table homologation.

## R8 identity

Every production policy is identified by the full tuple:

```text
Economic/Structure Profile
× Strategy Domain
× Ruleset
× Action Abstraction
× Utility Model
× Learning Profile
```

No policy may be reused across a different tuple by fallback or convenience. Strategy domains remain separate:

```text
TRUE_HEADS_UP
THREE_HANDED
```

HU and 3H therefore remain independent production artifacts even when they share encoder/network implementation.

## Frozen R8 sequence

### R8.0 — production profile / identity / rules

Before official production training, materialize a machine-readable production profile describing the real target game configuration. At minimum it must bind:

- platform/game family and ruleset source/version;
- table size, buy-in/currency and multiplier;
- starting chips;
- complete blind/ante structure needed by the solver input;
- normalized payout shares used by exact ICM utility;
- action abstraction identity;
- utility model identity;
- learning profile identity;
- immutable source/provenance evidence.

**No GGPoker economic constant may be invented from R7.4 pilot values.** The R7.4 pilot configuration is validation evidence only unless independently proven to equal a production state.

The machine-readable contract is `SPINCORE_R8_PRODUCTION_PROFILE_V3`. State-dependent evidence must be scoped to `SELECTED_PROFILE_STATE` and bound to the exact `table_size × buy_in_minor_units × multiplier`. Global evidence cannot prove state-dependent stack/blind/payout fields.

The acquisition/validation pipeline is already accepted as infrastructure:

```text
SPINCORE_R8_SELECTED_STATE_EVIDENCE_PACKET_V1
ProductionProfile V3 builder
validation/R8_0_EVIDENCE_ACQUISITION_PIPELINE_ACCEPTANCE_20260812.md
regression 31651412158 = PASS
```

R8.0 itself is **not PASS** until the exact selected-state GGPoker data are captured from bound first-party evidence. The current public web representation proves important global facts but does not reliably bind every dynamically rendered stack/blind/payout row to the selected buy-in/multiplier. Missing data remain fail-closed.

### R8.1 — deterministic production streams + durable transaction

The selected R7.3/R7.4 mechanism freezes:

```text
primary RNG = one persistent live bundle.batch_rng in execution order
```

That RNG is consumed by traversal/action sampling, Algorithm-R replacement and training minibatch sampling. Therefore naive root-level parallelism within one `(profile, domain, algorithm_seed)` stream would change the selected algorithm and is forbidden unless a separately designed dispatcher later proves exact stream preservation.

Accepted R8.1 execution model:

1. parallelize only genuinely independent `(profile, domain, algorithm_seed)` streams;
2. execute each individual stream serially in its persistent RNG order;
3. central Algorithm-R insertion remains bound to exact root order and stream identity;
4. scheduler progress advances only after a durable stream checkpoint receipt exists;
5. a production generation atomically binds stream/model/RNG checkpoint, scheduler state and Algorithm-R state.

Accepted infrastructure schemas:

```text
SPINCORE_R8_INDEPENDENT_STREAM_SCHEDULER_V2
SPINCORE_R8_CENTRAL_ALGORITHM_R_V2
SPINCORE_R8_SCHEDULER_DURABLE_CHECKPOINT_V1
SPINCORE_R8_PRODUCTION_TRANSACTION_V1
SPINCORE_R8_PRODUCTION_TRANSACTION_POINTER_V1
```

The central Algorithm-R reservoir is keyed by exact `profile × domain × algorithm_seed × roots_per_iteration`; it rejects cross-seed batches, root/iteration disagreement, sample/iteration disagreement and root gaps. Scheduler leases at most one whole iteration per exact stream and requires checkpoint locator, size, SHA-256 and parent SHA-256 before advancing.

The integrated production transaction uses immutable generations plus an atomic `CURRENT` pointer. Publication validates that stream checkpoint, scheduler and Algorithm-R all describe the same profile/domain/seed/iteration/root position and verifies their bytes/hashes. A crash therefore leaves either the previous complete generation or the next complete generation authoritative; a partial generation is not accepted.

Regression evidence:

```text
profile V3 regression                 31638697150 = PASS
Algorithm-R V2 regression             31639134952 = PASS
scheduler durable-checkpoint          31639331611 = PASS
integrated production transaction     31650375049 = PASS
```

Acceptance record:

```text
validation/R8_1_INFRASTRUCTURE_ACCEPTANCE_20260812.md
```

Accordingly **R8.1 = PASS — INFRASTRUCTURE ONLY**. This does not authorize official production training while R7.4 final and R8.0 remain unresolved prerequisites.

CPU utilization is not an R8.1 acceptance criterion. Correct semantic throughput is the objective.

### R8.2 — exact-profile Ryzen9 calibration

The calibration rule is frozen before physical calibration in:

```text
validation/R8_2_CALIBRATION_GATE_DESIGN_20260812.md
python/spincore/production_calibration.py
SPINCORE_R8_PRODUCTION_CALIBRATION_V1
```

Official R8.2 calibration is forbidden until:

1. finite R7.4 final PASS authorizes R8;
2. exact R8.0 production profile is materialized from bound first-party evidence;
3. R8.1 remains accepted;
4. profile/domain/algorithm-seed/workload identities are frozen for the benchmark.

Calibration is only across genuinely independent streams. For the exact frozen set of streams, a serial reference first produces an authoritative integrated production generation for each stream. Candidate concurrency levels execute the identical fixed workload.

A trial is eligible only if it completes without error/OOM and its complete stream-to-state mapping is **exactly equal** to the serial reference. The authoritative per-stream state identity is the validated R8.1 integrated `generation_id`, which binds production identity and hashes of stream/model/RNG, scheduler and Algorithm-R components. Caller-defined log checksums cannot substitute for it.

Among eligible trials, select the greatest measured semantic throughput (`completed_work_units / elapsed_seconds`). Exact throughput ties choose the lower concurrency. CPU utilization, peak memory and checkpoint cost are telemetry; there is no arbitrary CPU percentage or minimum speedup threshold.

Regression evidence for the precommitted selector is recorded in:

```text
validation/R8_2_CALIBRATION_INFRASTRUCTURE_ACCEPTANCE_20260812.md
```

The selector/precommit being accepted is **not R8.2 PASS**. R8.2 becomes PASS only after the actual Ryzen calibration runs under all prerequisites and persists a valid selected concurrency.

### R8.3 — official TRUE_HEADS_UP production training

Train the official HU policy for every accepted production profile using the R7 mechanism carried through R7.4:

```text
behavior semantic: SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1
ensemble size: 4
epsilon scale: 1.75
epsilon cap: 0.50
partial-exact opponent levels: 2
primary RNG: one persistent live batch RNG in execution order
utility: exact explicit-payout ICM delta for the accepted profile
action abstraction: frozen profile identity
```

Production training must cover the accepted production stack/blind/profile support rather than only the finite R7.4 pilot scenarios. R8.3 produces a candidate official HU artifact and evidence for R9; it never authorizes table use.

### R8.4 — official THREE_HANDED production training

Train the official 3H policy under the same identity discipline and accepted 3H production support. The 3H artifact is independent of HU; HU success cannot substitute for absent or failed 3H training.

R8.4 produces a candidate official 3H artifact and evidence for R9; it never authorizes table use.

### R8.5 — freeze official production policies

Freeze every official production policy artifact with immutable provenance sufficient for R9/R10 to verify exactly what is being audited and later loaded by OpenHoldem. At minimum bind:

- production profile and strategy domain;
- ruleset/action-abstraction/utility/learning identities;
- source commit/tree identities;
- network architecture/configuration;
- exact model bytes/hash;
- optimizer/checkpoint lineage;
- training counters/sample counts;
- production coverage evidence;
- R7.4 prerequisite provenance;
- explicit preservation of any still-open R7.3 exact-reproducibility debt;
- `ready_for_tables = false`.

R8 PASS requires R8.0 + R8.1 + R8.2 + R8.3 + R8.4 + R8.5. R8 PASS authorizes **R9 Strategic Audit only**.

## Non-negotiable fail-closed rules

R8 fails or remains blocked if any of the following occurs:

- R7.4 final PASS is absent;
- a production rule/economic parameter is guessed from pilot data;
- profile/domain/algorithm-seed identities are mixed;
- HU policy is used as 3H policy or vice versa;
- worker parallelism changes Algorithm-R, traversal RNG, minibatch RNG or sample weights;
- a restart advances scheduler progress without a durable matching stream checkpoint;
- integrated generation components are from different logical states;
- throughput calibration accepts a state different from serial reference;
- the selected R7/R7.4 strategic mechanism is silently changed during calibration;
- a production artifact cannot be tied to immutable model bytes and full production identity;
- any R8 stage claims `READY FOR TABLES`.

## Finite path after R8

```text
R8.0 exact profile/rules
-> R8.1 deterministic streams + durable transaction       [INFRA PASS]
-> R8.2 exact-profile Ryzen9 calibration                  [PRECOMMIT READY; NOT RUN]
-> R8.3 official HU training
-> R8.4 official 3H training
-> R8.5 freeze official policies
-> R9 strategic audit
-> R10 OpenHoldem runtime
-> R11 safe exploitation
-> R12 operational homologation
-> READY FOR TABLES
```
