# R8 Production Training — finite gate precommit 2026-08-12

`READY FOR TABLES = NO`.

This document freezes the **shape and sequencing** of R8 before any R8 production result is observed. It does not authorize R8 execution before `validation/R7_4_FINAL_GATE.json` reports `r7_4_pass = true` and `r7_4_ready_to_advance_to_r8 = true`.

The deferred R7.3 exact historical reproducibility issue remains a release debt. R8 may be engineered under the explicitly provisional strategy-quality path, but that debt cannot be silently converted to PASS and must still be closed or formally dispositioned before final table homologation.

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

No policy may be reused across a different tuple by fallback or convenience.

The strategy domains remain separate:

```text
TRUE_HEADS_UP
THREE_HANDED
```

R8 does not merge HU and 3H into one production policy artifact merely because they share the same encoder/network implementation.

## Frozen R8 sequence

### R8.0 — production profile / identity / rules

Before production training, materialize a machine-readable production profile describing the real target game configuration. At minimum it must identify:

- platform/game family and ruleset version/source;
- buy-in/currency and multiplier/economic profile identity;
- starting chip structure;
- blind/ante schedule and any level-dependent structure needed by the solver input;
- normalized payout shares / prize semantics needed by the exact ICM utility;
- TRUE_HEADS_UP and THREE_HANDED applicability;
- action abstraction identity;
- utility model identity;
- source/provenance timestamp and evidence.

**No GGPoker economic constant is to be invented from R7.4 pilot values.** The R7.4 `(1500 chips, 10/20, payout 0.5/0.3/0.2)` pilot configuration is validation evidence only unless independently proven to match a production profile.

R8.0 PASS requires complete, internally consistent, provenance-backed profiles. Missing or uncertain production rules fail closed; they are not filled with assumptions.

### R8.1 — deterministic production workers + central Algorithm-R reservoirs

Implement/validate the production data-generation architecture before scale-up:

- deterministic profile/domain assignment;
- central Algorithm-R reservoir semantics for Advantage and AveragePolicy samples;
- no duplicate weighting caused by worker count or completion order;
- deterministic checkpoint metadata and provenance;
- worker failure/restart without silent profile mixing;
- separate storage/identity for HU and 3H policies and for distinct economic profiles;
- parallel execution only where it is proven semantically neutral to the frozen learning/RNG contract.

#### Persistent-RNG constraint discovered before production

The selected R7.3/R7.4 mechanism explicitly freezes:

```text
primary RNG = one persistent live bundle.batch_rng in execution order
```

That RNG is consumed not only by Algorithm-R replacement, but also by traversal/action sampling and training minibatch sampling. Therefore a naive design that assigns independent per-root RNGs to parallel workers would **change the selected algorithm**, even if a central reservoir later merged the samples in root order.

`CentralAlgorithmRReservoirs` solves one necessary problem — worker completion order cannot change central Algorithm-R insertion/replacement order — but it does **not** by itself prove that traversal RNG consumption is equivalent to the frozen serial stream.

Consequently R8.1 must fail closed against naive root-level parallelism within a single `(profile, domain, algorithm-seed)` stream. Safe choices are, in priority order:

1. parallelize independent streams whose RNG histories are already independent by contract (for example different accepted profile/domain jobs), while preserving serial execution inside each stream;
2. implement a genuinely stream-preserving traversal dispatcher and prove exact equivalence against the persistent serial RNG contract before using it;
3. if no such speed-up has favorable complexity/throughput trade-off, keep root collection serial and optimize elsewhere.

CPU utilization is not an acceptance criterion. Strategic semantics and throughput per correct sample are the objective.

R8.1 is an infrastructure gate, not a strategic PASS.

### R8.2 — exact production profile + Ryzen9 calibration

For each accepted production profile/domain pair, calibrate throughput and resource use on the intended Ryzen9 execution path before committing a long official run.

Calibration chooses operational batch/worker/chunk sizes for throughput and memory safety only. It may not change the frozen strategic mechanism or relax strategic gates. Calibration evidence must record wall-clock throughput, sample throughput, memory usage, checkpoint cost, worker count, and any CPU/thread configuration that materially changes execution.

No long production run starts until the exact R8.0 profile identity and R8.1 reservoir/worker semantics are proven.

### R8.3 — official TRUE_HEADS_UP production training

Train the official HU policy for each accepted production profile using the selected R7 mechanism carried through R7.4:

```text
behavior semantic: SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1
ensemble size: 4
epsilon scale: 1.75
epsilon cap: 0.50
partial-exact opponent levels: 2
primary RNG: one persistent live batch RNG in execution order
utility: exact explicit-payout ICM delta using normalized payout shares for the accepted production profile
action abstraction: frozen profile identity
```

Production training must cover the accepted production stack/blind/profile support rather than only the finite R7.4 pilot scenarios.

R8.3 does not authorize table use; it produces a candidate official HU artifact plus training/checkpoint evidence for R9 audit.

### R8.4 — official THREE_HANDED production training

Train the official 3H policy under the same production identity discipline and the accepted 3H scenario/profile support.

The 3H artifact is independent of the HU artifact. A successful HU production run cannot substitute for an absent or failed 3H run.

R8.4 does not authorize table use; it produces a candidate official 3H artifact plus evidence for R9 audit.

### R8.5 — freeze official production policies

Freeze every official production policy artifact with immutable provenance sufficient for R9/R10 to verify exactly what is being audited and later loaded by OpenHoldem.

At minimum the freeze must bind:

- production profile identity;
- domain (`TRUE_HEADS_UP` or `THREE_HANDED`);
- ruleset/action-abstraction/utility/learning identities;
- source commit/tree identities;
- network architecture/configuration;
- exact model bytes/hash;
- relevant optimizer/checkpoint lineage;
- training counters/sample counts;
- production scenario/profile coverage evidence;
- R7.4 prerequisite provenance;
- explicit preservation of any still-open R7.3 exact-reproducibility debt;
- `ready_for_tables = false`.

R8 PASS requires R8.0 + R8.1 + R8.2 + R8.3 + R8.4 + R8.5. R8 PASS authorizes **R9 Strategic Audit only**.

## Non-negotiable fail-closed rules

R8 fails or remains blocked if any of the following occurs:

- R7.4 final PASS is absent;
- a production rule/economic parameter is guessed from pilot data;
- profile/domain identities are mixed;
- HU policy is used as 3H policy or vice versa;
- worker parallelism changes Algorithm-R sampling semantics, traversal RNG semantics, minibatch RNG semantics, or sample weights;
- a restart/checkpoint loses deterministic identity/provenance;
- the selected R7/R7.4 strategic mechanism is silently changed during throughput calibration;
- a production policy artifact cannot be tied to immutable model bytes and a complete production-profile identity;
- any stage claims `READY FOR TABLES` before R12.

## Finite path after R8

```text
R8.0 profile/rules
-> R8.1 deterministic worker + central reservoir semantics
-> R8.2 exact-profile Ryzen9 calibration
-> R8.3 official HU training
-> R8.4 official 3H training
-> R8.5 freeze official policies
-> R9 strategic audit
-> R10 OpenHoldem runtime
-> R11 safe exploitation
-> R12 operational homologation
-> READY FOR TABLES
```
