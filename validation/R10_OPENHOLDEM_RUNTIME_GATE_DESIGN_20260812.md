# R10 OpenHoldem runtime integration — finite gate precommit

`READY FOR TABLES = NO`

R10 begins only after `R9` has accepted the immutable official production policy artifacts. R10 integrates those already-audited artifacts with OpenHoldem. It is an execution/identity/routing layer, **not a second strategic engine** and not a place to retune poker decisions.

R10 PASS authorizes **R11 Safe Exploitation only**. It never authorizes table use.

## Core runtime principle

The OpenHoldem bridge must be deterministic and fail closed:

```text
scraped/runtime state
-> explicit production-profile/domain identity
-> exact frozen observation/action mapping
-> exact R9-approved model artifact
-> six-action policy inference
-> legal-action validation
-> OpenHoldem action command
```

There is no strategic fallback range, no silent model substitution, and no conversion of an invalid state into a plausible-looking poker action.

## Finite R10 sequence

### R10.0 — bridge ABI / deployment identity

Freeze the bridge interface and deployment manifest before table homologation work. At minimum bind:

- OpenHoldem version/build identity supported by the bridge;
- target OS/architecture;
- bridge DLL/user-DLL binary SHA-256;
- model-runtime dependencies and exact versions;
- policy-manifest schema/version;
- supported production profile IDs;
- exact HU and 3H policy hashes for every supported profile;
- action-abstraction identity;
- observation encoder identity;
- explicit `ready_for_tables = false`.

Loading an unsupported OpenHoldem/runtime/manifest version must fail closed.

### R10.1 — state acquisition and canonical runtime snapshot

Define one canonical runtime snapshot consumed by policy inference. The bridge must distinguish **observed inputs** from **derived values** and record enough identity to reproduce every decision offline.

The snapshot must include, where applicable to the production profile and encoder:

- hero hole cards;
- public board cards;
- seat occupancy / live-player set;
- hero seat / button or dealer position required by rules;
- per-seat stacks;
- pot;
- blinds/ante and blind-level identity;
- current bets / amount to call;
- betting street/round and action history required to recover the canonical solver state;
- tournament/profile/multiplier identity required for payout/ICM semantics;
- decision timestamp/hand identifier for audit linkage.

No field may be silently defaulted when it changes the neural observation, legal actions, domain, production profile or utility semantics.

### R10.2 — profile and domain router

The router must select **exactly one** accepted policy identity from explicit runtime facts.

Required invariants:

- 2 live players -> only a policy whose domain is `TRUE_HEADS_UP`;
- 3 live players -> only a policy whose domain is `THREE_HANDED`;
- no HU-as-3H or 3H-as-HU fallback;
- exact production profile ID must match the runtime economic/structure state;
- unsupported/missing/ambiguous profile -> no strategic action;
- policy bytes loaded must hash to the R9-approved artifact hash;
- a policy manifest may not alias two distinct production identities to one model unless that identity equivalence was explicitly audited in R9.

Routing failures are scrape/runtime errors, not poker decisions.

### R10.3 — observation and legal-action equivalence

For deterministic fixtures spanning the accepted production support, prove that the runtime bridge produces exactly the same:

- canonical state identity;
- SPNNIV1 neural observation bytes;
- observation SHA-256;
- legal-action mask;
- six-action action IDs;

as the offline audited SpinCore path.

The comparison is exact byte/action identity. R10 may not introduce a second independently maintained encoder or action mapping whose outputs are merely approximately equivalent.

### R10.4 — inference equivalence

For immutable R9 fixtures, prove that OpenHoldem runtime inference returns the same six-action probabilities/action fingerprint as the offline audited artifact under the frozen inference runtime.

Use the exact action-sentinel identity machinery where applicable. Model/runtime/observation mismatch must be detected before an action is emitted.

No randomized action sampler may consume an untracked RNG. If the production execution policy samples from a mixed strategy, the RNG algorithm, seed/state lifecycle, checkpoint/audit logging and mapping from policy probabilities to the sampled action must be explicitly frozen and reproducible. If execution is deterministic instead, that deterministic selection rule must be frozen and audited before use.

### R10.5 — action translation to OpenHoldem

Map each frozen abstract action ID to the exact OpenHoldem command/bet amount semantics for the current legal state.

The translator must prove:

- illegal abstract action cannot be emitted;
- bet/raise amount matches the frozen action abstraction after integer/chip normalization;
- all-in semantics are exact and capped by available stack;
- call/check/fold distinctions are state-correct;
- no hidden minimum-raise or chip-rounding behavior changes the intended abstract action;
- impossible/unrepresentable abstract action fails closed rather than substituting another strategic action.

Round-trip fixtures must verify `offline abstract action -> OpenHoldem command -> reconstructed abstract action` identity wherever a round trip is semantically defined.

### R10.6 — scrape/runtime safety barrier

Before policy inference or action emission, enforce a fail-closed validity barrier. At minimum reject:

- missing/invalid hero cards;
- duplicate/impossible cards;
- unsupported live-player count;
- inconsistent stacks/bets/pot;
- invalid blind/ante/profile identity;
- impossible amount-to-call/current-bet relations;
- action-history/canonical-state reconstruction failure;
- model/profile/domain hash mismatch;
- unsupported or ambiguous tournament/multiplier state;
- stale/cross-hand state identity;
- any runtime exception in encoder, router, model or action translator.

The safe result of a barrier failure is **no bot strategic action** plus diagnostic evidence. R10 itself does not decide how the operator/application should recover; that behavior is tested in R12 operational homologation.

### R10.7 — decision audit record

Every candidate runtime decision must be reproducible offline from an append-only audit record containing at least:

```text
hand / decision identity
bridge binary hash
runtime/model manifest hash
production profile ID
domain
policy artifact hash
canonical input/snapshot hash
observation SHA-256
legal-action mask
six-action policy probabilities
selected abstract action
selection RNG state/provenance if stochastic
translated OpenHoldem command/amount
barrier status / error reason
timing/latency telemetry
```

Sensitive/raw data retention can be operationally minimized, but hashes/identities must be sufficient to prove which exact state/model/action path executed.

### R10.8 — deterministic replay and fault injection

Run offline/integration fixtures covering normal and failure paths. Required fault classes include:

- wrong/missing model bytes;
- wrong profile manifest;
- HU/3H routing mismatch;
- corrupted observation/state;
- illegal action mask;
- stale hand state;
- invalid cards/stacks/pot/bets;
- unsupported multiplier/profile;
- model inference exception;
- OpenHoldem action translation mismatch.

Each injected fault must fail closed and produce the expected diagnostic class, never a substitute poker action.

### R10.9 — finite R10 gate

Materialize one machine-readable R10 final gate binding the exact bridge binary, OpenHoldem compatibility target, policy manifest and all R10.0–R10.8 evidence.

R10 PASS requires all supported production profile/domain policy artifacts to pass all applicable runtime tests. One passing policy/profile cannot mask another failed route.

The final gate must state at least:

```text
r10_pass
r10_ready_to_advance_to_r11
bridge_binary_sha256
openholdem_runtime_identity
all_r9_approved_policy_hashes_bound
profile_domain_routing_pass
observation_action_equivalence_pass
inference_equivalence_pass
action_translation_pass
runtime_safety_barrier_pass
deterministic_replay_fault_injection_pass
decision_audit_record_pass
ready_for_tables = false
```

## Non-negotiable anti-fallback rules

R10 remains FAIL/BLOCKED if any of the following occurs:

- R9 PASS is absent;
- the bridge contains a strategic range/model not present in the audited artifacts;
- HU and 3H can silently substitute for each other;
- one production profile can silently substitute for another;
- missing economic/tournament identity is filled with a default profile;
- observation/action mapping differs from the offline audited SpinCore path;
- model hash is not checked against the accepted manifest;
- an illegal or unrepresentable action is silently replaced by a different poker action;
- scrape/state validation error still permits strategic action emission;
- mixed-strategy RNG is untracked/non-reproducible;
- replay cannot identify the exact model/state/action executed;
- any R10 artifact claims table readiness.

## Finite transition

```text
R9.6 strategic audit PASS
-> R10.0 bridge ABI/deployment identity
-> R10.1 canonical runtime snapshot
-> R10.2 profile/domain router
-> R10.3 observation/action equivalence
-> R10.4 inference equivalence
-> R10.5 OpenHoldem action translation
-> R10.6 scrape/runtime safety barrier
-> R10.7 decision audit record
-> R10.8 deterministic replay + fault injection
-> R10.9 final R10 gate
-> R11 Safe Exploitation
```

No R10 step may redefine the strategy. Any strategic-model change returns to an explicit R8/R9 cycle with a new artifact identity.
