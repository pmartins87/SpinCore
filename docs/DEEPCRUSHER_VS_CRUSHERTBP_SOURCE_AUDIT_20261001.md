# DeepCrusher vs CrusherTBP — Source & Infrastructure Audit

Date: 2026-10-01  
Status: **BLOCKED_SOURCE_FREEZE / NO STRENGTH BENCHMARK AUTHORIZED**

## Purpose

Establish the exact executable sources, runtime semantics, reusable benchmark infrastructure, and fail-closed blockers for a scientific DeepCrusher-vs-CrusherTBP comparison.

This document is descriptive. It does not select a winner and does not authorize a large simulation.

## 1. Source freeze status

### DeepCrusher

The repository still contains a reproducibly pinned historical benchmark baseline:

- branch: `r8-v22-stable-20260914`;
- operational source: `DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_ASCII_20260914.txt`;
- operational SHA256: `0113badc99727a7dd47c02448d4d042b5e008534cd63fd79a461a72b24eeb68d`;
- strategic recovered source: `DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_RECOVERED_20260914.txt`;
- strategic SHA256: `9fc2d00aacc915f3c265429f764056f3c6270df616244026aac22e455c803ee9`.

Those pins were created for the SpinCore external benchmark. They are **not automatically the canonical DeepCrusher version for the new DeepCrusher-vs-CrusherTBP benchmark**.

Later DeepCrusher work exists outside the old R8 pin. Until the latest corrected operational artifact is copied into a versioned repository path and its bytes/hash are frozen, DeepCrusher source selection for this new benchmark remains **BLOCKED**.

Rule: version names such as R8/R10/R10B/R10c are not sufficient identity. Benchmark identity is immutable bytes + SHA256 + repository commit.

### CrusherTBP

Exact candidate currently available to the audit:

- source: `CrusherTBP(2).txt`;
- size: 505,030 bytes;
- SHA256: `f164207d3b5eaad3f47137e75ea6d4245b76fa78888a897fc9fd67aa858aa000`.

A same-sized Library artifact named `CrusherTBP.txt` also exists, but same size is not proof of byte identity. The two must be reconciled before canonical freeze.

Static scan of the exact candidate plus the integrated OpenPPL library found:

- 1,252 strategy sections;
- 897 library sections;
- 2 strategy/library section-name collisions, both stack/chip helper overrides;
- 477 transitively reachable functions from `f$preflop/f$flop/f$turn/f$river`;
- zero missing reachable `f$` functions after library overlay;
- substantial external-state surface: PT statistics, colour codes, network flags, named-chair lookups, log flags, user variables and memory symbols.

This is dependency closure only; it is not runtime parity.

### OpenPPL / OpenHoldem reference assets

Current exact audit inputs:

- OpenHoldem source dump: SHA256 `8a2809bf32b226775a237c9a51f970e8fd55148e777890f9a275b5fd6bd8521e`;
- manuals/library archive: SHA256 `2aea57b284d214be21a4d6fa6d1283a5772cf1c672e3c787c5a66348d6b14411`;
- integrated OpenPPL library extracted from that archive: SHA256 `eeb0fe6a842e7a6381f0bad31bd216763075da5c2f35d3a20f0b19588b2c0340`;
- observed runtime family in project logs: OpenHoldem 14.0.2.0, build marker `05fae382d72c5b9133fe3b8271782a82`.

The source/library hashes must be copied into the final benchmark manifest rather than inferred from filenames.

## 2. Infrastructure that can be reused

The SpinCore DeepCrusher benchmark line contains valuable generic components:

### Reuse directly after policy-ID generalization

- ordered OpenPPL expression / WHEN / SET execution;
- hand-list parser and hand-class support;
- exact observable state and exact-suit deal snapshot;
- OpenHoldem card/hand/evaluator symbols already closed for Hold'em;
- public-action transcript and history reconstruction;
- exact solver action ABI: Fold / Check / Call / BetTo / RaiseTo / AllIn;
- HU paired same-deal seat swap;
- 3H six-game AAB/ABB exposure-balanced rotation;
- per-row zero-sum assertions;
- decision-level trace and pathology surfaces;
- deterministic scenario/deal identity generation;
- multiprocess discipline using compact inference/runtime assets.

### Reuse only after modification/revalidation

- lifecycle replay: must support every reserved callback actually used by each source, including `f$ini_function_on_connection` when present;
- native/environment provider: the old R8 `GGPoker_NoPT_NoNotes_V1` profile is a candidate controlled environment, not automatically a valid new benchmark environment;
- preflop equity/versus multiplex support: old R8 range IDs and fixtures cannot be assumed to cover CrusherTBP or later DeepCrusher;
- R8-specific source-closure and source-pin modules: must become policy-generic;
- OpenHoldem action translator: see hard blockers below;
- OpenHoldem parity fixtures: must be regenerated for **both** benchmark policies.

### Do not reuse as evidence

- any previous SpinCore-vs-DeepCrusher EV result;
- any R8-v22 source pin as proof that the latest DeepCrusher is frozen;
- SpinCore's seven-action policy abstraction for either OpenPPL strategy;
- development seeds, inspected pathology samples, or old benchmark outcomes as canonical holdout evidence.

## 3. Hard semantic blockers discovered in this audit

### A-001 — missing autoplayer all-in-adjustment parity

The current portable DeepCrusher action bridge translates OpenPPL decisions to exact simulator actions, but it does not yet reproduce the complete OpenHoldem autoplayer conversion through `f$allin_on_betsize_balance_ratio`.

OpenHoldem evaluates this callback against the planned bet/raise and may replace an otherwise ordinary f$betsize or bet-pot action with all-in before execution.

CrusherTBP has its own executable `f$allin_on_betsize_balance_ratio`, so omitting this layer can change final actions materially.

**Disposition: HARD BLOCKER.**

### A-002 — exact technical bet-pot factors

OpenHoldem technical bet-pot buttons use their own literal factors, including 0.333 and 0.667. The current portable map uses mathematical 1/3 and 2/3 for named pot-size actions.

These are normally close but are not byte/mechanically equivalent once chip rounding, minimum raises, maximum raises, or the all-in-adjustment threshold is involved.

**Disposition: HARD BLOCKER until the action backend distinguishes exact OpenHoldem technical-button factors from arbitrary percentage-pot numeric decisions.**

### L-001 — connection lifecycle

CrusherTBP contains connection-scoped initialization. The prior R8 benchmark policy replays startup, hand reset, new round and my turn, but was not designed around a CrusherTBP `f$ini_function_on_connection` requirement.

**Disposition: HARD BLOCKER until actual OpenHoldem callback order/lifetime is frozen and parity-tested.**

### S-001 — DeepCrusher current-version ambiguity

The old R8-v22 benchmark source is exact, but later DeepCrusher development exists and the newest corrected operational artifact has not yet been pinned into GitHub with an exact hash.

**Disposition: HARD BLOCKER. Do not choose an older version merely because it is easier to reproduce.**

### S-002 — CrusherTBP duplicate identity

`CrusherTBP(2).txt` is hashable and exact, while a same-sized Library artifact `CrusherTBP.txt` exists without raw-byte materialization in this environment.

**Disposition: BLOCKER to canonical source freeze, not to continued engineering.**

## 4. Controlled environment recommendation

Primary comparison should target the intrinsic/readless strategy under one identical externally observable environment:

- GGPoker network flag true;
- other network flags false;
- no private PokerTracker history;
- no manual player colour notes;
- named-chair lookups unresolved;
- logging symbols set according to OpenHoldem semantics;
- card/equity/history symbols still computed faithfully, not zero-filled.

Reason: private opponent-history databases are asymmetric information, not strategy strength under a controlled contest.

This environment is **PROPOSED, NOT YET FROZEN**. CrusherTBP uses negative PT/exploit conditions, so disconnected/unknown values are behaviorally significant. The exact profile becomes canonical only after parity probes against real OpenHoldem under the same conditions.

A later operational/exploit benchmark may be defined separately if equivalent opponent-profile inputs can be supplied to both policies. It must not be mixed into the primary readless result.

## 5. Admission rule

No DeepCrusher-vs-CrusherTBP EV run is scientifically valid until:

1. both source identities are frozen;
2. every reachable semantic dependency is explicit;
3. final OpenHoldem action/sizing behavior is reproduced, including all-in adjustment;
4. state/session lifetime is reproduced;
5. both policies independently pass real OpenHoldem parity fixtures.

Until then, engineering smoke tests are allowed only when labelled mechanical/development evidence and may not be interpreted as strategy strength.
