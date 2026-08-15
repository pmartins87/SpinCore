# R7.5 Representation Integrity Audit — Addendum 2026-08-15

## Status

**R8 production representation freeze remains BLOCKED.**

`READY FOR TABLES = NO`.

This addendum records findings and remediations discovered after `R7_5_REPRESENTATION_INTEGRITY_AUDIT_20260814.md`. It does not overwrite historical V1/V2 evidence and does not promote a representation before the precommitted candidate-specific Phase 2 end-to-end gate.

## Newly confirmed defects / limitations

### A12 — Legal public histories can exceed the historical 32-event neural window

Confirmed by a reachable legal betting-engine regression. The old fixed-length V1/V2 neural history therefore performs real strategic truncation, not merely theoretical defensive clipping.

**Remediation:** SPNNIV3 serializes the complete public event vector with variable length. Network batching may pad only to the batch-local maximum and must carry exact lengths; reaching any future protocol sanity cap in a legal production state is a hard error, never silent truncation.

### A13 — Historical six-slot V1 action abstraction can contain exact-action aliases

Different nominal aggressive slots can clamp to the same exact `BetTo/RaiseTo/AllIn` in shallow-stack states. The old `legal_abstract_actions()` path does not deduplicate those aliases.

**Boundary:** the R7.5.4 universal V2 resolver already deduplicates state-local exact aliases and its collector explicitly uses the deduplicated legal set, so this defect is historical V1/action-control debt and does not invalidate the immutable R7.5.4A sizing run.

### A14 — Old V2 wheel-edge straight-draw label was wrong for 2345

The old open-ended detector started at low rank 3, so 2345 could be classified as a double-gutshot instead of an open-ended draw with A/6 completion ranks.

**Remediation:** old V2 helper fixed and regression added. More importantly, H3 final no longer relies on subjective OESD/gutshot labels as primary information: SPNNIV3 objective semantics counts exact one-card completion cards and distinct completion ranks.

### A15 — A helper named `stack_geometry_v3.py` still consumed V1 layout

This was not on the active H3 semantic path, which had an internal SPNNIV3 pairwise implementation, but the filename/API was an integration trap.

**Remediation:** helper and tests now consume `DecodedInputV3` only; true-HU absent seat, real all-in opponent, folded opponent and two independent 3H opponent geometries are covered.

### A16 — Post-hoc V1 card canonicalization is not a valid test of end-to-end symmetry quality

R7.5.3B canonicalized candidate inputs after the paired corpus had already been generated under V1 continuation behavior. Symmetry-breaking V1 continuation policies can generate inconsistent labels across states that a correct invariant representation must merge.

**Consequence:** R7.5.3B remains valid as a V1-imitation diagnostic but has no authority to reject end-to-end invariant H2/H3.

## Newly closed structural items

### SPNNIV3 carrier

Implemented and cross-language parity tested:

- exact visible ranks 2..14;
- no physical suit IDs;
- all 21 pairwise same-suit relations over seven card-role slots;
- Hero/private vs public role retained;
- flop/turn/river chronology retained;
- exact current BB-normalized chip geometry;
- primitive exact legal-action mask;
- complete variable-length structured public history;
- true-HU canonical seats `[Hero, live opponent, absent]` while preserving dealer/SB/BB relations.

### Exact card orbit canonicalization

A production-candidate canonical card key now minimizes over exactly `2! x 3!` allowed order permutations:

- swap of Hero's two private cards;
- six permutations of the simultaneously dealt flop.

Turn and river are never permuted. Global physical suit renaming is already quotiented by SPNNIV3's label-free same-suit relation.

Mandatory exhaustive regression enumerates all `C(52,3)=22,100` physical flops and requires exactly **1,755** rank+suit-isomorphism classes. This is the lossless flop structural count; the 53 strategic flop clusters remain auxiliary semantics only and never replace exact state.

### H2/H3 final neural path

Versioned final candidate networks now exist in parallel with old models:

- **H2 final:** exact card orbit + exact current state + complete structured history;
- **H3 final:** H2 + objective poker semantics derived from the same lossless carrier.

Network invariance is structural, not learned:

- card orbit is canonical before rank embedding;
- rank embedding domain is exactly `0..14`; no K/A clamp alias;
- physical suit labels never enter the network;
- GRU uses exact packed lengths;
- zero-event history explicitly zeros the recurrent output;
- right padding is therefore neutral;
- histories longer than 32 are accepted.

### Objective H3 semantics

Current semantic channels include:

- board multiplicity/suit/straight-window texture;
- exact made category and board-only category;
- minimum/maximum number of Hero hole cards contributing to an equally best five-card hand;
- objective rank/multiplicity/overcard relations;
- exact one-card straight/flush completion counts and distinct straight-completion ranks;
- board-only completion counts to distinguish Hero-created draws from public-board draws;
- backdoor/combo/pair+draw facts;
- public preflop/postflop lineage reconstructed from exact event commitments;
- pairwise opponent presence/contesting/actionability, effective remaining stack, SPR, total cap and commitment gap.

Adversarial tests cover all nine made-hand categories, board-playing hands, 0/1/2-hole contribution, wheel/OESD/gutshot, Hero flush draw vs board-only four-flush, suit/hole/flop invariance, preflop all-in-call vs all-in-raise semantics, HU dead-seat handling and 3H pairwise stacks.

A separate physical-deck audit now samples flop/turn states with real physical suit labels, converts them to SPNNIV3 relational form, and compares H3 straight/flush completion counts against independent brute-force enumeration of the physically unseen deck.

### Solver/runtime identity path

The authoritative C++ C API already exposes variable-length `spincore_solver_state_neural_input_v3`. A versioned Python bridge now reads that exact payload; no Python-side reconstruction is required for candidate training. Cross-language byte/semantic parity remains mandatory.

### Universal CFR integration

H2/H3 final are wired into the already-audited universal action collector by overriding only the observation source/model. Exact-action resolution and state-local alias deduplication remain the same as R7.5.4. A mechanical CI smoke test exercises:

`C++ state -> universal CFR collection -> SPNNIV3 reservoir samples -> Advantage training -> AveragePolicy training`.

This smoke test is an integration test only and has no strategic selection authority.

## Phase 1 interpretation

The immutable Phase 1 diagnostic completed successfully. H3 was the best invariant candidate at reproducing V1-generated targets, but H0 remained better on those historical labels. This is expected to remain secondary evidence because the target policy itself is V1.

The important Phase 1 engineering result is that the earlier V2 defeat cannot be attributed simply to "semantic networks are too heavy": the H1/H2/H3 prototypes remained within a modest CPU/RAM overhead while H3 recovered a meaningful fraction of H1/H2's V1-imitation gap.

## Remaining blocking items before Phase 2 strategic evidence

1. Current `main` regression must be green with the exhaustive 1,755-orbit test, physical-deck outs test, final H2/H3 model tests, Solver SPNNIV3 bridge and universal-CFR smoke test.
2. Freeze exact H2/H3 final model schema/hash and measured parameter counts.
3. Run a non-selecting runtime/resource preflight on actual solver observations.
4. Execute the already-precommitted Phase 2 schedule independently for H2 and H3; each candidate must generate its own continuation policy and targets.
5. Evaluate with held-out seeds/common reference/cross-play and uncertainty; do not call a proxy "exact exploitability".
6. Only after H3 structural/resource admission may H4 capacity reinvestment run under the precommitted Phase 3 envelope.
7. If the final winner is not V1, representation-dependent action-sizing/stability evidence must be rerun or formally bridged before R8.

`READY FOR TABLES = NO`.
