# R7.5 Representation Integrity Audit — 2026-08-14

## Status

**BLOCKING FOR R8 PRODUCTION REPRESENTATION FREEZE**

`READY FOR TABLES = NO`.

This audit was opened after a review of card-role and symmetry semantics exposed that the frozen V1 neural representation preserves strategically meaningless permutations and absolute suit labels. The purpose of this document is not to infer that the whole engine is wrong. It separates confirmed defects, disproved concerns, and still-open questions, and defines the revalidation boundary before R8.

## Scope reviewed so far

Path under review:

`game state -> topology/hand engine -> canonical infoset -> neural encoder -> neural network -> Deep CFR traversal/targets -> action abstraction -> eventual runtime reconstruction`

Files directly inspected include:

- `src/hand_engine.cpp`
- `src/betting_engine.cpp`
- `src/hand_infoset_adapter.cpp`
- `src/neural_encoder.cpp`
- `src/neural_encoder_v2.cpp`
- `src/card_semantics_v2.cpp`
- `src/scenario_sampler.cpp`
- `src/tournament_value.cpp`
- `python/spincore/deep_cfr.py`
- `python/spincore_nn/codec.py`
- `python/spincore_nn/models.py`
- `python/spincore_nn/codec_v2.py`
- `python/spincore_nn/models_v2.py`
- `tools/r7_5_paired_corpus_worker.py`
- the frozen R7.5.3 representation-ablation contract/result
- the immutable R7.5.3B card-symmetry experiment and logs

This is a living audit. A finding is not closed by intuition; it is closed by code inspection plus deterministic/property tests where appropriate.

---

## A. Confirmed findings

### A1 — V1 preserves private-card order, flop order and absolute suit names

Severity: **material sample-efficiency / inductive-bias defect**.

V1 places raw physical card IDs in seven fixed slots. Hero-vs-board roles are preserved correctly, but the encoder treats strategically equivalent presentations as different inputs:

- swapping Hero's two private cards;
- permuting the three simultaneously revealed flop cards;
- globally renaming suits while preserving all suit relations.

The roles `[hole,hole | flop,flop,flop | turn | river]` are not confused; the defect is failure to quotient true game symmetries.

Required remediation: representation-level invariance, not post-hoc relabeling alone. Candidate card encodings must preserve exact ranks, public/private roles, street chronology and all same-suit relations while eliminating absolute suit labels and unordered-card permutations.

### A2 — V1 history GRU consumes trailing padding as real recurrent timesteps

Severity: **material neural-history defect**.

The C++ wire supplies `history_len`; Python `codec.py` exposes it; frozen `models.py` ignores it and runs the GRU over all 32 timesteps. Padding token 0 has a zero embedding, but a GRU still updates hidden state through recurrent weights/biases. Therefore the final hidden state can depend on the amount of trailing padding rather than solely on the real action sequence.

Required remediation: use packed sequences/masking so trailing padding has exactly zero semantic effect. Add a property test asserting output identity for the same real history under different right-padding capacity.

### A3 — V1 public-history tokens are materially lossy

Severity: **material state-aliasing defect**.

V1 history tokens encode only action type + street. They omit at least:

- actor;
- forced-vs-voluntary status;
- amount/size;
- pot-before/pot-after;
- resulting commitment.

Current numeric state preserves the current bet/to-call/commitments, but earlier action provenance and sizing can be lost. Distinct histories such as different preflop raise sizes or c-bet vs donk/probe can therefore alias after their immediate numeric consequences are no longer sufficient to reconstruct the line.

Required remediation: structured public-event sequence with actor-relative seat, street, action, forced flag, and normalized sizing/pot fields.

### A4 — True-HU observation contains arbitrary dead-seat identity

Severity: **material redundancy / symmetry defect**.

The sampler randomizes which original 3-max seat is dead in true HU. `build_current_actor_infoset` rotates all three physical slots actor-relative. Consequently the one live opponent can occupy relative slot 1 or 2 solely because a strategically irrelevant empty chair had a different physical index.

Required remediation: in true-HU domain, canonicalize relative seats to `[Hero, live opponent, absent]` while separately and losslessly preserving dealer/SB/BB roles. Do **not** apply this collapse to 3H-origin states after a fold, where original position/history remains strategically meaningful.

### A5 — Old V2 is semantic-rich but lossy; it is not `exact + semantics`

Severity: **architecture-comparison validity defect**.

`NeuralInputV2` contains a 169-class preflop token, a flop abstraction/signature, semantic numeric/categorical fields and structured history, but it does not preserve a complete exact Hero+board card representation through turn and river. Strategically distinct exact states can therefore share the same V2 summary.

Consequence: the old R7.5.3 result `C0 V1 > V2` does **not** answer whether a lossless hybrid `exact cards + semantics + structured history` is better than V1.

Required remediation: every new semantic candidate must retain a lossless exact observable card state up to explicitly proven game symmetries.

### A6 — Old V2 leaks absolute suit identity

Severity: **material invariance defect**.

V2 categorical field 33 encodes `flush_draw_suit + 1`, i.e. the physical suit label. Hearts and spades can become different neural categories despite global suit renaming being strategically meaningless.

Required remediation: remove physical suit IDs from semantic fields. Encode nutness/count/relations only.

### A7 — Old V2 contains private-card-order-dependent suit features

Severity: **material invariance defect**.

V2 categorical fields 65 and 66 separately count board cards matching `hole[0].suit` and `hole[1].suit`. No lossless canonical ordering of the two private cards precedes these fields. Swapping the two private cards can therefore change the observation.

Required remediation: replace ordered hole-specific suit counters with permutation-invariant relational features.

### A8 — Old V2 true-HU effective-stack/SPR can be zeroed by the dead seat

Severity: **high / semantic corruption**.

Zero-stack seats are initialized `all_in=true`. The actor-relative infoset exposes status 2 for that seat. V2's minimum-opponent-stack calculation excludes only folded status 1, so the absent HU seat can be treated as a live zero-stack opponent. This can force:

- minimum live opponent stack = 0;
- effective remaining stack = 0;
- derived SPR = 0;

while Hero and the real opponent still have chips.

Required remediation: introduce an explicit seat-presence/actionability notion and compute pairwise effective stacks correctly. In 3H, avoid reducing multiway stack geometry to one ambiguous minimum when pairwise values matter.

### A9 — Old V2 was capacity-confounded relative to V1

Severity: **experimental-design limitation**.

The old V1 body is approximately `320 -> 128`; old V2 was compressed to approximately `192 -> 96` so its larger semantic input/embedding surface would remain near the same total parameter count. Thus V2 paid for richer inputs by reducing nonlinear core capacity.

Required remediation: future ablation must answer two separate questions:

1. representation quality with comparable core capacity;
2. best achievable quality within the frozen Ryzen compute/RAM envelope.

Equal total parameter count alone is not a sufficient fairness criterion.

### A10 — R7.5.3 / R7.5.3B paired corpus is generated under V1 behavior

Severity: **experimental-interpretation limitation**.

`tools/r7_5_paired_corpus_worker.py` bootstraps the accepted V1 behavior policy and records paired V1/V2 observations at the same sampled states. Candidate inference is explicitly not used during paired collection.

Deep-CFR advantage/strategy targets therefore reflect V1 continuation behavior. A representation that enforces symmetries absent from V1 can be penalized by contradictory V1-generated labels after post-hoc state merging.

Consequence: paired offline fitting is useful as a regression/retention diagnostic, but it is **not sufficient to select a new end-to-end representation**.

Required remediation: final selection requires a small precommitted candidate-specific end-to-end Deep-CFR pilot in which each representation participates in generating its own continuation policy/targets.

### A11 — R7.5.3B scientific result completed; persistence failed only

Severity: **infrastructure debt, not scientific failure**.

Immutable run `31828057750` completed gate, implementation tests, six paired fits and aggregation successfully. Aggregate decision retained S0/V1 raw because S1 post-hoc card canonicalization materially worsened worst-domain advantage heldout NRMSE (about 0.6149 -> 0.6571), outside the frozen non-inferiority band.

The workflow top-level failure occurred later because `.gitignore` ignores `validation/*.json`, so normal `git add` refused the generated result file.

Interpretation: this proves S1 was worse at reproducing the V1-generated paired corpus under the frozen V1 architecture. It does **not** prove that end-to-end suit/order invariance is strategically inferior.

---

## B. Concerns checked and not confirmed as defects

### B1 — Hero private cards confused with board cards: NOT CONFIRMED

V1 uses fixed card roles `[hole0,hole1,flop0,flop1,flop2,turn,river]`; card embeddings are flattened, not pooled across all seven slots. Hero/private-vs-public role is therefore structurally distinguishable.

### B2 — Dead HU seat receives hole cards / removes cards from deck: NOT CONFIRMED

`HandEngine` deals only to topology live seats. The absent seat does not consume two cards.

### B3 — Phantom three-way ICM after entering true HU: NOT CONFIRMED

Tournament continuation logic locks eliminated placements and values the surviving players with remaining stacks/prizes. No obvious recalculation as a three-live-player state was found in the reviewed path.

### B4 — V2 reports post-river draws: NOT CONFIRMED

Reviewed draw semantics are street-conditioned; ordinary draws are not treated as live river draws and backdoors are flop-only in the inspected implementation.

### B5 — V2 forced-action flag cannot distinguish blinds: NOT CONFIRMED

Betting-engine blind events are recorded with `forced=true`; ordinary actions are recorded `forced=false`; V2 history carries that bit.

### B6 — Core external-sampling Deep-CFR traversal pattern grossly inverted: NOT CONFIRMED

The reviewed implementation enumerates legal actions at traverser nodes and samples opponent actions in the expected external-sampling pattern, with separate strategy-memory collection. This is a high-level structural audit only, not a complete proof of every estimator/weighting detail.

---

## C. Open audit items before representation freeze

The following must be closed by deterministic tests or explicit review before R8 production representation can be frozen:

1. exact train/runtime observation parity for the eventual winner;
2. complete metamorphic invariance suite over cards/seats/history;
3. villain private-card non-leakage under every decision path;
4. no accidental aliasing of turn vs river or private vs public roles;
5. multiway pairwise stack/effective-stack/SPR semantics;
6. preflop lineage/aggressor/c-bet/donk/probe derivation on edge cases;
7. public-history truncation semantics and sufficiency;
8. exact C++/Python serialization parity for all new fields;
9. action-mask/abstract-action identity after representation changes;
10. Deep-CFR weighting/reach/reservoir equations against deterministic toy games/reference calculations;
11. network output invariance under all declared game symmetries;
12. sensitivity/non-invariance tests showing strategically meaningful perturbations do change the representation;
13. runtime/RAM/throughput feasibility on the Ryzen envelope;
14. end-to-end candidate-specific learning/convergence, not only offline imitation of V1 targets.

---

## D. Mandatory metamorphic test matrix for R7.5.3C

A candidate cannot enter candidate-specific training unless all applicable properties pass:

- all 24 global suit renamings -> identical observation/output within numerical tolerance;
- Hero hole-card swap -> identical observation/output;
- all 6 flop permutations -> identical observation/output;
- turn/river swap -> **different** whenever the state itself differs;
- private/public card swap -> **different**;
- exact turn or river change -> **different** unless globally suit-isomorphic under the full observable state;
- true-HU dead-seat physical permutation -> identical canonical observation;
- meaningful BTN/SB/BB history difference in 3H -> **different**;
- trailing history padding -> exactly no effect;
- 33% vs 50% sizing -> **different**;
- forced blind vs voluntary bet -> **different**;
- c-bet vs donk/probe when lineage differs -> **different**;
- no physical suit number appears as a semantic category;
- no opponent hole card reaches the observation;
- serialized C++ observation == Python-decoded semantic contract.

---

## E. R7.5.3C architectural hypothesis

The next comparison must not repeat the old `raw exact` versus `semantic lossy` mistake.

Candidate family, subject to precommit before observing outputs:

- **H0 FIXED_V1** — exact V1 cards and current state, but correct history packing/padding and true-HU dead-seat canonicalization;
- **H1 RELATIONAL_EXACT** — lossless exact observable cards up to true game symmetries, using rank + relational suit information rather than physical suit IDs; unordered Hero pair/flop set handled invariantly; turn/river remain ordered street roles;
- **H2 RELATIONAL_EXACT_STRUCTURED_HISTORY** — H1 + structured action history;
- **H3 HYBRID_EXACT_SEMANTIC** — H2 + corrected invariant poker semantics (board texture, made hand, objective draws, lineage/aggressor facts, pairwise stack/SPR context), with no exact-card information removed;
- **H4 HYBRID_CAPACITY** — H3 with additional model capacity only if it remains inside a precommitted Ryzen inference/training/RAM envelope.

The preferred relational suit representation is label-free: exact ranks + visibility/role + pairwise same-suit relations (or another representation proven equivalent under all 24 suit relabelings). A monolithic remapped `rank+suit` lookup is not sufficient by itself because canonical suit labels are context-dependent.

---

## F. Evidence boundary / roadmap consequence

- Existing engine/rules/dealing/ICM evidence is not discarded without a demonstrated dependency.
- The ongoing R7.5.4A sizing run at immutable old SHA may finish as baseline/control evidence.
- No R8 production training may be authorized from the old representation while R7.5.3C is unresolved.
- If H1/H2/H3/H4 replaces V1, representation-dependent stability/sizing/strategic evidence must be rerun or explicitly bridged; it is not inherited silently.
- `READY FOR TABLES = NO`.
