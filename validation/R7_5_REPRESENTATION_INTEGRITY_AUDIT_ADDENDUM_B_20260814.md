# R7.5 Representation / Engine Integrity Audit — Addendum B — 2026-08-14

This addendum extends `validation/R7_5_REPRESENTATION_INTEGRITY_AUDIT_20260814.md` and records evidence obtained after the original audit was frozen.

`READY FOR TABLES = NO`.

## 1. Confirmed additional representation defect: legal public history can exceed 32 events

A deterministic C++ regression constructs a true-heads-up hand in the frozen 1500-chip / 10-20 test geometry and repeatedly applies ordinary legal full minimum raises while staying away from the all-in boundary.

The state reaches a fully legal 33-event public history while the hand remains nonterminal and preflop. Both V1 and V2 expose only 32 events because both keep the most recent fixed window.

At exactly 33 events, the full history consists of two forced blind posts plus 31 voluntary raises; the neural window contains exactly one forced blind plus the 31 raises, proving that the first public event has been discarded.

Consequence: fixed `last 32` is not lossless even in a mechanically reachable test state. This is no longer a theoretical concern.

Production decision frozen separately in `R7_5_3C_PHASE2_STRUCTURAL_ADMISSION_FREEZE.json`: H2/H3 final must use complete variable-length structured public history with no strategic last-N truncation.

## 2. Old card-semantics defect fixed: 2345 wheel-edge OESD

The old OESD detector iterated four-consecutive-rank sequences starting at low rank 3. This omitted the legal `2,3,4,5` sequence, whose two completion ranks are Ace and Six. The old semantics therefore could label 2345 as double-gutshot rather than open-ended.

The detector now admits low rank 2 and a dedicated regression asserts:

- straight draw true;
- missing-rank count 2;
- open-ended true;
- gutshot false;
- double-gutshot false.

The regression passes in the C++ suite.

This reinforces the Phase-2 rule that H3 final must use corrected objective semantic descriptors rather than blindly inheriting old V2 ontology fields.

## 3. Three-handed postflop ontology coverage expanded

Adversarial 3H tests now cover:

- BTN preflop raiser versus two callers, SB/BB check, BTN c-bet, first caller folds, second caller still correctly faces c-bet;
- caller lead before the lineage aggressor correctly identified as donk;
- turn probe after a three-way flop checks through and preflop aggressor misses c-bet;
- multiway 3-bet pot where BB is lineage aggressor, SB checks, BB checks, BTN caller receives the intended float/stab opportunity while a third player remains live.

All added C++ ontology regressions pass on the reviewed `main` regression run.

This does not prove every possible 3H action history, but it materially closes the previous HU-heavy coverage gap.

## 4. Pairwise stack geometry V3 added and property-tested

New Python representation helper:

`python/spincore_nn/stack_geometry_v3.py`

It deliberately does **not** collapse 3H to one effective stack / one SPR. For each actor-relative opponent it retains:

- present mask;
- contesting mask;
- actionable mask;
- pairwise effective remaining stack in BB;
- pairwise SPR against the current pot;
- pairwise effective total chip cap;
- commitment gap.

Example property test: Hero 12bb versus opponents 4bb and 20bb in an 8bb pot must expose effective remaining `(4bb, 12bb)` and pairwise SPR `(0.5, 1.5)`, not a single 4bb/0.5 summary.

Three-handed opponent slots are not sorted by stack size because doing so would erase position. In canonical true HU the layout is `[Hero, live opponent, absent]` and the absent seat is explicitly masked rather than treated as a zero-stack all-in player.

The R7.5.3C hybrid property preflight including these stack-geometry checks passed in workflow run `31833687485`.

## 5. Independent engine/evaluator integrity evidence

### 5.1 Exhaustive five-card evaluator census — PASS

Dedicated audit workflow run `31833963887` exhaustively evaluated all `C(52,5) = 2,598,960` five-card hands.

Observed counts exactly matched the canonical combinatorial census:

- High card: 1,302,540
- One pair: 1,098,240
- Two pair: 123,552
- Trips: 54,912
- Straight: 10,200
- Flush: 5,108
- Full house: 3,744
- Quads: 624
- Straight flush: 40

The evaluator also produced exactly 7,462 distinct five-card `HandRank` values.

This is strong evidence against gross category/tiebreak construction errors in the five-card evaluator. It does not, by itself, prove every integration path that calls `evaluate_best`, but `evaluate_best` separately enumerates all five-card subsets of 5-7 visible cards.

### 5.2 Deterministic legal-hand property fuzz — PASS

The same integrity gate executed 3,000 randomized legal three-handed hands plus 3,000 randomized legal true-HU hands over asymmetric/short stack partitions.

After every action it checked:

- no negative stacks/commitments;
- street commitment <= total commitment;
- pot == sum(total commitments);
- remaining stacks + pot == frozen total chips;
- valid nonterminal actor and at least one legal action;
- no duplicate dealt cards;
- dead HU seat receives no valid hole cards;
- no legal hand exceeds a 256-action loop guard.

At terminal settlement it checked every final stack nonnegative and exact chip conservation.

The 6,000-hand property audit passed.

Interpretation: this materially lowers the probability that the representation findings are symptoms of a grossly broken dealing/betting/settlement core. It is evidence, not a mathematical proof of every possible legal betting state.

## 6. Action-abstraction alias finding localized

The old six-slot V1 path can expose multiple abstract slots that clamp to the same exact action in short-stack states. This is a historical sample-efficiency/theory concern for old V1 behavior/corpora.

The R7.5.4A action-sizing collector does **not** inherit this defect: its universal action resolver deduplicates state-local exact aliases before CFR branching and records nominal-versus-unique aggressive branch telemetry.

Therefore the alias finding is not grounds to invalidate the current R7.5.4A sizing experiment, though it is another reason not to treat old V1-generated paired targets as authoritative for a new representation.

## 7. Phase-2 structural admission consequence

The production candidates are now structurally constrained before end-to-end outputs:

- H0/H1: control only; not production-eligible because they retain confirmed lossy V1 public history.
- H2 final: lossless relational exact cards + complete variable-length structured public history + exact C++/Python parity.
- H3 final: H2 final + corrected objective semantics + pairwise stack geometry.
- H4 final: capacity-reinvestment candidate only after H3 structural/resource admission.

No Phase-1 offline fit can override these structural requirements because Phase 1 uses V1-generated targets and has explicitly frozen diagnostic-only authority.
