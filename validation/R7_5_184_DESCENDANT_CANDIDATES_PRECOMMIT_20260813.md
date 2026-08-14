# R7.5 — suit-invariant descendants of the recovered legacy 184 map

Date: 2026-08-13
Status: **CANDIDATE FAMILY PRECOMMITTED; H1/H2/H3 STRUCTURALLY VALIDATED; NO STRATEGIC WINNER**

Parent evidence:

```text
validation/R7_5_LEGACY_CRUSHER_AND_184_AUDIT_20260813.md
validation/R7_5_LEGACY_184_MAPPING_AUDIT.json
validation/R7_5_CRUSHER_FRAMEWORK_FULL_SEMANTIC_AUDIT_20260813.md
```

The recovered historical `184Flops.json` has complete 22,100-flop coverage but fails suit-permutation invariance for 40 of the 1,755 exact suit-isomorphic flop classes. The raw historical mapping is therefore a reference/control and is not eligible unchanged.

This candidate family was frozen before comparative Deep-CFR learning-quality outputs. Subsequent structural validation below does not select a strategic winner.

`READY FOR TABLES = NO`.

## Candidate H0 — RAW_LEGACY_184

Purpose: historical control only.

```text
active representatives                         184
physical assignments changed from history        0
exact suit-isomorphic classes split              40
suit-permutation invariance                    FAIL
production eligible                              NO
```

H0 may be used only to quantify the effect of the historical defect. It cannot win R7.5.

## Candidate H1 — LEGACY184_CANONICAL_INPUT_V1

Definition:

1. Canonicalize each physical flop to SpinCore's exact suit-isomorphic key.
2. Use the historical bucket assigned to that exact class's deterministic canonical physical suit spelling.
3. All physical suit spellings of that exact class inherit that one bucket.

This is not a learned repair and contains no performance-driven tie-breaking.

Physical audit against the recovered map:

```text
exact classes                              1,755
active historical representatives             184
physical assignments changed                 240
fraction changed                       0.0108597  (~1.086%)
exact suit-isomorphic classes split             0
suit-permutation invariance                  PASS
representatives lost                            0
expanded bucket size min                         8
expanded bucket size median                     96
expanded bucket size max                       600
```

Important property: H1 preserves the complete historical 184-representative vocabulary while removing dependence on absolute suit spelling.

## Candidate H2 — LEGACY184_MAJORITY_MIN_CHANGE_V1

Definition:

1. Canonicalize physical flops into the same 1,755 exact suit-isomorphic classes.
2. For each exact class, select the historical representative used by the largest number of physical suit spellings in that class.
3. Exact count ties, if any, are broken by lexical representative name.

This deterministically minimizes changed historical physical assignments within each exact class.

Physical audit:

```text
exact classes                              1,755
active historical representatives             181
physical assignments changed                 200
fraction changed                       0.00904977 (~0.905%)
exact suit-isomorphic classes split             0
suit-permutation invariance                  PASS
representatives lost                            3
lost representatives:
  4s5sKd
  5s9sQd
  7sQsAd
```

H2 is more conservative by changed physical assignment count, but it collapses the effective bucket vocabulary from 184 to 181. That tradeoff must be evaluated, not silently preferred.

## Candidate H3 — SUIT_INVARIANT_RECLUSTERED_184_V1

H3 is a genuinely new 184-medoid abstraction. It does **not** inherit the historical centroid identities or historical physical assignments.

Its atomic states are the 1,755 exact flop classes obtained after removing only card order and one global renaming of suit names. The generation protocol was designed before comparative learning-quality outputs and uses only objective flop structure; it contains no hard-coded action, legacy handlist, learned policy value or observed candidate performance.

### H3 hard structural strata

No cluster is allowed to cross these deterministic boundaries:

```text
suit texture:
  rainbow / two-tone / monotone

rank shape:
  unpaired / paired / trips

basic connectivity:
  maximum occupied ranks in a five-rank straight window
```

The physically present combinations form **14 hard strata** across the 1,755 exact classes.

### H3 within-stratum features

Four objective groups are normalized independently so no single wide group dominates merely by dimension count:

```text
1. rank identity / rank structure
2. suit relationships
3. straight-window connectivity / rank gaps
4. deterministic next-turn transition profile
```

The turn-transition profile measures structural events such as pairing a flop rank, overcards/undercards, three-/four-suit development and straight-window development. It does not use a strategy or opponent range.

### H3 clustering algorithm

```text
cluster target: 184
allocation: proportional across hard strata, >= 1 per stratum
initialization: deterministic farthest-first
cluster center: real exact-class medoid, never a synthetic centroid
refinement: deterministic PAM-style within-cluster medoid update
exact ties: lexical exact-class key
```

Generation runtime dependency used for the frozen mapping:

```text
Python reference CI: 3.11.15
NumPy:               2.3.5 pinned
```

### H3 validated result

```text
exact suit-isomorphic classes covered      1,755
physical flops covered                    22,100
active medoids                               184
hard-stratum mismatches                        0
suit-permutation invariance                  PASS
exact-class bucket size min                     1
exact-class bucket size median                  9
exact-class bucket size max                    23
physical-flop bucket size min                   4
physical-flop bucket size median              108
physical-flop bucket size max                 384
```

Frozen mapping SHA256:

```text
2c83cf993bcc4003223d184bd6f5584720b23cf04b95e6db69f84b09a86a64d0
```

CI evidence:

```text
workflow: SpinCore main regression
run:      31764848608
result:   SUCCESS
```

This establishes **mapping integrity and deterministic reproducibility only**. It does not establish that H3 is strategically better than H1, H2, H4 or V1.

## Candidate H4 — EXACT_ISO_1755

The 1,755 exact suit-isomorphic classes remain the lossless structural reference candidate. H4 removes only absolute suit names and flop-card order; it performs no strategic compression beyond that symmetry.

H4 may be more expensive than 184 but is essential as the accuracy-side reference for measuring what 184-class compression loses.

## Selection discipline

No candidate above is selected by this document.

R7.5.3 must compare eligible representations under frozen seeds, budgets, held-out states, strategic sentinels, learning-quality metrics and runtime/resource metrics. H0 cannot win. H1, H2, H3 and H4 are eligible for engineering comparison after their mapping integrity checks; final production eligibility still depends on the R7.5.3 numerical gates.

The current V1 raw-card representation remains a required external control alongside these flop-abstraction candidates.

No candidate may be removed or added after seeing comparative learning-quality outputs without a new precommitted generation and invalidation/re-run of the affected comparison.

## Reproducible tooling

```text
H1/H2:
  python/spincore/flop184_descendants.py
  tools/r7_5_legacy184_descendant_audit.py
  python_tests/test_r7_5_flop184_descendants.py

H3:
  python/spincore/flop184_recluster.py
  tools/r7_5_h3_recluster.py
  python_tests/test_r7_5_flop184_recluster.py

C++ exact-class language:
  include/spincore/flop_canonicalization.hpp
  src/flop_canonicalization.cpp
  tests/test_flop_canonicalization.cpp
```

Production candidate code should canonicalize the flop first and then apply the selected abstraction at the neural-observation boundary. The authoritative poker engine remains exact.

## Gate impact

```text
R7.5.1 historical recovery                    PASS
R7.5.1 raw historical mapping integrity       FAIL on suit invariance
R7.5.1 deterministic H1/H2 descendants        PASS structural/invariance
R7.5.1 deterministic H3 generation            PASS structural/invariance/reproducibility
R7.5.1 H4 exact reference                     PASS definition/reference
R7.5.3 strategic comparison                   PENDING
R7.5.5 production selection                   PENDING
R8.3/R8.4 production training                 BLOCKED
READY FOR TABLES                              NO
```
