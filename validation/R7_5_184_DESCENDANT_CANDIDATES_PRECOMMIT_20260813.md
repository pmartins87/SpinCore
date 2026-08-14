# R7.5 — suit-invariant descendants of the recovered legacy 184 map

Date: 2026-08-13
Status: **CANDIDATE FAMILY PRECOMMITTED BEFORE LEARNING-QUALITY RESULTS**

Parent evidence:

```text
validation/R7_5_LEGACY_CRUSHER_AND_184_AUDIT_20260813.md
validation/R7_5_LEGACY_184_MAPPING_AUDIT.json
```

The recovered historical `184Flops.json` has complete 22,100-flop coverage but fails suit-permutation invariance for 40 of the 1,755 exact suit-isomorphic flop classes. The raw historical mapping is therefore a reference/control and is not eligible unchanged.

This document freezes deterministic repair descendants before any comparative Deep-CFR learning-quality outputs are inspected.

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

H3 is reserved for a newly generated 184-class clustering whose atomic inputs are the 1,755 exact suit-isomorphic flop classes rather than 22,100 absolute-suit physical strings.

The clustering objective and deterministic features must be frozen before H3 is generated. At minimum it must not use absolute suit names. It may use deterministic structural poker features and/or strategically measured value/policy features only under a separately frozen generation protocol.

H3 is **not yet generated** and no result is implied by this precommit.

## Candidate H4 — EXACT_ISO_1755

The 1,755 exact suit-isomorphic classes remain the lossless structural reference candidate. H4 removes only absolute suit names and flop-card order; it performs no strategic compression beyond that symmetry.

H4 may be expensive relative to 184 but is essential as the accuracy-side reference for measuring what 184-class compression loses.

## Selection discipline

No candidate above is selected by this document.

R7.5.3 must compare eligible representations under frozen seeds, budgets, held-out states, strategic sentinels, learning-quality metrics and runtime/resource metrics. H0 cannot win. H1, H2, H3 and H4 are eligible for engineering comparison only after their mapping integrity checks pass; final production eligibility still depends on the R7.5.3 numerical gates.

The current V1 raw-card representation remains a required external control alongside these flop-abstraction candidates.

No candidate may be removed or added after seeing comparative learning-quality outputs without creating a new precommitted generation and invalidating the affected comparison.

## Reproducible tooling

```text
python/spincore/flop184_descendants.py
tools/r7_5_legacy184_descendant_audit.py
python_tests/test_r7_5_flop184_descendants.py
```

The descendant tooling produces compact `1,755 exact class -> legacy representative` mappings. Production code should canonicalize the flop first and then consult that compact map; there is no need to keep 22,100 suit-spelling-specific entries in NeuralInputV2.

## Gate impact

```text
R7.5.1 historical recovery                    PASS
R7.5.1 raw historical mapping integrity       FAIL on suit invariance
R7.5.1 deterministic descendant family        PRECOMMITTED
R7.5.1 descendant physical audits             PASS for H1/H2 invariance
R7.5.1 H3 generation                          PENDING
R7.5.3 strategic comparison                   PENDING
R7.5.5 production selection                   PENDING
R8.3/R8.4 production training                 BLOCKED
READY FOR TABLES                              NO
```
