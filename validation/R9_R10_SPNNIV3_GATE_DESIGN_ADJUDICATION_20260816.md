# R9/R10 SPNNIV3 downstream contract adjudication — 2026-08-16

`READY FOR TABLES = NO`

This adjudication is frozen before any R9 or R10 execution and does not relax any strategic, safety, provenance, replay, or fail-closed requirement in the original finite gate designs.

The 2026-08-12 R9/R10 precommits predate the R7.5 SPNNIV3 migration work. Their finite sequence and anti-tuning rules remain authoritative, but the following representation-specific phrases are stale and are superseded by this document.

## R9 supersession

In R9.1, **`six-action output shape`** means:

> exact policy-output width declared by the immutable R7.5.5/R8.5 production action-schema identity, with identical model-head width, legal-mask width and action-ID namespace.

For the current SPNNIV3 candidate lineage the universal action width is **10**. R9 may not accept a six-action V1 artifact merely because the historical precommit used that phrase. If R7.5.5 freezes a different final production action schema, R9 binds that exact frozen identity; it never infers or defaults the width.

## R10 supersession

Every occurrence of **`SPNNIV1 neural observation bytes`** in R10 means:

> exact neural-observation bytes/schema frozen by R7.5.5 and carried unchanged through R8.5 and R9.

Every occurrence of **`six-action policy inference`**, **`six-action action IDs`**, **`six-action probabilities`**, or equivalent wording means:

> exact production action width/action-ID namespace frozen by R7.5.5 and carried unchanged through R8.5 and R9.

Under the current SPNNIV3 lineage the expected observation schema is `SPNNIV3` and the universal action width is **10**. No production R10 path may silently fall back to SPNNIV1, a six-action head, a 32-event history cap, or the legacy package-level V1 exports.

## Legacy-code quarantine rule

The following files remain useful for historical reconstruction/control comparisons but are **not eligible as implicit production defaults**:

```text
python/spincore_nn/__init__.py
python/spincore_nn/models.py
python/spincore_nn/codec.py
```

`models.py` has a six-action head; `codec.py` is explicitly SPNNIV1 and caps history at 32 events; and `__init__.py` currently exposes those V1 classes at package top level. They may remain in the repository for reproducibility. Production code must bind an explicit R7.5.5-selected module/schema through the immutable production manifest, and R10 must prove that exact binding.

## Unchanged downstream requirements

R11 and R12 remain generically bound to the exact action/observation schema identities admitted by R10/R9/R8.5. No representation-specific relaxation is introduced here.

This adjudication changes no R7.5 admission gate, no action-sizing gate, no R8 training gate, no R9 strategic threshold, no R10 runtime safety barrier and no table-readiness condition. R9 remains blocked until R8.5; R10 remains blocked until R9 PASS; R11 remains blocked until R10 PASS; R12 remains blocked until R11 PASS and all release debts are closed.
