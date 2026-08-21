# R7.5.3D — V1+ post-mortem implementation audit

Date: 2026-08-21
Status: FROZEN_BEFORE_FORENSIC_OUTPUTS
READY FOR TABLES: NO
Production training authorized: NO

## Scope

This document audits the implementation of `tools/r7_5_3d_v1plus_forensic_postmortem.py` against the already-frozen scientific precommit in `validation/R7_5_3D_V1PLUS_POSTMORTEM_PRECOMMIT_20260821.md`.

It does not change the hypotheses, hard gates, training seeds, evaluation seeds, action candidate, trained weights, x16 artifacts, or any production authorization. It exists only to close diagnostic coverage gaps before the first R7.5.3D forensic output is generated.

## Findings

The original read-only post-mortem tool correctly covers the core evidence path:

- x16 final-checkpoint identity validation;
- extraction/loading of both frozen training-seed policies;
- per-state cross-seed TV on the frozen heldout states;
- street/history/action-path/legal-count/SPR summaries;
- history-refinement diagnostics for V1-like, structured-categorical and exact V3 history identities;
- action-slot L1 decomposition;
- final Advantage/Strategy reservoir capacity, seen, retained, uniqueness, saturation, iteration distribution and cross-seed overlap;
- paired H3-minus-H2 TV on aligned heldout state identities.

Before execution, four implementation gaps were identified relative to the precommit:

1. current pot, to-call, current bet and stack geometry were present only partially/correlationally rather than as explicit frozen-bin TV slices;
2. forced/non-forced counts, actor diversity, last actor and full action-type composition were not all emitted as grouped TV slices;
3. the state-level `dominant_delta_slot` used `argmax` over all ten slots, so a zero-difference state or a numerically irrelevant illegal slot could receive a misleading dominant-slot label;
4. the high-TV summary listed state indices, but the precommit requires enough decoded context for manual inspection.

## Mechanical completion strategy

Do not mutate the already-created raw forensic tool or the x16 artifacts. Add a deterministic read-only enrichment stage:

`tools/r7_5_3d_v1plus_forensic_enrich.py`

The enrichment consumes the raw forensic JSON plus the same frozen heldout tree, validates identities, and emits the missing precommitted diagnostics. This separation preserves the original raw readout as an auditable intermediate artifact.

The enrichment must:

- recompute dominant disagreement only across legal slots and return `null` when TV is effectively zero;
- add fixed, code-defined bins for pot, to-call, current bet, stack depth, stack spread, stack geometry and SPR;
- add grouped TV slices for forced/non-forced counts, actor diversity, last actor/action, action-type composition and quantitative history-sizing statistics;
- re-decode the candidate-independent heldout observations to recover the complete rel0/rel1/rel2 stack geometry without changing model inputs;
- emit context-rich top high-TV states;
- emit paired H3-minus-H2 diagnostics across the same relevant strata;
- summarize action-slot concentration without interpreting it as causation;
- preserve `production_training_authorized=false` and `ready_for_tables=false`.

## Fixed binning policy

The following bins are frozen before outputs:

- pot BB: `<=2`, `(2,5]`, `(5,10]`, `(10,20]`, `>20`;
- to-call/current-bet BB: `0`, `(0,1]`, `(1,2]`, `(2,5]`, `>5`;
- live stack BB: `0`, `(0,5]`, `(5,10]`, `(10,20]`, `(20,40]`, `>40`;
- stack spread BB: `0`, `(0,5]`, `(5,10]`, `(10,20]`, `>20`;
- SPR: `<=1`, `(1,2]`, `(2,5]`, `(5,10]`, `>10`, plus `NA`;
- history-count variables: `0`, `1-2`, `3-4`, `5-8`, `9-16`, `17+`;
- paid/pot and commitment/pot statistics: `0`, `(0,.25]`, `(.25,.5]`, `(.5,1]`, `(1,2]`, `>2`.

Action-type composition is represented by the exact six-count vector `(type0,...,type5)` already derivable from each decoded V3 history. Because this is forensic rather than a production feature, no frequency pruning or post-output rebucketing is allowed.

Stack geometry is represented by the ordered live relative-stack bins, preserving the actor-relative geometry while avoiding post-hoc continuous thresholds.

## Validation

A synthetic test must cover at minimum:

- all fixed bin boundaries;
- legal-only dominant-slot selection;
- null dominant slot at zero TV;
- action-composition identity;
- grouped-TV accounting;
- paired H3-minus-H2 alignment and duplicate/missing-identity rejection.

No x16 result is to be interpreted until the raw and enrichment stages both complete and their output identities are persisted.

## Governance

- No new training in this phase.
- No architecture winner is selected by the raw or enriched forensic output.
- No threshold or seed may change.
- No inference of causation from a single correlation or slice.
- The later ablation matrix is frozen only after these outputs exist.
