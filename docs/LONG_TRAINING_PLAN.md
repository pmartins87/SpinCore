# SpinCore — Long-Training Plan

Status: **ENS8 8100 FINAL HU CANDIDATE FROZEN — SEALED HOLDOUT NEXT**
Date: 2026-09-21

## Design-set closure

The candidate survived:
- broad weak baselines;
- online Deep-CFR feedback;
- post-pilot forensic validation;
- independent learned-policy ecosystem crossplay.

Independent design-set learned ecosystem:
- ENS8 8100 absolute +7.538, CI fully positive;
- ENS8 8100 − ENS8 8000 +1.453, unresolved/no regression;
- ENS8 8100 − AVG8100 +13.059, resolved positive.

Direct seat-balanced:
- ENS8 8100 vs ENS8 8000 +1.227, unresolved/no regression;
- ENS8 8100 vs AVG8100 +12.663, resolved positive.

## Frozen deployment semantics

For TRUE_HEADS_UP:
- current ENS8 iteration 8100;
- checkpoint + ensemble sidecar;
- not AveragePolicy.

No further roots before holdout.

## Final holdout contract

Seeds 20261001..20261006.

Criteria and non-inferiority margin are frozen in:
`docs/LT2_HU_ENS8_FINAL_HOLDOUT_PROTOCOL_20260921.md`.

All criteria must pass.

A completed holdout result is not reusable for tuning. Any candidate changed after seeing it requires a new untouched validation family.

## After holdout

PASS:
- preserve final candidate;
- implement/finalize deployment inference path for HU ensemble semantics;
- run only mechanical integration checks, not policy tuning.

FAIL:
- preserve failure;
- do not retune on holdout;
- return to design data and create a genuinely new candidate/validation family.
