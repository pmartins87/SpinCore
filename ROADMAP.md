# SpinCore finite roadmap — canonical recovery generation 2

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until all gates pass.

- R0 Foundation / canonical repository — **PASS REBUILT**
- R1 Complete poker engine — **PASS REBUILT**
- R2 Canonical infoset + neural encoder — **PASS REBUILT**
- R3 Tournament continuation value (`ICM_EXACT_V1`, explicit payout) — **PASS REBUILT**
- R4 Neural infrastructure — **PASS REBUILT**
- R5 CFR correctness oracle — **PASS REBUILT**
- R6 Deep CFR integration on authoritative `SpinTraversalState` — **PASS REBUILT**
- R7 Pilot / performance / statistical stability — IN PROGRESS
  - R7.0 approximation metrics / full-reservoir audit — **PASS REBUILT**
  - R7.1 native own-reach frontier — **PASS REBUILT**
  - R7.2 LCFR weighting / exact checkpoint+resume / fresh-process worker — **PASS REBUILT**
  - R7.3 multi-seed stability — **FAIL / ACTIVE**
    - corrected 640-root fit gates — **PASS**, cross-seed — **FAIL**
    - 1280-root scale — **FAIL**, cross-seed essentially flat and policy fit degraded
    - controlled 640-root variance decomposition — **DONE: CFR-memory variance dominant**
    - shared-deck + support-conditioned diagnostic — **DONE: common root cards not dominant; off-support extrapolation material**
    - exact/card-isomorphic support-overlap diagnostic — **DONE: support remains sparse and shared CFR targets strongly disagree**
    - traversal/training RNG-coupling screen — **DONE: coupling not dominant at screen scale**
    - stronger AdvantageNet fit screen — **DONE: material improvement in iteration-2 support/target stability**
    - strong-Advantage 640-root candidate — **RUNNING**
    - brute-force root scaling — **PAUSED; only evidence-driven candidate runs allowed**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 convergence
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Current R7.3 evidence

Corrected 640 roots/seed achieved individual fit gates but failed cross-seed stability: mean TV `0.47765`, p95 TV `0.90240`. Doubling to 1280 roots/seed left mean TV essentially flat at `0.47319`, so brute-force scaling was paused.

The controlled variance decomposition identified strategy-memory/CFR variance as the larger source. Holding the root deal/future-board stream identical barely changed divergence, and card-isomorphic support analysis showed both very sparse shared support and large disagreement in the CFR targets themselves. A card-encoding rewrite is therefore deferred. Splitting traversal RNG from training RNG also produced only small support changes and worsened shared-target TV, so the production RNG/checkpoint contract remains unchanged.

The stronger-Advantage screen then tested the approximation component most directly. With identical shared root decks and the same isolated traversal RNG setup, the current `0.70` internal fit target was compared against `0.50`. The stronger fits reached NRMSE `0.47118` / `0.47672` after iteration 1, versus `0.65302` / `0.68509` under the weak schedule. In iteration 2, poker-isomorphic Jaccard improved from `0.03696` to `0.04910` (`1.3287x`), mean LCFR-weight coverage from `0.08515` to `0.10888` (`1.2786x`), and shared-target weighted mean TV fell from `0.56144` to `0.52776` (ratio `0.9400`). Persisted diagnosis: `STRONGER_ADVANTAGE_FIT_MATERIAL_AT_SCREEN_SCALE`.

That effect is meaningful but still far from proving R7.3 convergence. A full two-seed **640-root strong-Advantage candidate** is therefore running now with the recovered acceptance semantics: independent deck stream by algorithm seed, current coupled production RNG semantics, Advantage internal target `0.50` with up to `4096` optimizer steps/iteration, AveragePolicy target `0.105` with up to `12288` steps, and all four frozen acceptance gates unchanged. This is not brute-force scaling; it is a direct validation of the only causal correction that has shown material benefit so far.

## Frozen R7.3 gates

- Advantage weighted normalized RMSE <= 0.75
- Average-policy weighted mean TV <= 0.12
- Cross-seed mean TV <= 0.15
- Cross-seed p95 TV <= 0.35

Historical pre-loss checkpoint: 640 HU roots/seed, cross-seed mean TV `0.3714`, p95 TV `0.6878`. It remains evidence of the previous implementation, not a direct metric baseline for the rebuilt network.

## Recovery invariants

- No frozen gate may be relaxed.
- `TRUE_HEADS_UP` and `THREE_HANDED` are separate domains for the whole hand.
- A hand starting 3H does not switch strategic domain merely because a player folds/all-ins during that hand.
- Production utility is exact explicit-payout ICM continuation delta; chip delta is diagnostics only.
- Equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful step is persisted to GitHub `main`.
- Root scaling is not resumed while a lower-cost causal diagnostic can distinguish approximation, chance-coverage, representation, RNG coupling, advantage-fit quality, and CFR path-variance failures.
