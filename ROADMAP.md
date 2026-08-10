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
    - stronger AdvantageNet fit screen — **RUNNING**
    - brute-force root scaling — **PAUSED until causal diagnosis closes**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 convergence
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Current R7.3 evidence

Corrected 640 roots/seed achieved individual fit gates but failed cross-seed stability: mean TV `0.47765`, p95 TV `0.90240`. Doubling to 1280 roots/seed left mean TV essentially flat at `0.47319`, so brute-force scaling was paused.

The controlled variance decomposition showed that changing the CFR strategy memory while holding AveragePolicy initialization/optimizer seed fixed produced mean TV `0.46989`, versus `0.24265` average disagreement between replicas trained on the same memory. This identified strategy-memory/CFR variance as the larger source.

Holding the root deal/future-board stream identical across algorithm seeds barely changed across-memory divergence (`0.46007` shared-deck versus `0.46989` independent-deck reference). Off-support AveragePolicy replica disagreement was `0.29231`, versus `0.12031` on each memory's own support, so policy extrapolation is material but not sufficient to explain the strategy-memory disagreement.

The completed support-overlap diagnostic then compared the actual strategy samples rather than neural predictions. Raw exact support Jaccard was only `0.04343`; after diagnostic poker card isomorphism it was `0.02993`. LCFR-weight coverage improved modestly from `0.07773` to `0.09219`, and weighted shared-target TV fell from `0.53832` to `0.42455`, but the shared CFR targets still disagree strongly. Card canonicalization is therefore not promoted to a representation rewrite from current evidence.

The 256-root traversal/training RNG-coupling screen also failed to identify the known shared `bundle.batch_rng` as a dominant mechanism. Splitting traversal from minibatch RNG changed poker-isomorphic Jaccard by only `+4.5%`, LCFR-weight coverage by `+5.5%`, and actually increased shared-target TV by about `10.4%`. Production RNG/checkpoint semantics remain unchanged.

The active diagnostic now tests a more direct hypothesis: **the rebuilt AdvantageNet may be strategically underfit despite passing the frozen NRMSE gate**. Corrected runs commonly stop around NRMSE `0.65–0.70`, while historical pre-loss fits were materially lower (`~0.45–0.55`). A controlled 256-root screen compares the present internal target `0.70` against a stronger `0.50` target, with the same roots/shared decks and traversal RNG separated from optimizer RNG. Iteration 1 is collected before any fitted AdvantageNet exists; iteration-2 support and shared-target divergence therefore provide a direct screen of whether stronger advantage fitting stabilizes CFR dynamics.

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
