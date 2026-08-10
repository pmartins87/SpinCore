# SpinCore finite roadmap — canonical recovery generation 2

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until all gates pass.

- R0 Foundation / canonical repository — **PASS REBUILT**
- R1 Complete poker engine — **PASS REBUILT**
- R2 Canonical infoset + neural encoder — **PASS REBUILT**
- R3 Tournament continuation value (`ICM_EXACT_V1`, explicit payout) — **PASS REBUILT**
- R4 Neural infrastructure — **PASS REBUILT**
- R5 CFR correctness oracle — **PASS REBUILT**
- R6 Deep CFR integration on authoritative `SpinTraversalState` — **PASS REBUILT**
- R7 Pilot / performance / statistical stability — **IN PROGRESS**
  - R7.0 approximation metrics / full-reservoir audit — **PASS REBUILT**
  - R7.1 native own-reach frontier — **PASS REBUILT**
  - R7.2 LCFR weighting / exact checkpoint+resume / fresh-process worker — **PASS REBUILT**
  - R7.3 multi-seed stability — **FAIL / ACTIVE**
    - corrected 640-root fit gates — **PASS**, cross-seed — **FAIL**
    - 1280-root brute-force scale — **FAIL**, cross-seed essentially flat and policy fit degraded
    - controlled variance decomposition — **DONE: CFR-memory variance dominant**
    - shared-deck diagnostic — **DONE: root card/chance stream not dominant**
    - support-conditioned policy diagnostic — **DONE: off-support extrapolation material**
    - exact/card-isomorphic support overlap — **DONE: support sparse; card representation not dominant**
    - traversal/training RNG coupling — **DONE: not dominant at screen scale**
    - stronger AdvantageNet fit screen — **DONE: material at 256-root screen scale**
    - stronger AdvantageNet 640 candidate — **DONE: modest improvement only; still far from frozen cross-seed gates**
    - own-reach bootstrap support-density curve — **DONE: sampling density strongly controls support overlap**
    - Advantage bootstrap target-variance curve — **DONE: external sampling materially noisy, but 8x replication only modestly reduces regret-matching TV**
    - exact own-reach expectation feasibility benchmark — **DONE: feasible at 4-root benchmark, but exact support is enormous; 8 sampled paths covered only 2.11%**
    - external-sampling path replication decomposition — **RUNNING: baseline vs strategy_x4 vs advantage_x4 vs both_x4**
    - brute-force root scaling — **PAUSED until causal diagnosis closes**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 convergence
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Current R7.3 evidence

### Acceptance-scale baselines

Corrected 640 roots/seed:

- Advantage and AveragePolicy individual fit gates: **PASS**
- cross-seed mean TV: `0.477649`
- cross-seed p95 TV: `0.902403`
- frozen cross-seed gates: mean `<= 0.15`, p95 `<= 0.35`

1280 roots/seed:

- mean TV: `0.473190`
- p95 TV: `0.875278`
- AveragePolicy fit exceeded the frozen `0.12` gate on both seeds
- doubling roots therefore gave essentially no useful convergence and brute-force scaling was stopped.

### Causal decomposition already completed

1. **CFR memory dominates AveragePolicy optimizer/init variance.** Across-memory mean TV `0.469892`, versus within-memory replica mean TV `0.242653`.
2. **Different hidden-card/deck streams are not dominant.** A common root deck stream produced across-memory mean TV `0.460068`, only about 2.1% below the independent-deck reference.
3. **Off-support neural extrapolation is material.** Same-memory replica disagreement averaged `0.120310` on own support and `0.292305` on the other memory's support (`2.43x`).
4. **Card canonicalization is not the dominant correction.** Raw exact support Jaccard at 640 was `0.043432`; diagnostic poker-card isomorphism did not expand it (`0.029931`). It modestly improved LCFR-weight coverage and reduced shared-target TV, but shared CFR targets still disagreed strongly.
5. **Training/traversal RNG coupling is not dominant at the 256-root screen scale.** Splitting those streams changed support only a few percent and worsened shared-target TV.
6. **Stronger AdvantageNet fitting is beneficial but insufficient.** At full 640 acceptance scale mean TV improved only from `0.477649` to `0.464474` (`2.76%`) and p95 from `0.902403` to `0.886204` (`1.80%`), despite both individual fit gates passing.
7. **Own-reach Monte-Carlo sampling is a proven source of support fragmentation even under an identical exact policy.** With the same 256 root deals and exact zero-regret uniform behavior, increasing independent own-reach trajectories from 1 to 8 raised poker-isomorphic Jaccard `0.033946 -> 0.074383` (`2.191x`) and LCFR-weight coverage `0.083230 -> 0.204593` (`2.458x`). Shared strategy-target TV remained zero to floating-point precision, proving the difference is sampling support rather than true policy disagreement.
8. **External-sampling Advantage targets are also noisy.** Under the same exact uniform policy and same root deals, 1->8 Advantage trajectories raised weight coverage `0.077203 -> 0.158538` (`2.054x`), but shared-target relative RMSE fell only `1.00943 -> 0.88105` and induced regret-matching mean TV only `0.42100 -> 0.37127`; p95 remained `1.0`. More Advantage paths help, but with weaker dose response than own-reach support density.
9. **Exact own-reach expectation is computationally possible on a bounded benchmark, but exposes the scale of the support problem.** Four unique HU deals (both target players) required `1,265,152` tree nodes and `188,440` target-state samples in `4.95 s`, yielding `116,192` unique raw observations. On those same four deals, 1 sampled own-reach trajectory covered only `0.301%` of exact support, 4 trajectories `0.803%`, and 8 trajectories `2.107%`. Exact enumeration is therefore a plausible estimator-design direction, not a drop-in production change: at acceptance scale its raw sample volume would be enormous and would interact with reservoir design/capacity.

Strong-Advantage 640 details:

- seed `20260829`: final Advantage NRMSE `0.504845`, policy TV `0.106862` — PASS/PASS
- seed `20260807`: final Advantage NRMSE `0.496334`, policy TV `0.103695` — PASS/PASS
- cross-seed mean TV `0.464474` — FAIL
- cross-seed p95 TV `0.886204` — FAIL
- evidence workflow `31364029367`, commit `75885791b5d8894d4c590a038233f96db5879925`

## Active R7.3 experiment: downstream path-variance decomposition

The active 256-unique-root diagnostic runs four controlled modes under the same root-deck stream and historical-range Advantage fitting:

- `baseline`: 1 advantage traversal replicate + 1 own-reach strategy replicate per unique deal
- `strategy_x4`: 1 advantage replicate + 4 strategy replicates
- `advantage_x4`: 4 advantage replicates + 1 strategy replicate
- `both_x4`: 4 + 4

The diagnostic separates advantage, strategy and optimizer RNG streams so `baseline` versus `strategy_x4` is a strict isolation test: CFR advantage memory/training must remain identical, while only average-policy own-reach sampling density changes. A workflow invariant fails the run if those Advantage checkpoint NRMSEs differ beyond numerical tolerance.

Each mode fits its AveragePolicy and measures acceptance-like cross-seed mean/p95 TV, the same models on the baseline common corpus, iteration-specific strategy-memory support, LCFR-weight coverage and shared-target disagreement. The already-completed exact bootstrap curves give causal priors; this downstream experiment decides which variance reduction actually improves the fitted policy metric.

Workflow: `31366433008`.

The exact own-reach feasibility evidence is workflow `31367407567`, commit `a16e043fffda04f2c2fa228611e3e352d7ca39b8`. The result justifies keeping exact expectation/stratified reach as an estimator-redesign option, but **not** replacing the recovered sampled own-reach contract without a versioned design, bounded-memory strategy and checkpoint/resume recertification.

Detailed control evidence: `validation/R7_3_PATH_VARIANCE_CONTROLS_20260810.md`.

## Frozen R7.3 gates

- Advantage weighted normalized RMSE `<= 0.75`
- Average-policy weighted mean TV `<= 0.12`
- Cross-seed mean TV `<= 0.15`
- Cross-seed p95 TV `<= 0.35`

Historical pre-loss checkpoint: 640 HU roots/seed, cross-seed mean TV `0.3714`, p95 TV `0.6878`. It remains historical evidence, not a directly comparable generation-2 gate result.

## Recovery invariants

- No frozen gate may be relaxed.
- `TRUE_HEADS_UP` and `THREE_HANDED` are separate domains for the whole hand.
- A hand starting 3H does not switch strategic domain merely because a player folds/all-ins during that hand.
- Production utility is exact explicit-payout ICM continuation delta; chip delta is diagnostics only.
- Equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful step is persisted to GitHub `main`.
- Root scaling is not resumed while a lower-cost causal diagnostic can distinguish approximation, chance coverage, representation, RNG coupling, fit quality and estimator/path variance.
