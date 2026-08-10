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
    - Advantage bootstrap target-variance curve — **DONE: external sampling materially noisy**
    - exact own-reach expectation feasibility — **DONE: feasible at bounded scale but support/memory volume enormous**
    - downstream path-replication decomposition — **DONE: Advantage external-sampling variance is the dominant isolated path mechanism**
    - replicated 640 candidates — **RUNNING IN PARALLEL: advantage_x4 and both_x4**
    - brute-force unique-root scaling — **PAUSED**
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
- doubling unique roots therefore gave essentially no useful convergence and brute-force root scaling was stopped.

### Causal decomposition completed

1. **CFR memory dominates AveragePolicy optimizer/init variance.** Across-memory mean TV `0.469892`, versus within-memory replica mean TV `0.242653`.
2. **Different hidden-card/deck streams are not dominant.** A common root deck stream produced across-memory mean TV `0.460068`, only about 2.1% below the independent-deck reference.
3. **Off-support neural extrapolation is material.** Same-memory replica disagreement averaged `0.120310` on own support and `0.292305` on the other memory's support (`2.43x`).
4. **Card canonicalization is not the dominant correction.** Raw exact support Jaccard at 640 was `0.043432`; diagnostic poker-card isomorphism did not expand it (`0.029931`).
5. **Training/traversal RNG coupling is not dominant at screen scale.** Splitting those streams changed support only a few percent and worsened shared-target TV.
6. **Stronger AdvantageNet fitting is beneficial but insufficient.** At full 640 acceptance scale mean TV improved only `0.477649 -> 0.464474` (`2.76%`) and p95 `0.902403 -> 0.886204` (`1.80%`), despite both individual fit gates passing.
7. **Own-reach Monte-Carlo sampling provably fragments support under an identical exact policy.** With the same 256 deals and exact uniform behavior, 1->8 independent own-reach trajectories raised poker-isomorphic Jaccard `0.033946 -> 0.074383` (`2.191x`) and LCFR-weight coverage `0.083230 -> 0.204593` (`2.458x`), while shared target TV stayed zero to floating-point precision.
8. **Advantage external sampling is also materially noisy.** Under the same exact uniform policy and same deals, 1->8 Advantage trajectories raised weight coverage `0.077203 -> 0.158538` (`2.054x`) and reduced induced regret-matching mean TV `0.421004 -> 0.371266`; p95 remained `1.0`.
9. **Exact own-reach expectation is feasible only as a versioned estimator-design option.** Four HU deals required `1,265,152` nodes and `188,440` target-state samples; 8 sampled own-reach trajectories covered only `2.107%` of exact action-path support.
10. **The downstream four-mode experiment identifies Advantage path variance as the stronger fitted-policy lever.** All four modes passed individual fit gates and the baseline-vs-`strategy_x4` Advantage checkpoint invariant was exactly `0.0` NRMSE difference. On each mode's own 2048-observation common corpus:

| mode | Advantage reps | Strategy reps | cross-seed mean TV | p95 TV | mean ratio vs baseline | p95 ratio vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 1 | `0.305382` | `0.870543` | 1.000 | 1.000 |
| strategy_x4 | 1 | 4 | `0.275642` | `0.865904` | `0.9026` | `0.9947` |
| advantage_x4 | 4 | 1 | `0.219118` | `0.690974` | `0.7175` | `0.7937` |
| both_x4 | 4 | 4 | `0.197598` | `0.726534` | `0.6471` | `0.8346` |

The persisted diagnosis is `ADVANTAGE_EXTERNAL_SAMPLING_VARIANCE_MATERIAL`. `advantage_x4` gives the strongest isolated improvement and the best p95 of the tested modes; `both_x4` gives the best mean but a worse p95 at higher compute cost. `strategy_x4` alone has little effect on the p95 despite greatly increasing strategy support. Evidence: workflow `31366433008`, commit `a9c57fe6e3c9149ed3010ead280912295bd4f5f6`.

### Active acceptance-scale candidates

Because both mean and p95 gates matter and the 256-root screen produced a Pareto tradeoff, two 640-root candidates are running **in parallel** rather than serial micro-tests:

- `advantage_x4`: four independent Advantage external-sampling trajectories per traverser/unique deal, one own-reach strategy trajectory;
- `both_x4`: four Advantage trajectories plus four own-reach strategy trajectories.

Both candidates use:

- 5 x 128 = 640 unique roots/seed;
- independent per-algorithm-seed hidden-deal schedules matching acceptance semantics;
- separate Advantage/strategy/optimizer RNG streams as an explicitly experimental contract;
- Advantage internal target `0.50`, max `4096` steps/iteration;
- AveragePolicy target `0.105`, max `32768` steps;
- reservoir capacity `400000`, preventing the replication experiment from being silently erased by the former 100k cap;
- unchanged frozen R7.3 acceptance gates.

Workflow `31368447316` runs both matrix jobs concurrently. The candidate runner was physically smoke-certified in workflow `31368044199`. A success at the frozen cross-seed gates would still require versioning the changed replication/RNG checkpoint contract and deterministic stop/restore/continue recertification before R7.3 can close.

Detailed causal record: `validation/R7_3_PATH_VARIANCE_CONTROLS_20260810.md`.

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
- More unique roots are not resumed while action-path variance can be attacked more efficiently per deal.
