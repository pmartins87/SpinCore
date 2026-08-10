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
    - 1280-root brute-force unique-deal scale — **FAIL**, essentially flat and policy fit degraded
    - variance decomposition — **DONE: CFR-memory variance dominant**
    - common-deck / support conditioning — **DONE: chance stream not dominant; off-support extrapolation material**
    - card-isomorphic support audit — **DONE: card representation not dominant**
    - traversal/training RNG coupling — **DONE: bookkeeping coupling not dominant**
    - strong Advantage fitting — **DONE: useful, but only modest 640 improvement**
    - exact own-reach support curve — **DONE: own-reach sampling strongly fragments support**
    - exact Advantage target curve — **DONE: external sampling materially noisy**
    - exact own-reach expectation benchmark — **DONE: feasible oracle, enormous support volume**
    - downstream four-mode path decomposition — **DONE: Advantage external-sampling variance is the dominant isolated fitted-policy lever**
    - exact opponent-expectation Advantage oracle — **DONE: x4 captures much of independent-replication benefit; p95 tail remains severe**
    - partial-exact opponent estimator screen — **RUNNING: exact levels 0/1/2 with level-0 regression oracle**
    - 640 replicated candidates — **RUNNING IN PARALLEL: separated advantage_x4, separated both_x4, recovered-coupled advantage_x4**
    - brute-force unique-root scaling — **PAUSED**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 convergence
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Current R7.3 evidence

### Acceptance-scale baselines

Corrected 640 roots/seed: individual fit gates PASS, cross-seed mean TV `0.477649`, p95 `0.902403`.

1280 roots/seed: mean `0.473190`, p95 `0.875278`; both AveragePolicy fits exceeded `0.12`. Doubling unique deals therefore did not solve convergence and brute-force root scaling remains paused.

Strong-Advantage 640: both individual fit gates PASS; mean `0.464474`, p95 `0.886204`, only `2.76% / 1.80%` better than corrected 640. Better neural fit is beneficial but not the main remaining problem.

### Causal path-variance evidence

Under an identical exact uniform policy and identical hidden deals:

- own-reach strategy sampling 1->8 trajectories raises poker-isomorphic support Jaccard `0.033946 -> 0.074383` (`2.191x`) and LCFR-weight coverage `0.083230 -> 0.204593` (`2.458x`) while shared target TV stays zero;
- Advantage external sampling 1->8 trajectories raises weight coverage `0.077203 -> 0.158538`, lowers target relative RMSE `1.00943 -> 0.88105`, and lowers induced regret-matching mean TV `0.42100 -> 0.37127`, while p95 remains `1.0`.

Exact own-reach expectation on four HU deals required `1,265,152` nodes and `188,440` target-state samples; eight sampled own-reach paths covered only `2.107%` of exact action-path support.

The downstream controlled four-mode experiment (`31366433008`, evidence `a9c57fe6e3c9149ed3010ead280912295bd4f5f6`) restored fitted networks and isolated the two path estimators. All individual fit gates passed and baseline-vs-`strategy_x4` Advantage checkpoint NRMSE was exactly identical.

| mode | Adv reps | Strategy reps | mean TV | p95 TV | mean ratio vs baseline | p95 ratio vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 1 | `0.305382` | `0.870543` | 1.000 | 1.000 |
| strategy_x4 | 1 | 4 | `0.275642` | `0.865904` | `0.9026` | `0.9947` |
| advantage_x4 | 4 | 1 | `0.219118` | `0.690974` | `0.7175` | `0.7937` |
| both_x4 | 4 | 4 | `0.197598` | `0.726534` | `0.6471` | `0.8346` |

Persisted diagnosis: `ADVANTAGE_EXTERNAL_SAMPLING_VARIANCE_MATERIAL`. `advantage_x4` is the strongest isolated lever and has the best p95; `both_x4` has the best mean at higher cost.

### Exact Advantage oracle

Workflow `31368837895`, evidence `45c68d2028ac658ae12870c97b9bf758e47f2a89`, exactly enumerated opponent expectation on the same four-deal bootstrap tree:

- exact nodes `1,265,152`
- exact Advantage samples `188,440`
- exact phase `15.91 s`

Sampled memories versus the exact oracle:

| paths | exact weight coverage | target relative RMSE | regret-matching mean TV | p95 TV | weighted greedy agreement |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.04256` | `0.83586` | `0.38186` | `1.0` | `0.48024` |
| 4 | `0.11034` | `0.67602` | `0.27044` | `1.0` | `0.73384` |
| 8 | `0.15755` | `0.68697` | `0.25715` | `1.0` | `0.74734` |

The large 1->4 gain and small 4->8 incremental gain independently support x4 as the first acceptance-scale replication factor. The persistent p95=`1.0` motivates bounded partial-exact or stratified opponent estimators if x4 still misses the frozen tail gate.

## Active physical work

### Three 640 acceptance-scale candidates

1. Workflow `31368447316`, `advantage_x4`: 4 Advantage paths + 1 strategy path, separated Advantage/strategy/optimizer RNG streams.
2. Same workflow, `both_x4`: 4 + 4, same separated streams.
3. Workflow `31368894934`, `advantage_x4` using the **recovered coupled RNG contract**.

All use 640 unique roots/seed, independent per-seed acceptance deck schedules, Advantage target `0.50`, AveragePolicy target `0.105`, policy max `32768`, reservoir capacity `400000`, and unchanged frozen gates.

The coupled x4 candidate is operationally important: if it matches separated x4, the recovery can preserve the existing RNG-state structure and version only the sampling schedule before checkpoint/resume recertification.

### Partial-exact opponent estimator

Workflow `31369138285` tests exact-opponent levels `0`, `1`, and `2` at 256 roots. Level 0 must reproduce the prior controlled baseline to `1e-9`. Positive levels enumerate the next N opponent decisions exactly and probability-weight downstream Advantage samples, then resume ordinary external sampling. This is an unbiased bounded-variance estimator candidate and reports cross-seed improvement versus total node cost.

Detailed records:

- `validation/R7_3_CONVERGENCE_640_1280_20260810.md`
- `validation/R7_3_PATH_VARIANCE_CONTROLS_20260810.md`
- `validation/R7_3_ADVANTAGE_ESTIMATOR_DESIGN_20260810.md`

## Frozen R7.3 gates

- Advantage weighted normalized RMSE `<= 0.75`
- Average-policy weighted mean TV `<= 0.12`
- Cross-seed mean TV `<= 0.15`
- Cross-seed p95 TV `<= 0.35`

Historical pre-loss 640 checkpoint: cross-seed mean `0.3714`, p95 `0.6878`. It remains historical evidence, not a directly comparable generation-2 gate result.

## Recovery invariants

- No frozen gate may be relaxed.
- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful step is persisted directly to `main`.
- A successful experimental estimator/sampling schedule must be versioned into checkpoint/resume semantics and deterministically recertified before R7.3 can close.
