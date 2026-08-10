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
    - corrected 640 — **FIT PASS / CROSS-SEED FAIL**
    - 1280 unique-root scale — **FAIL; brute-force root scaling paused**
    - CFR-memory / support / chance / card / RNG / fit decomposition — **DONE**
    - exact own-reach and exact Advantage bootstrap controls — **DONE**
    - two-iteration path replication decomposition — **DONE**
    - exact opponent-expectation Advantage oracle — **DONE**
    - replicated-candidate V1 deck-control audit — **DONE; V1 not paired to acceptance reference**
    - deck-exact coupled Advantage x4 640 V2 — **RUNNING**
    - partial-exact V1 — **INVALID CONTROL; not scientific evidence**
    - partial-exact V2 256 — **DONE: levels 1/2 MATERIAL, level 2 strongest**
    - full-exact end-to-end 64 — **DONE: NOT BENEFICIAL, memory/model bottleneck exposed**
    - deck-exact partial-exact 640 levels 1/2 — **RUNNING IN PARALLEL**
    - common-random-number Advantage screen — **RUNNING**
    - five-iteration divergence/compounding diagnostic — **RUNNING IN PARALLEL**
    - antithetic/rotated-lattice Advantage x4 — **RUNNING**
    - same-memory Advantage fit/regret-sign sensitivity — **RUNNING**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 convergence
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Frozen R7.3 acceptance gates

- Advantage weighted normalized RMSE `<= 0.75`
- AveragePolicy weighted mean TV `<= 0.12`
- cross-seed mean TV `<= 0.15`
- cross-seed p95 TV `<= 0.35`

No gate has been relaxed.

## Authoritative generation-2 acceptance schedule

The deterministic deal schedule is frozen for paired acceptance experiments:

```text
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
```

`global_root` is continuous across CFR iterations. Replicated-candidate V1 used a different formula and its metrics remain independent physical experiments, not paired deltas. Correction: `validation/R7_3_REPLICATED_V1_DECK_CONTROL_CORRECTION_20260810.md`.

Authoritative references:

- corrected 640: mean `0.477649`, p95 `0.902403`, per-seed fits PASS;
- strong-Advantage 640: mean `0.464474`, p95 `0.886204`, per-seed fits PASS.

## Partial-exact V2 — decisive short-screen result

Workflow `31412806987`, evidence `9c23f9945ad543fa811d90ef2cfefb93d782cff3` used the authoritative recovered `ExternalSamplingCollector` for level 0 and probability-weighted experimental opponent enumeration only for positive levels. All modes passed individual Advantage/AveragePolicy fit gates.

At 256 roots/seed, two CFR iterations:

| estimator | mean TV | p95 TV | mean ratio vs level 0 | p95 ratio | node ratio |
|---|---:|---:|---:|---:|---:|
| authoritative level 0 | `0.313641` | `0.882657` | 1.000 | 1.000 | 1.000 |
| exact next 1 opponent decision | `0.223853` | `0.807704` | `0.7137` | `0.9151` | `2.667x` |
| exact next 2 opponent decisions | **`0.191695`** | **`0.669413`** | **`0.6112`** | **`0.7584`** | `9.349x` |

Persisted diagnosis: `PARTIAL_EXACT_OPPONENT_EXPECTATION_MATERIAL`.

Level 2 reduced mean cross-seed TV by about 38.9% and p95 by about 24.2% at two iterations. Its Advantage memories remained within/near the 100k screen cap (`93,740` retained/seen for seed29; `100,000` retained from `109,566` seen for seed07), and all frozen fit gates still passed even though the internal stricter `0.50` Advantage target was not reached at the 4096-step ceiling. This makes level 2 a credible bounded-estimator candidate rather than merely “more optimizer”.

## Full-exact end-to-end result — why more exact is not always better

Workflow `31412933368`, evidence `ee016e41deba227fff2e99f9926118a8ad219329`, compared authoritative level 0 to effectively full opponent enumeration (`level 128`) at 64 roots/seed.

Authoritative level 0:

- mean `0.300694`
- p95 `0.820360`
- total nodes `28,472`
- all fit gates PASS.

Full exact:

- mean `0.328689`
- p95 `0.826717`
- mean ratio `1.0931` — worse
- p95 ratio `1.00775` — worse
- total nodes `22,245,222`
- node ratio **`781.30x`**
- all fit gates PASS.

The full-exact traversal generated about `1.68M` Advantage samples per seed and saturated the `400k` reservoir even at only 64 roots. Thus full tree expectation removes opponent-action Monte Carlo variance but simultaneously explodes support/memory/function-approximation burden. It is **not** an end-to-end policy-stability upper bound once the finite reservoir and neural approximator are included. The useful regime is bounded variance reduction, not exhaustive enumeration.

## Acceptance-scale promotion of partial exact

Because level 1 and especially level 2 materially improved both mean and tail while full exact crossed into a memory/model bottleneck, workflow `31415196119` now runs two **deck-exact 640-root candidates in parallel**:

- level 1: exact next opponent decision;
- level 2: exact next two opponent decisions.

Both use:

- exact authoritative generation-2 deal schedule;
- 5 x 128 = 640 roots/seed;
- recovered single coupled `batch_rng` state for partial sampling, own-reach sampling and optimizer minibatches;
- strong Advantage fitting target `0.50` with max `4096` steps/iteration;
- AveragePolicy target `0.105`, max `32768` steps;
- reservoir `400000`;
- unchanged frozen gates.

This is an explicit experimental estimator change. Any PASS must be versioned and checkpoint/resume recertified before R7.3 closes.

## Other active causal screens

### Deck-exact coupled Advantage x4 640 V2 — `31414208511`

Corrects the V1 deal-schedule error and measures x4 on the exact authoritative deals. Use V2, not V1, for any paired x4 acceptance conclusion.

### Common-random-number screen — `31413103901`

`independent_1`, `independent_4`, `common_1`, `common_4`. Common modes require byte-identical iteration-1 Advantage memories under shared decks/uniform behavior.

### Five-iteration compounding — `31413646505`

Baseline and Advantage x4 measure freshly fitted regret-matching behavior TV after every CFR iteration to locate when divergence amplifies.

### Antithetic x4 — `31413970227`

Four marginally correct trajectories use quarter-turn shifted Uniform streams. This tests lower-discrepancy correlation at the same x4 path cost.

### Same-memory Advantage fit/sign sensitivity — `31414959700`

Four AdvantageNets train on the **exact same frozen Advantage reservoir** with different init/minibatch seeds. Pairwise raw outputs are converted through the production hard regret-matching rule. The diagnostic measures sign-support disagreement, near-zero regret fragility, policy TV, and post-hoc epsilon-floor sensitivity. This directly tests whether neural fit variance plus hard sign thresholding amplifies a common target memory into different policies.

## Decision tree

- If partial-exact level 1/2 survives five-iteration 640 scale, prefer the cheapest level satisfying both frozen cross-seed gates, then version/checkpoint-recertify it.
- If common-path or antithetic correlation materially outperforms independent paths, combine only if the partial-exact acceptance result still needs more variance reduction.
- If same-memory Advantage fits yield large regret-policy disagreement despite similar NRMSE, the next primary target becomes regret-sign/function-approximation stability, not traversal sampling.
- Full exact is closed as a practical estimator: `781x` tree work plus reservoir saturation with no final stability gain.
- Keep raw unique-root scaling and independent x8/x16 paused.

Historical pre-loss mean/p95 `0.3714 / 0.6878` remains historical evidence only.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful step is persisted directly to `main`.
- Experimental estimator/RNG/sampling changes require explicit versioning and deterministic checkpoint-resume recertification before R7.3 can close.

`READY FOR TABLES = NO`.
