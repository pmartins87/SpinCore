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
    - four-mode path replication screen — **DONE: Advantage external-sampling variance is the strongest isolated path lever**
    - exact opponent-expectation Advantage oracle — **DONE**
    - three x4 acceptance-scale candidates — **DONE: ALL CROSS-SEED FAIL**
    - partial-exact V1 — **INVALID DIAGNOSTIC CONTROL; solver/regressions PASS**
    - partial-exact V2 — **RUNNING**
    - full-exact opponent-expectation upper bound — **RUNNING**
    - common-random-number Advantage path screen — **RUNNING**
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

## Acceptance-scale evidence

| candidate | mean TV | p95 TV | individual fits | result |
|---|---:|---:|---|---|
| corrected 640 | `0.477649` | `0.902403` | PASS | FAIL |
| strong-Advantage 640 | `0.464474` | `0.886204` | PASS | FAIL |
| separated Advantage x4 640 | `0.459596` | `0.898250` | PASS | FAIL |
| separated both x4 640 | `0.458853` | `0.908883` | PASS | FAIL |
| recovered-coupled Advantage x4 640 | **`0.451112`** | `0.893292` | PASS | FAIL |

Physical x4 evidence:

- separated `advantage_x4`: workflow `31368447316`, commit `94b5e423fa51e1dad8445e6ce36b8832d8161648`;
- separated `both_x4`: workflow `31368447316`, commit `871967f777f7cec17479ed3ec9f476543452912d`;
- coupled `advantage_x4`: workflow `31368894934`, commit `87547311076fd6a015b7d855de1a9c26124b924f`.

All three x4 candidates cleanly passed Advantage and AveragePolicy fit gates. The coupled x4 schedule produced the best mean (`0.451112`), but even that is still roughly 3x the frozen `0.15` mean gate and its p95 (`0.893292`) is essentially unchanged from the failing baseline tail. Therefore **plain independent path replication is not an acceptance-scale solution**. The large improvement seen in the two-iteration 256-root screen does not survive five CFR iterations at 640 roots, which points to variance/divergence compounding through the iterative regret-learning dynamics.

`both_x4` is especially unattractive: it costs substantially more policy collection/training and does not improve the tail. No x8 acceptance run is promoted because the exact Advantage oracle already showed strong diminishing returns from 4 to 8 independent paths while p95 remained saturated at 1.0.

## Causal evidence retained

The investigation has established:

1. CFR-memory variance dominates AveragePolicy optimizer/init variance.
2. Root card/deck variation is not dominant.
3. Off-support AveragePolicy extrapolation is material but downstream.
4. Card/suit representation alone is not dominant.
5. Training/traversal RNG bookkeeping coupling alone is not dominant.
6. Stronger Advantage fitting helps but is insufficient.
7. Own-reach sampling provably fragments strategy support under an identical exact policy.
8. Advantage external sampling provably injects target/regret-matching noise.
9. Four independent Advantage paths materially improve a short controlled screen, but that gain collapses at five-iteration acceptance scale.
10. Exact opponent expectation provides a small-scale oracle; 1 -> 4 sampled paths gives a large gain versus exact, while 4 -> 8 gives little extra and leaves the p95 tail unresolved.

## Active estimator-design experiments

### Partial-exact V2 — workflow `31412806987`

The first partial-exact workflow (`31369138285`) is **not valid evidence about levels 1/2** because its experimental reimplementation of level 0 did not reproduce the persisted baseline exactly. Build, CTest and 26 Python tests passed; the failure was the diagnostic-control assertion itself.

V2 removes that ambiguity. Level 0 is now executed through the authoritative recovered `ExternalSamplingCollector`; only positive levels use the experimental partial-exact estimator. Levels 1 and 2 enumerate the next one/two opponent decisions, probability-weight downstream Advantage samples, then return to ordinary external sampling. The physical 256-root V2 run is active.

### Full-exact opponent upper bound — workflow `31412933368`

A bounded 64-root/seed experiment compares the authoritative estimator against effectively full opponent-action enumeration (`exact level 128`, safely beyond the observed max depth). Its purpose is to answer the decisive question: **if opponent external-sampling variance is removed entirely, how much cross-seed instability remains?** This is an upper-bound diagnostic, not a proposed production schedule.

### Common-random-number path screen — workflow `31413103901`

Modes `independent_1`, `independent_4`, `common_1`, `common_4` test whether synchronizing opponent-action random numbers across algorithm seeds suppresses iterative divergence more efficiently than adding independent paths. Under iteration-1 uniform behavior and shared decks, common-path Advantage memories are required to be byte-identical across seeds; the workflow fails if that invariant does not hold.

## Decision tree after active runs

- If common random numbers materially reduce cross-seed TV, promote a counter-based/common-path RNG estimator as a **versioned** candidate and recertify checkpoint/resume semantics.
- If partial/full exact opponent expectation is materially stronger, promote the cheapest bounded enumeration level that captures most of the gain.
- If full exact still leaves large divergence, stop treating external-sampling opponent variance as sufficient explanation and move directly to iteration-to-iteration regret-policy instability / neural regret-sign sensitivity and target aggregation diagnostics.
- Do **not** resume brute-force unique-root scaling or x8/x16 replication without new causal evidence.

Historical pre-loss 640 mean/p95 `0.3714 / 0.6878` remains historical evidence only and is not substituted for the generation-2 acceptance gate.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful step is persisted directly to `main`.
- Experimental estimator/RNG/sampling changes require explicit versioning and deterministic checkpoint-resume recertification before R7.3 can close.

`READY FOR TABLES = NO`.
