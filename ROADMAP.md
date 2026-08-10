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
    - corrected 640 and strong-Advantage 640 — **FIT PASS / CROSS-SEED FAIL**
    - 1280 unique-root scaling — **FAIL / PAUSED**
    - CFR-memory/chance/support/card/RNG decomposition — **DONE**
    - exact own-reach + exact Advantage path controls — **DONE**
    - bounded partial-opponent expectation — **LEVELS 1/2 MATERIAL AT 256; 640 RUNNING**
    - exhaustive opponent expectation — **DONE / IMPRACTICAL AND WORSE END-TO-END**
    - common-path RNG — **DONE / NOT MATERIAL**
    - antithetic x4 — **DONE / NOT MATERIAL**
    - same-memory Advantage fit/sign sensitivity — **DONE / MATERIAL**
    - Advantage init-versus-minibatch factorial — **DONE / MIXED; BOTH MATERIAL**
    - five-iteration behavior compounding — **BASELINE DONE; x4 RUNNING**
    - common Advantage fit randomness — **RUNNING**
    - Advantage ensemble stability — **RUNNING**
    - deck-exact x4 V2 — **RUNNING**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 convergence
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Frozen R7.3 gates

- Advantage weighted normalized RMSE `<= 0.75`
- AveragePolicy weighted mean TV `<= 0.12`
- cross-seed mean TV `<= 0.15`
- cross-seed p95 TV `<= 0.35`

No gate is relaxed.

## Authoritative paired acceptance control

The generation-2 reference schedule is:

```text
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
```

with `global_root` continuous across CFR iterations.

Corrected 640: `0.477649 / 0.902403` mean/p95, per-seed fits PASS.
Strong-Advantage 640: `0.464474 / 0.886204`, per-seed fits PASS.

Replicated-candidate V1 used a different deterministic deal formula. Its physical results remain valid independent-deal experiments, but not paired deltas against the authoritative references. The correction is recorded in `validation/R7_3_REPLICATED_V1_DECK_CONTROL_CORRECTION_20260810.md`. Deck-exact coupled x4 V2 is running as workflow `31414208511`.

## Current causal model

R7.3 now has **two distinct upstream variance sources**.

### A. CFR target / memory variance

Exact controls prove:

- own-reach Monte Carlo fragments strategy-memory support even under identical exact policy;
- opponent external sampling injects material Advantage target noise;
- x4 independent Advantage paths substantially help a two-iteration screen;
- x8 gives little extra mean benefit against the exact oracle and still has p95 `1.0`;
- common-random-number paths improve only modestly;
- antithetic quarter-turn x4 barely improves mean and worsens p95;
- exhaustive opponent enumeration costs ~`781x` tree nodes, saturates the reservoir, and worsens final policy stability.

The best bounded target-variance result is partial exact level 2 at 256 roots:

| estimator | mean TV | p95 TV | mean ratio | p95 ratio | node ratio |
|---|---:|---:|---:|---:|---:|
| authoritative level 0 | `0.313641` | `0.882657` | 1.000 | 1.000 | 1.000 |
| level 1 | `0.223853` | `0.807704` | `0.7137` | `0.9151` | `2.667x` |
| level 2 | **`0.191695`** | **`0.669413`** | **`0.6112`** | **`0.7584`** | `9.349x` |

Workflow `31415605322` is physically running deck-exact 640 level-1 and level-2 candidates. Workflow `31415642047` runs level-2 with doubled Advantage fit ceiling (`8192`) to distinguish estimator quality from fitting capacity.

### B. AdvantageNet function-approximation / sign-support variance

Workflow `31414959700`, evidence `d0fa315ee79af013a7e7e3294b0877a0e656f820`, trained four AdvantageNets on the **exact same frozen Advantage memory**. Final weighted NRMSEs were all historical-quality (`0.472–0.487`), yet:

- pairwise hard-regret-matching mean TV averaged **`0.224349`**;
- pairwise p95 averaged **`0.757529`**;
- identical positive-regret action support occurred on only **`55.49%`** of observations.

Thus a network can pass the Advantage approximation gate comfortably and still induce a very different CFR behavior policy purely because training finds another approximation with different regret signs.

Workflow `31415792326`, evidence `555df805e5f14814b1f3e742481bcff110d6cc49`, decomposed this on the same memory:

- same initialization, different minibatch order: mean policy TV `0.201320`;
- different initialization, same minibatch order: `0.221737`;
- both different: `0.224532`;
- init/batch ratio `1.1014`;
- diagnosis `ADVANTAGE_INIT_AND_MINIBATCH_VARIANCE_MIXED`.

Neither random initialization nor minibatch order alone explains the problem; both matter.

Post-hoc smoothing confirms hard positive-regret thresholding is an amplifier but not the complete cause. A scale-normalized epsilon `0.10` floor reduced representative same-memory policy TV from `0.2254` to `0.1853`, still far above the R7.3 mean gate.

## Five-iteration behavior divergence

Workflow `31413646505` baseline completed. Freshly fitted AdvantageNets were converted through production regret matching after each iteration. Mean cross-seed behavior TV was:

```text
iteration 1  0.476820
iteration 2  0.578266
iteration 3  0.562001
iteration 4  0.553982
iteration 5  0.538547
```

p95 was `1.0` at **every** iteration. Final AveragePolicy cross-seed mean/p95 was `0.453395 / 0.948678`.

This changes the earlier narrative: the system does not first become unstable only after several feedback iterations. Severe regret-policy divergence is already present at the **first fitted AdvantageNet**, then iteration 2 amplifies it further. This is consistent with the same-memory fit experiment and makes Advantage function-approximation stability a primary R7.3 target, not a secondary detail.

The parallel Advantage-x4 compounding job is still running and will show whether more path samples reduce the iteration-1 fit instability or merely alter later feedback.

## Closed or deprioritized branches

- More unique roots: paused after 640 -> 1280 was essentially flat and worsened policy fit.
- Independent x8/x16: not promoted; exact oracle shows diminishing returns and p95 remains saturated.
- Common path RNG: `common4` mean/p95 `0.288607 / 0.804410` versus baseline `0.313641 / 0.882657`; improvement is too small relative to the problem, diagnosis NOT MATERIAL.
- Antithetic x4: mean ratio `0.97395`, p95 ratio `1.05415`; NOT MATERIAL.
- Full exact opponent expectation: ~`781x` nodes, ~`1.68M` Advantage samples seen/seed at only 64 roots, reservoir saturation, and worse final mean/p95; closed as a practical estimator.
- Card/suit representation rewrite: not dominant in controlled support audit.
- Merely increasing optimizer steps: not sufficient by itself.

## Active high-value physical work

1. **Deck-exact coupled Advantage x4 640 V2** — `31414208511`.
2. **Partial-exact level 1 + level 2 640** — `31415605322`.
3. **Partial-exact level 2 strong-fit 640** — `31415642047`.
4. **Advantage-x4 five-iteration compounding** — `31413646505`.
5. **Common Advantage fit randomness** — `31415931101`; same Advantage init + same per-iteration minibatch RNG across seeds, traversal/final-policy RNG still seed-specific.
6. **Advantage ensemble stability** — `31416468310`; eight independent same-memory fits, disjoint 1/2/4-model ensembles average raw Advantage predictions before unchanged hard regret matching.

## Decision tree

- If partial exact survives 640, choose the cheapest level that satisfies both cross-seed gates; then version the estimator and recertify checkpoint/resume determinism.
- If common deterministic Advantage fitting materially suppresses cross-seed behavior, make fit randomness deterministic as an explicit algorithm contract rather than multiplying traversals.
- If ensembling collapses same-memory policy variance while retaining/improving NRMSE, test the smallest useful ensemble end-to-end; model averaging is then a candidate variance-control mechanism.
- If neither common fit nor ensembling sufficiently fixes same-memory hard-regret policy variance, test sign-aware calibration/objectives or a rigorously validated continuous regret mapping; do not relax gates.
- Combine partial exact with fit-stability changes only after each mechanism proves independent value; avoid stacking expensive changes without attribution.

Historical pre-loss `0.3714 / 0.6878` remains historical evidence only and is not substituted for generation-2 gates.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful recovery/evolution step is persisted directly to `main`.
- Experimental estimator, RNG, ensemble or regret-map changes require explicit versioning and deterministic continuous-vs-stop/restore/continue recertification before R7.3 can close.

`READY FOR TABLES = NO`.
