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
    - four-mode path replication screen — **DONE: Advantage external-sampling variance strongest isolated path lever**
    - exact opponent-expectation Advantage oracle — **DONE**
    - replicated-candidate V1 x4 runs — **DONE / ALL CROSS-SEED FAIL / NOT DECK-IDENTICAL TO REFERENCE**
    - deck-control V1 audit — **DONE: V1 formula mismatch found and documented**
    - deck-exact coupled Advantage x4 640 V2 — **RUNNING**
    - partial-exact V1 — **INVALID DIAGNOSTIC CONTROL; solver/regressions PASS**
    - partial-exact V2 — **RUNNING**
    - full-exact opponent-expectation upper bound — **RUNNING**
    - common-random-number Advantage path screen — **RUNNING**
    - five-iteration divergence/compounding diagnostic — **RUNNING IN PARALLEL: baseline + Advantage x4**
    - antithetic/rotated-lattice Advantage x4 screen — **RUNNING**
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

## Acceptance references

Corrected generation-2 640, workflow `31348997168`, evidence `7275ada20279c6b18d93d0539c6b44989632a605`:

- mean TV `0.477649`
- p95 `0.902403`
- both seeds pass Advantage and AveragePolicy fit gates.

Strong-Advantage 640, workflow `31364029367`, evidence `75885791b5d8894d4c590a038233f96db5879925`:

- mean `0.464474`
- p95 `0.886204`
- both individual fits pass.

## Replicated-candidate V1 results and control correction

The three physical V1 x4 runs remain valid experiments and all failed the frozen cross-seed gates:

| V1 candidate | mean TV | p95 TV | individual fits | evidence |
|---|---:|---:|---|---|
| separated Advantage x4 | `0.459596` | `0.898250` | PASS | `94b5e423fa51e1dad8445e6ce36b8832d8161648` |
| separated both x4 | `0.458853` | `0.908883` | PASS | `871967f777f7cec17479ed3ec9f476543452912d` |
| coupled Advantage x4 | `0.451112` | `0.893292` | PASS | `87547311076fd6a015b7d855de1a9c26124b924f` |

A post-run line-by-line audit found an experimental-control error: V1 **did not use the exact authoritative deal schedule** despite its comment/JSON saying it did.

Authoritative generation-2 schedule:

```text
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
```

with `global_root` continuous across all five iterations.

V1 instead used:

```text
deck_seed = (seed << 32) ^ (iteration << 16) ^ root_index_within_iteration
```

and restarted `root_index` every iteration.

Consequences:

- the V1 metrics remain evidence that those x4 configurations fail badly on independent hidden-deal samples;
- they **cannot** support a tightly paired percentage-improvement claim against corrected/strong-Advantage 640;
- the broad finding that x4 did not approach the gates remains true;
- the exact x4 delta versus the authoritative reference is being remeasured by V2.

Correction record: `validation/R7_3_REPLICATED_V1_DECK_CONTROL_CORRECTION_20260810.md`.

## Deck-exact x4 V2

`tools/run_r7_3_replicated_640_candidate_v2.py` replaces only the V1 deal scheduler while reusing the smoke-certified collection/fitting implementation. V2 self-checks cross-iteration global-root continuity and records the exact formula in its JSON.

The highest-priority corrected candidate is **coupled Advantage x4**, because if x4 is useful this variant retains the recovered RNG-state structure. Its workflow uses:

- exact authoritative `seed*1000003 + global_root*97 + iteration` deals;
- 5 x 128 = 640 unique roots/seed;
- four Advantage trajectories / one strategy trajectory;
- recovered coupled RNG contract;
- Advantage fit target `0.50`;
- AveragePolicy fit target `0.105`, max `32768` steps;
- reservoir `400000`;
- unchanged frozen gates.

This V2 run is active; V1 is not substituted for it.

## Causal evidence retained

1. CFR-memory variance dominates AveragePolicy optimizer/init variance.
2. Root card/deck variation is not dominant in the dedicated shared-deck experiment.
3. Off-support AveragePolicy extrapolation is material but downstream.
4. Card/suit representation alone is not dominant.
5. Training/traversal RNG bookkeeping coupling alone is not dominant.
6. Stronger Advantage fitting helps but is insufficient.
7. Own-reach Monte Carlo provably fragments strategy support under an identical exact policy.
8. Advantage external sampling provably injects target/regret-matching noise.
9. Four Advantage paths materially improve a controlled two-iteration screen.
10. The exact Advantage oracle shows large 1->4 benefit but small 4->8 incremental benefit, with p95 still saturated at `1.0`.

The V1 deck correction does **not** invalidate items 1–10 because each came from separate controlled workflows. It only weakens paired interpretation of the three V1 640 x4 deltas.

## Active estimator-design experiments

### Partial-exact V2 — workflow `31412806987`

Level 0 is now the authoritative recovered `ExternalSamplingCollector`; only levels 1/2 are experimental. They enumerate one/two upcoming opponent decisions, probability-weight downstream Advantage samples, then return to ordinary external sampling.

### Full-exact opponent upper bound — workflow `31412933368`

A bounded experiment asks how much stability is possible if opponent-action sampling variance is effectively removed entirely. This is an upper-bound diagnostic, not a production proposal.

### Common-random-number screen — workflow `31413103901`

`independent_1`, `independent_4`, `common_1`, `common_4`. Under shared decks plus iteration-1 uniform behavior, common modes must create byte-identical Advantage memories across seeds or fail closed.

### Iteration compounding — workflow `31413646505`

Baseline and Advantage x4 run five CFR iterations. After each refit, freshly fitted AdvantageNets are converted through exact regret matching on a common iteration corpus. This directly locates when cross-seed regret-policy divergence accelerates. Both jobs passed smoke and entered the physical phase.

### Antithetic x4 — workflow `31413970227`

Four marginally correct external-sampling trajectories share the same underlying Uniform sequence with quarter-turn offsets. This tests lower-discrepancy/antithetic correlation at the same four-path cost rather than blindly increasing path count.

## Decision tree

- First obtain the **deck-exact V2 x4** result before making any paired acceptance-scale claim about x4.
- If common random numbers materially suppress divergence, promote a versioned counter-based/common-path estimator.
- If partial/full expectation is stronger, promote the cheapest bounded enumeration level that captures most of the upper-bound gain.
- If antithetic x4 materially beats independent x4, use the correlated estimator rather than more independent paths.
- Use iteration-compounding evidence to target the exact refit where divergence amplifies.
- If full exact is weak, pivot to regret-sign sensitivity, policy-support discontinuity and target aggregation/control-variate design.
- Do **not** resume brute-force roots or x8/x16 independent replication without new causal evidence.

Historical pre-loss mean/p95 `0.3714 / 0.6878` remains historical only.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful step is persisted directly to `main`.
- Experimental estimator/RNG/sampling changes require explicit versioning and deterministic checkpoint-resume recertification before R7.3 can close.

`READY FOR TABLES = NO`.
