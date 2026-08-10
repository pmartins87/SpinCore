# SpinCore finite roadmap — canonical recovery generation 2

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate passes.

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

## Authoritative acceptance contract

Generation-2 deal schedule:

```text
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
```

with `global_root` continuous across CFR iterations. The recovered acceptance path preserves one persistent live `bundle.batch_rng` through collection and primary training in execution order unless a diagnostic explicitly declares otherwise.

Authoritative 640 references:

| candidate | mean TV | p95 TV | fit gates |
|---|---:|---:|---|
| corrected 640 | `0.477649` | `0.902403` | PASS |
| strong-Advantage 640 | `0.464474` | `0.886204` | PASS |
| partial-exact level 1, 640 | `0.437426` | `0.890053` | PASS |
| partial-exact level 2, 640 | `0.436110` | `0.888244` | PASS |
| partial-exact level 2 strong-fit, 640 | **`0.412893`** | **`0.871708`** | PASS |

Partial opponent expectation helps, but by itself is nowhere near the frozen cross-seed gates.

## Current paired 256 control

The within-run authoritative partial-exact level-2 control is:

```text
mean TV = 0.2456560284
p95 TV  = 0.6287055612
fit gates = PASS
```

This exact size-1 result was reproduced independently by the final-AveragePolicy-ensemble diagnostic, giving a deterministic semantic cross-check.

### Raw Advantage ensemble

| size | mean TV | p95 TV | fit gates |
|---:|---:|---:|---|
| 1 | `0.245656` | `0.628706` | PASS |
| 2 | `0.215379` | `0.632143` | PASS |
| 4 | `0.210869` | `0.627987` | PASS |

Raw model averaging improves the center by ~14% at size 4 but leaves p95 essentially unchanged. **Raw Advantage ensembling is not a primary solution.**

### Policy-mixture Advantage ensemble — first materially tail-sensitive paired candidate

Instead of averaging raw Advantage values before regret matching, size-4 policy mixture applies the production hard-regret map to each independently fitted AdvantageNet first and then averages the resulting legal-action probabilities.

Authoritative paired 256 physical result, workflow `31428299914`, evidence `76ac23287961cc3c650a1b60891648bfe975b145`:

```text
size-1 baseline:
mean = 0.2456560284
p95  = 0.6287055612

size-4 policy mixture:
mean = 0.1719404310
p95  = 0.4136051536
p50  = 0.1451251805
max  = 0.7699102759
```

Relative to the paired control:

- mean ratio `~0.6999` (~30.0% reduction);
- p95 ratio `~0.6579` (~34.2% reduction).

Both per-seed fit gates pass. Final ensemble Advantage NRMSEs are `0.501434` and `0.542430`; final AveragePolicy weighted TVs are `0.096840` and `0.094627`.

This is the first theory-conservative candidate in generation 2 to reduce **both** the center and tail strongly under the authoritative deck/primary-RNG contract. It still fails the frozen cross-seed gates (`0.17194 > 0.15`, `0.41361 > 0.35`) and is not production-promoted.

Because the effect may decay as CFR feedback compounds, the next escalation is deliberately **not 640**. Workflow `31432403037` runs five CFR iterations × 64 roots = 320 roots/seed with the same size-4 policy-mixture semantics. Promotion to an acceptance-scale test requires the improvement to survive this longer feedback horizon.

### Final AveragePolicy ensemble

| size | mean TV | p95 TV |
|---:|---:|---:|
| 1 | `0.245656` | `0.628706` |
| 2 | `0.226901` | `0.598557` |
| 4 | `0.212912` | `0.599173` |

All fit gates pass. Four final policy models improve mean ~13.3% but p95 only ~4.7%. Final-policy approximation is real but not the dominant tail source; this is retained only as a possible secondary component.

## Strongest causal finding: target divergence already exists before final-policy fitting

Authoritative partial-exact level-2 support forensic, 256 roots:

### Exact/raw infosets

- Jaccard support overlap: `0.0046825`
- mean LCFR-weight mass coverage: `0.0094187`
- shared-target weighted mean TV: `0.329293`
- shared-target p95 TV: `0.750000`

### Poker-isomorphic comparison key

- Jaccard: `0.0208888`
- mean LCFR-weight mass coverage: `0.0785715`
- shared-target weighted mean TV: `0.349296`
- shared-target p95 TV: `0.803535`

Thus the two seeds share very little Strategy Memory support even after suit/hole/flop structural canonicalization, and **the sigma targets themselves already disagree strongly on the small shared subset**. The remaining p95 cannot be attributed mainly to final AveragePolicy training.

A prior controlled iteration-split physical experiment already showed the mechanism sharply: with exact zero-regret uniform behavior, iteration-1 shared sigma targets had TV exactly `0`; after the first fitted Advantage behavior, iteration-2 shared-target p95 jumped to `1.0` and weighted mean TV to about `0.528` in poker-isomorphic mode. An authoritative partial-exact iteration-split replication is running as workflow `31431939967`.

## Advantage function-approximation / regret-map findings

Same-memory AdvantageNet fitting remains an independent variance source:

- hard-RM pairwise mean TV around `0.22–0.23`;
- p95 around `0.76–0.82` despite good NRMSE;
- initialization and minibatch order both matter.

Legal-action common-mode centering was tested and closed:

```text
raw size-1 mean/p95       = 0.231120 / 0.803678
best centering (midrange) = 0.219685 / 0.824299
```

Mean improves slightly while p95 worsens; diagnosis `ADVANTAGE_COMMON_MODE_CENTERING_NOT_MATERIAL`.

## Direct behavior surrogate: large same-memory signal, algorithmically non-equivalent

A direct policy surrogate trained on sample-level regret-matched Advantage targets reduced same-memory independent-fit disagreement from:

```text
AdvantageNet -> hard RM:
mean = 0.230515
p95  = 0.818211

Direct behavior surrogate:
mean = 0.110418
p95  = 0.363998
```

Ratios: mean `0.4790`, p95 `0.4449`. Diagnosis: `DIRECT_BEHAVIOR_SURROGATE_MATERIAL_SAME_MEMORY`.

This is **not assumed theoretically equivalent to Deep CFR** because regret matching is nonlinear. Workflow `31431672631` has passed build/regression and smoke and is physically testing the 256-root E2E causal effect while leaving production semantics unchanged.

## Active high-value physical work

1. **Five-iteration policy-mixture compounding** — workflow `31432403037`, 5 × 64 = 320 roots/seed. This decides whether the strongest paired candidate survives a longer CFR feedback horizon before any 640 escalation.
2. **Direct behavior surrogate E2E** — workflow `31431672631`; physical 256-root candidate running after smoke PASS. It is algorithmically experimental and cannot be promoted without a separate semantics/versioning review.
3. **Authoritative partial-exact support by iteration** — workflow `31431939967`; physical 256-root forensic running after smoke PASS to reproduce the iteration-1-zero / iteration-2-divergence causal chain under the current paired contract.

## Closed / deprioritized primary branches

- raw root scaling beyond 1280;
- independent x8/x16 path multiplication;
- common-path RNG;
- antithetic x4;
- exhaustive opponent expectation;
- merely raising Advantage optimizer ceiling;
- behavior-aware MSE auxiliary objective (`V2` valid result improved mean but failed Advantage fit and barely moved p95);
- exact weighted duplicate-target aggregation;
- behavior-aware multistart model selection;
- raw Advantage ensemble sizes 2/4 as standalone solution;
- final AveragePolicy ensemble as standalone solution;
- legal-action common-mode centering;
- card/suit representation rewrite as the dominant explanation.

## Immediate decision tree

- If five-iteration **policy mixture** retains a large reduction in both mean and p95 with fit gates PASS, it earns the next acceptance-scale 640 test; if the effect collapses with feedback depth, do not brute-force it.
- If **direct behavior E2E** gives a very large tail reduction, do not promote immediately: first formalize its algorithmic objective, prove the intended regret semantics or explicitly version it as a new algorithm, then add checkpoint/resume determinism before acceptance-scale testing.
- If the authoritative iteration split confirms zero shared-target disagreement in iteration 1 and large disagreement in iteration 2, the causal chain is pinned to first Advantage fit/regret-map instability feeding back into trajectories; subsequent work should stabilize that transition rather than smooth the final AveragePolicy.
- No estimator, objective, ensemble, RNG or mapping change becomes production semantics without explicit versioning and deterministic continuous-vs-stop/restore/continue recertification.

Historical pre-loss `0.3714 / 0.6878` remains historical evidence only and is never substituted for generation-2 gates.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful recovery/evolution step is persisted directly to `main`.

`READY FOR TABLES = NO`.
