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

The new within-run authoritative partial-exact level-2 control is:

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

Raw model averaging improves the center by ~14% at size 4 but leaves the p95 essentially unchanged. **Raw Advantage ensembling is therefore not a primary solution.**

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

At this scale all genuinely shared strategy-target keys in the forensic were preflop (`street=0`); exact postflop support overlap across the two independent deck streams was effectively absent under these comparison keys. Four-legal-action shared states were noisier than three-action states (`weighted mean TV ~0.3805` vs `~0.3177` in poker-isomorphic mode).

## Advantage function-approximation / regret-map findings

Same-memory AdvantageNet fitting remains an independent variance source:

- hard-RM pairwise mean TV around `0.22–0.23`;
- p95 around `0.76–0.82` despite good NRMSE;
- initialization and minibatch order both matter.

Legal-action common-mode centering was tested and closed:

```text
raw size-1 mean/p95      = 0.231120 / 0.803678
best centering (midrange)= 0.219685 / 0.824299
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

This is **not assumed theoretically equivalent to Deep CFR** because regret matching is nonlinear. It has therefore advanced only to an explicitly experimental end-to-end causal screen that preserves the authoritative partial-exact collection, primary RNG stream, Advantage fit gate and final policy gate while isolating surrogate training on a side RNG.

## Active high-value physical work

1. **Paired partial-exact size-4 policy mixture** — workflow `31428299914`; physical 256-root run active. Each Advantage member is regret-matched independently before policy probabilities are averaged. Same-memory evidence had reduced size-4 p95 from `0.472424` (raw averaging) to `0.360312` with this nonlinear mapping.
2. **Direct behavior surrogate E2E** — workflow `31431672631`; launched after strong same-memory result. Build/regression/smoke precede the physical 256-root candidate; no production algorithm change is implied.

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

- If paired **policy mixture** materially reduces both mean and p95 while fit gates pass, test the smallest useful mixture over more CFR iterations before any 640 escalation.
- If **direct behavior E2E** gives a very large tail reduction, do not promote immediately: first formalize its algorithmic objective, prove the intended regret semantics or explicitly version it as a new algorithm, then add checkpoint/resume determinism before acceptance-scale testing.
- If neither changes the p95 enough, the new support forensic says the next intervention must target **CFR target/trajectory stability**, not merely final-policy smoothing. Candidate directions are regret-target aggregation/control variates or a rigorously justified smooth behavior update that operates before trajectory divergence compounds.
- No estimator, objective, ensemble, RNG or mapping change becomes production semantics without explicit versioning and deterministic continuous-vs-stop/restore/continue recertification.

Historical pre-loss `0.3714 / 0.6878` remains historical evidence only and is never substituted for generation-2 gates.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful recovery/evolution step is persisted directly to `main`.

`READY FOR TABLES = NO`.
