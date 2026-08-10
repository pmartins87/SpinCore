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

### Raw Advantage ensemble

| size | mean TV | p95 TV | fit gates |
|---:|---:|---:|---|
| 1 | `0.245656` | `0.628706` | PASS |
| 2 | `0.215379` | `0.632143` | PASS |
| 4 | `0.210869` | `0.627987` | PASS |

Raw averaging improves the center but not the tail, so it remains closed as a primary solution.

## Policy-mixture Advantage ensemble — strongest conservative candidate

Each independently fitted AdvantageNet is first converted through hard regret matching and the resulting legal-action policies are averaged. Authoritative paired 256 result, workflow `31428299914`, evidence `76ac23287961cc3c650a1b60891648bfe975b145`:

```text
baseline size 1:
mean = 0.2456560284
p95  = 0.6287055612

policy-mixture size 4:
mean = 0.1719404310
p95  = 0.4136051536
p50  = 0.1451251805
max  = 0.7699102759
```

Mean falls ~30.0% and p95 ~34.2%, with both per-seed frozen fit gates PASS. This is the first generation-2 candidate to materially reduce both center and tail while staying close to the recovered Deep-CFR behavior semantics. It still fails the frozen cross-seed gates (`0.17194 > 0.15`, `0.41361 > 0.35`).

Workflow `31432403037` is physically running a five-iteration × 64-root = 320-root/seed compounding screen. This deliberately precedes any 640 escalation: the candidate must first show that its gain survives a longer feedback horizon.

## Causal transition now physically confirmed

Authoritative partial-exact support-by-iteration workflow `31431939967`, evidence `086df6ed397ecc4a2e61728aa3b40f0f58593675`, reproduced the expected causal break under the current paired contract.

### Iteration 1 — before any fitted AdvantageNet can affect behavior

Poker-isomorphic shared Strategy Memory:

```text
Jaccard                      = 0.0125865
shared-target weighted TV    = 0.0000000
shared-target p95 TV         = 0.0000000
```

The shared sigma targets are exactly identical.

### Iteration 2 — after the first Advantage fit/regret map feeds behavior back into collection

```text
Jaccard                      = 0.0284171
shared-target weighted TV    = 0.4739458
shared-target p95 TV         = 1.0000000
```

Diagnosis: `TARGET_DIVERGENCE_APPEARS_AFTER_FIRST_ADVANTAGE_FIT`.

This pins the main generation-2 instability to the transition:

```text
Advantage approximation -> regret mapping -> sampled behavior -> next CFR trajectories -> next strategy targets
```

The final AveragePolicy is downstream of an already-divergent target distribution and cannot be the primary cause.

## Final AveragePolicy ensemble — secondary only

| size | mean TV | p95 TV |
|---:|---:|---:|
| 1 | `0.245656` | `0.628706` |
| 2 | `0.226901` | `0.598557` |
| 4 | `0.212912` | `0.599173` |

All fits pass, but size 4 only reduces p95 ~4.7%. Final-policy approximation remains a secondary component.

## Direct behavior surrogate E2E — strong causal signal, not promotable as-is

Same-memory evidence had reduced independent-fit behavior disagreement from `0.230515 / 0.818211` to `0.110418 / 0.363998` mean/p95. The authoritative E2E run then completed successfully: workflow `31431672631`, evidence `1d1ac9b23abd2afedec8390e3f8bb482c11b0625`.

```text
baseline partial-exact:
mean = 0.245656
p95  = 0.628706

Direct Behavior E2E:
mean = 0.142553   PASS against 0.15
p95  = 0.426860   FAIL against 0.35
p50  = 0.100895
max  = 0.870134
```

The ordinary frozen Advantage/final-AveragePolicy fits pass. However, the surrogate itself does **not** faithfully fit the sample-level regret-matched behavior target within the reference `0.12` TV threshold: for seed `20260829`, its iteration-1/iteration-2 audit TVs are about `0.2723 / 0.2205` even after the 4096-step ceiling. Therefore the stability gain is at least partly a smoothing/regularization effect rather than evidence that the surrogate is a faithful implementation of the recovered regret behavior.

Because regret matching is nonlinear, theoretical equivalence is not claimed. This branch is retained as a causal clue — **smoothing the first feedback transition can strongly improve stability** — but it is not eligible for production promotion as currently defined.

## Other closed / deprioritized primary branches

- raw root scaling beyond 1280;
- independent x8/x16 path multiplication;
- common-path RNG;
- antithetic x4;
- exhaustive opponent expectation;
- merely raising Advantage optimizer ceiling;
- behavior-aware MSE auxiliary objective;
- exact weighted duplicate-target aggregation;
- behavior-aware multistart model selection;
- raw Advantage ensemble sizes 2/4 as standalone solution;
- final AveragePolicy ensemble as standalone solution;
- legal-action common-mode centering;
- card/suit representation rewrite as dominant explanation.

## Immediate decision tree

1. Complete workflow `31432403037`.
2. If five-iteration policy mixture retains a large reduction in both mean and p95 with frozen fit gates PASS, launch the 640 acceptance-scale candidate under exactly the same authoritative deal/RNG contract.
3. If the policy-mixture gain collapses with feedback depth, do not brute-force more roots. Use the direct-behavior result only as evidence that behavior smoothing is valuable, then test an explicit and auditable smoothing/interpolation mechanism around the first Advantage-fit transition rather than promoting an opaque underfit surrogate.
4. Even if direct behavior were to meet both cross-seed gates later, it requires an explicit algorithm/versioning decision plus checkpoint/resume determinism before acceptance-scale production status.
5. No estimator, objective, ensemble, RNG or policy mapping becomes production semantics without recertification. No gate relaxation.

Historical pre-loss `0.3714 / 0.6878` remains historical evidence only and is never substituted for generation-2 gates.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful recovery/evolution step is persisted directly to `main`.

`READY FOR TABLES = NO`.
