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

```text
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
```

`global_root` is continuous across CFR iterations. The recovered acceptance path preserves one persistent live `bundle.batch_rng` through collection and primary training in execution order unless a diagnostic explicitly declares otherwise.

## Causal transition — confirmed

Under authoritative partial-exact level 2, iteration-1 shared strategy targets are identical (`mean/p95 TV = 0/0`). Immediately after the first fitted Advantage behavior feeds back into collection, iteration 2 reaches:

```text
shared-target weighted mean TV = 0.473946
shared-target p95 TV           = 1.0
```

The dominant known transition is therefore:

```text
Advantage approximation -> regret map -> behavior -> sampled trajectories -> next strategy targets
```

Final AveragePolicy approximation is downstream of already-divergent targets.

## Policy-mixture size 4 — short-horizon success, durability failure

Authoritative paired 2×128:

```text
mean TV = 0.171940
p95 TV  = 0.413605
fit gates = PASS
```

Mandatory 5×64 durability run `31432403037`:

```text
mean TV = 0.266591
p50 TV  = 0.246805
p95 TV  = 0.567002
max TV  = 0.905547
fit gates = PASS
```

The short-horizon improvement decays materially. **The prepared size-4 640 workflow remains dormant and must not be launched.** A new mechanism must first beat `0.266591 / 0.567002` at the five-iteration horizon.

## Direct Behavior — strong smoothing clue, not recovered Deep CFR

At 2×128, Direct Behavior reached:

```text
mean TV = 0.142553  PASS
p95 TV  = 0.426860  FAIL
```

Its surrogate itself underfits the sample-level regret-matched targets, and theoretical equivalence to Deep CFR is not claimed. A new 5×64 durability control (`31441650915`) is running solely to determine whether the smoothing effect survives repeated feedback.

## Static residual-tail program

### Policy-mixture size 8

Corrected run `31440425854`: build/regression PASS, smoke PASS, physical paired 256-root candidate running. A size-8 win still requires its own 5×64 durability test before any scale-up.

### Policy-mixture + final AveragePolicy ensemble

Run `31440493410`: build/regression PASS, smoke PASS, physical 256-root factorial running. CFR uses size-4 policy mixture; final AveragePolicy sizes 1/2/4 are fitted only after collection is frozen.

### Support-conditioned tail forensic

Run `31440576227`: build/regression PASS, smoke PASS, physical 256-root forensic running. It separates disagreement on seed-A support, seed-B support, exact byte-identical shared SPNNIV1 observations, and exact one-sided observations.

### Robust probability aggregation — CLOSED

Run `31440742014`, evidence `b9a756af32341e7e6d51047ff584c94e606c8dce`:

```text
ordinary mean: mean 0.137421 / p95 0.388113
median:        mean 0.143639 / p95 0.488827
trimmed mean: mean 0.143639 / p95 0.488827
```

Median/trimmed aggregation worsens the p95 by ~26%. Diagnosis `ROBUST_POLICY_AGGREGATION_NOT_MATERIAL`. Rare individual-member outliers are not the primary explanation; this branch is closed.

### Behavior-target aggregation order

Run `31441852607`: regression + smoke PASS, physical same-memory screen running. It compares:

1. hard regret matching per stored Advantage sample;
2. weighted mean of those sample-level policies per exact observation/legal mask;
3. hard regret matching of the weighted mean regret vector per exact observation/legal mask.

This tests whether the Direct Behavior surrogate's target construction itself is injecting avoidable nonlinearity/noise. No production algorithm change is assumed.

## Five-iteration temporal-damping program

All candidates below are tested directly at the failing 5×64 horizon, not on another short proxy.

### Decaying uniform tremble — `31441018067`

Three size-4 jobs, all smoke PASS and physical 320 running:

```text
epsilon0 = 0.15 / 0.30 / 0.45
decay    = 0.50 per fit
pi_used  = (1-epsilon_k)*pi_policy_mixture + epsilon_k*uniform
```

### Ensemble × tremble 2×2 factorial — `31441110526`

Physical size-1 jobs running after smoke PASS:

```text
size1, epsilon0 = 0.00
size1, epsilon0 = 0.30, decay=.50
```

Together with size4/no-tremble and size4/e30, this isolates the contributions of ensembling versus explicit damping.

### Previous-policy temporal blending — `31441224117`

Two size-4 physical 320 jobs running after smoke PASS:

```text
current weight = 0.50 / 0.75
```

The first reference is exact uniform; later references are the previous iteration's fitted policy mixture. This tests whether abrupt model replacement drives the feedback instability.

### First-transition-only damping — `31441567261`

Smoke PASS; physical 320 running:

```text
epsilon schedule = [0.30, 0, 0, 0]
```

If this matches repeated damping, the first break seeds most later divergence. If repeated damping wins but this fails, stabilization is needed across multiple iterations.

### Direct Behavior durability control — `31441650915`

Regression + smoke PASS; physical 320 running. This remains a causal control only, not a promotion candidate.

## Reproducibility and automatic consolidation

Fresh duplicate physical runs are retained as determinism checks:

- paired size4 256: `31440366942`, expected `0.171940 / 0.413605`;
- size4 320 compounding: `31440366909`, expected `0.266591 / 0.567002`.

`tools/check_r7_3_fresh_run_reproducibility.py` plus workflow `r7_3_fresh_run_reproducibility.yml` automatically compare the latest two evidence commits, requiring exact structural/sample/node counters and cross-seed metrics within `1e-9`. This is fresh-run determinism only; checkpoint/resume recertification remains mandatory for any new production semantic.

`tools/summarize_r7_3_durability_matrix.py` plus `r7_3_durability_matrix_summary.yml` automatically consolidate the completed five-iteration matrix. Direct Behavior is included as a causal control but excluded from conservative automatic promotion because equivalence is unproven.

Detailed design record: `validation/R7_3_DURABILITY_PROGRAM_20260810.md`.

## Closed / deprioritized primary branches

- raw root scaling beyond 1280;
- independent x8/x16 path multiplication;
- common-path RNG;
- antithetic x4;
- exhaustive opponent expectation;
- merely raising Advantage optimizer ceiling;
- behavior-aware MSE auxiliary objective;
- exact weighted duplicate-target aggregation;
- behavior-aware multistart selection;
- raw Advantage ensemble 2/4 standalone;
- final AveragePolicy ensemble standalone;
- legal common-mode centering;
- robust median/trimmed policy aggregation;
- card/suit rewrite as dominant explanation;
- direct size4 policy-mixture 640 escalation.

## Promotion rule

A candidate does not advance merely by ranking first. Before 640 it must:

1. PASS every frozen per-seed fit gate;
2. materially improve **both** mean and p95 versus the 5×64 durability baseline `0.266591 / 0.567002`;
3. survive fresh-run reproducibility checks;
4. be the smallest/interpretable mechanism among statistically comparable candidates;
5. have any new behavior semantics explicitly versioned;
6. pass deterministic continuous-vs-stop/restore/continue checkpoint recertification;
7. keep the frozen acceptance gates unchanged.

`READY FOR TABLES = NO`.
