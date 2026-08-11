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

`global_root` remains continuous across CFR iterations. The recovered acceptance path preserves one persistent live `bundle.batch_rng` through primary collection/training in execution order unless a diagnostic explicitly declares otherwise.

## What is now known causally

The instability is seeded upstream of final AveragePolicy fitting. Under authoritative partial-exact level 2, exact shared strategy targets are `mean/p95 TV = 0/0` in iteration 1 and become `0.473946 / 1.0` immediately after the first fitted Advantage behavior is fed back into collection.

The known causal chain remains:

```text
Advantage approximation -> nonlinear regret map -> behavior
-> sampled next-iteration trajectories -> divergent strategy targets
```

The final AveragePolicy can add residual approximation error, but it is not the primary origin of the cross-seed tail.

## Five-iteration durability is the current optimization target

Policy-mixture size 4 at 2×128 remains a strong short-horizon result:

```text
mean TV = 0.171940
p95 TV  = 0.413605
fit gates = PASS
```

A fresh physical rerun (`31440366942`) reproduced the same cross-seed metrics exactly, strengthening fresh-run determinism evidence.

Mandatory 5×64 durability baseline:

```text
mean TV = 0.266591
p50 TV  = 0.246805
p95 TV  = 0.567002
max TV  = 0.905547
fit gates = PASS
```

Therefore the prepared size-4 640 workflow remains dormant. A mechanism must first materially improve **both** `0.266591` mean and `0.567002` p95 at 5×64 before any acceptance-scale escalation.

## New completed evidence

### Final AveragePolicy ensemble on top of size-4 policy-mixture CFR

Workflow `31440493410`, evidence `eb115121b9a4002615f3310767c89aebbfce05c9`:

| final policy members | mean TV | p95 TV |
|---:|---:|---:|
| 1 | `0.179750` | `0.434644` |
| 2 | `0.159165` | `0.404792` |
| 4 | **`0.138377`** | **`0.368730`** |

All fit gates pass. Size 4 crosses the frozen **mean** gate and misses the p95 gate by only `0.01873`. This proves that final-policy approximation is a material residual layer once upstream behavior has been stabilized, but it is still not enough by itself and this is only a two-iteration result.

**Decision:** do not spend a full 5×64 run on this downstream layer alone. Stack it only after an upstream mechanism first demonstrates five-iteration durability.

### Support-conditioned residual tail

Workflow `31440576227`, evidence `3e05501faf64da91d5ca257b70423271603b6013`:

```text
union A+B:                 mean 0.167806 / p95 0.411491
exact shared observations: mean 0.181562 / p95 0.398769
A-only exact support:       mean 0.187291 / p95 0.417470
B-only exact support:       mean 0.157756 / p95 0.418203
```

The tail is **not** merely off-support extrapolation. Even byte-identical shared SPNNIV1 observations retain material disagreement. Therefore a support-only fix cannot close R7.3.

### Behavior target order

Workflow `31441852607`, evidence `57df6d80b457b77cb3e6c1f1906b5678bb43df12`, uses the same exact zero-regret uniform Advantage memory for all fits.

```text
raw Advantage samples             = 10,565
unique observation/legal groups   = 10,515
compression ratio                 = 0.995267
```

Same-memory independent surrogate fits:

| target construction | mean TV | p95 TV |
|---|---:|---:|
| mean of sample-level RM policies | `0.104576` | `0.357100` |
| **RM(weighted mean regret)** | **`0.102435`** | **`0.347652`** |

The regret-first ordering marginally crosses the frozen p95 threshold on this same-memory diagnostic. Because exact duplicate observations are rare, the gain is not being attributed to compression itself. The important hypothesis is the ordering:

```text
E_LCFR[regret | infoset] -> regret matching
```

rather than applying the nonlinear regret map to every noisy sample first.

I therefore launched workflow `31444324235`, with both **2×128** and **5×64** physical E2E candidates using this aggregated-regret Direct Behavior target. Both jobs passed build/regression and smoke and are physically running. This remains an experimental/non-equivalent behavior surrogate and is excluded from automatic production promotion until semantic review.

### Size-1 damping factorial completed

Workflow `31441110526`:

```text
size1, no damping:       mean 0.438845 / p95 0.878729
size1, epsilon0=.30:     mean 0.395333 / p95 0.798287
```

The decaying tremble improves mean by about 9.9% and p95 by about 9.2% even without ensembling. Therefore damping has an independent causal benefit, but size-4 ensembling remains the much larger effect. The missing interaction term is the active size-4 epsilon matrix.

### Robust member aggregation remains closed

Ordinary probability averaging beat median and trimmed mean; robust aggregation worsened p95. Rare single-member outliers are not the dominant tail source.

## Active five-iteration durability matrix

All of the following are being tested at the failing 5×64 horizon unless explicitly marked short-horizon:

- size-4 decaying uniform tremble: `epsilon0 = 0.15 / 0.30 / 0.45`, decay `0.50` — workflow `31441018067`, physical running;
- previous-policy temporal blend: current weight `0.50 / 0.75` — `31441224117`, physical running;
- first-transition-only tremble: schedule `[0.30, 0, 0, 0]` — `31441567261`, physical running;
- Direct Behavior durability control — `31441650915`, physical running, causal control only;
- uncertainty-adaptive damping: epsilon proportional to policy-mixture member disagreement, scale `0.50 / 1.00`, cap `0.50` — `31442367579`, physical running;
- aggregated-regret Direct Behavior: 2×128 + 5×64 — `31444324235`, physical running;
- size-4 320 fresh-run reproducibility copy — `31440366909`, physical running;
- policy-mixture size 8 — `31440425854`, paired 2×128 physical running; any short-horizon win still needs its own five-iteration gate.

## Why uncertainty-adaptive damping is attractive

Global tremble flattens both uncertain and already-stable states. The adaptive variant instead computes disagreement among the independently fitted regret-policy members and mixes toward legal-uniform only where they disagree:

```text
uncertainty = mean member TV from ensemble mean policy
epsilon = min(cap, scale * uncertainty)
pi_used = (1-epsilon)*pi_policy_mixture + epsilon*uniform
```

If it beats global tremble at comparable fit quality, it is the more strategically conservative intervention because reliable states are left nearly unchanged.

## Automatic consolidation now covers the full matrix

`tools/summarize_r7_3_durability_matrix.py` is now V2 and waits for 12 candidate evidence rows plus the authoritative size-4 baseline. It includes global tremble, temporal blend, first-transition damping, uncertainty-adaptive damping, size-1 factorial controls, ordinary Direct Behavior and aggregated-regret Direct Behavior.

Direct Behavior variants are reported but explicitly excluded from automatic conservative promotion because theoretical equivalence is not established.

The summary workflow is triggered by every expected evidence path and will commit `validation/R7_3_DURABILITY_MATRIX_SUMMARY.json` only after the complete matrix exists.

## Promotion rule

Before any new mechanism advances to 640 it must:

1. PASS every frozen per-seed fit gate;
2. materially improve both mean and p95 versus `0.266591 / 0.567002` at 5×64;
3. survive fresh-run reproducibility checks;
4. be the smallest/interpretable mechanism among statistically comparable candidates;
5. have any changed behavior semantics explicitly versioned;
6. pass deterministic continuous-vs-stop/restore/continue checkpoint recertification;
7. keep the frozen acceptance gates unchanged.

If the best upstream durable candidate approaches the gates but does not clear them, the already-proven final AveragePolicy ensemble becomes the next residual layer to stack. We will not run that combination until the upstream winner is known.

## Closed / deprioritized primary branches

- raw root scaling beyond 1280;
- independent x8/x16 path multiplication;
- common-path RNG;
- antithetic x4;
- exhaustive opponent expectation;
- merely raising Advantage optimizer ceiling;
- behavior-aware MSE auxiliary objective;
- exact duplicate-target aggregation as a standalone solution;
- behavior-aware multistart selection;
- raw Advantage ensemble 2/4 standalone;
- final AveragePolicy ensemble standalone;
- legal common-mode centering;
- robust median/trimmed policy aggregation;
- card/suit rewrite as dominant explanation;
- direct size4 policy-mixture 640 escalation.

`READY FOR TABLES = NO`.
