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

`global_root` is continuous across CFR iterations. The recovered primary path preserves one persistent live `bundle.batch_rng` in execution order. Side ensemble fits do not advance this authoritative RNG stream.

## R7.3 causal state

The confirmed instability chain remains:

```text
Advantage approximation -> nonlinear regret map -> behavior
-> next trajectories -> next strategy targets
```

Iteration-1 shared strategy targets are identical under the initial uniform behavior. Large shared-target divergence appears after the first fitted Advantage behavior feeds back into iteration 2. Later fitted-policy replacement keeps regenerating instability. Support fragmentation and exact shared-state disagreement are both material, so off-support extrapolation is not the sole cause.

The strongest supported upstream mechanisms are now:

1. **Advantage policy ensembling** — size8 is the first current-generation 2×128 candidate to clear both frozen cross-seed gates.
2. **Temporal inertia** — repeated temporal blending materially reduces five-iteration amplification.
3. **Uncertainty-adaptive damping** — currently the strongest completed five-iteration mechanism; it damps only states where independently fitted regret policies disagree.

## Mandatory durability baseline

Authoritative paired size4 policy mixture:

```text
2×128: mean 0.171940 / p95 0.413605, fits PASS
5×64:  mean 0.266591 / p95 0.567002, fits PASS
```

The five-iteration result is the promotion baseline. A short-horizon pass never justifies direct 640 escalation.

## Completed five-iteration results

All values are 5 CFR iterations × 64 roots = 320 roots/seed.

| candidate | mean TV | p95 TV | fit | decision |
|---|---:|---:|---|---|
| size4 no damping | `0.266591` | `0.567002` | PASS | baseline |
| size4 decay tremble e15 | `0.231886` | `0.475154` | PASS | FAIL |
| size4 decay tremble e30 | `0.217853` | `0.457102` | PASS | FAIL |
| size4 decay tremble e45 | `0.211607` | `0.448567` | PASS | FAIL |
| size4 temporal w75 | `0.222885` | `0.481673` | PASS | FAIL |
| size4 temporal w50 | `0.179915` | `0.395478` | PASS | strong / FAIL |
| size4 first-transition-only e30 | `0.239409` | `0.512631` | PASS | FAIL |
| size4 uncertainty s05 | `0.224640` | `0.471332` | PASS | FAIL |
| **size4 uncertainty s10** | **`0.168098`** | **`0.356780`** | **PASS** | **best completed / FAIL** |
| size4 regret-floor e05 | `0.201896` | `0.427662` | PASS | deprioritized |
| size4 regret-floor e10 | `0.183337` | `0.384941` | PASS | deprioritized |
| size1 no damping | `0.438845` | `0.878729` | — | FAIL |
| size1 decay tremble e30 | `0.395333` | `0.798287` | — | FAIL |
| Direct Behavior control | `0.276185` | `0.828670` | PASS | closed |
| Direct Behavior aggregated regret | `0.307350` | `0.914166` | PASS | closed |

Uncertainty-s10 improves the baseline by about **36.9% in mean TV** and **37.1% in p95 TV**, but still misses the frozen gates by:

```text
mean gap = 0.018098
p95 gap  = 0.006780
```

Regret-floor e10 is a real improvement over baseline, but it is worse than uncertainty-s10 in both mean and p95, so regret-floor is no longer a primary path.

## Size8 short-horizon milestone

Workflow `31440425854`:

```text
2×128
mean TV = 0.139615  PASS <= 0.15
p95 TV  = 0.329689  PASS <= 0.35
fits    = PASS
```

This remains a short-horizon milestone only. Size8 no-damping 5×64 is still running in workflow `31446308103`.

## Current physical durability program

The active long-horizon comparison now runs three complementary groups under the same frozen 5×64 contract.

### Local size4 uncertainty calibration

Workflow `31450032347`:

```text
s1.25 cap .50
s1.50 cap .50
```

Both jobs passed build/regression and smoke and are executing the physical five-iteration candidate. If a size4 candidate clears both gates, it is preferred over a statistically comparable size8 candidate because it is simpler and cheaper.

### Expanded size4 scale/cap calibration

Workflow `31451592073`:

```text
s1.75 cap .50
s2.00 cap .50
s1.50 cap .65
s1.50 cap .80
```

All four jobs passed build/regression and smoke and are executing the physical five-iteration candidate. This separates two questions: whether more state-local damping helps and whether the `.50` cap is clipping the unstable tail.

`tools/run_r7_3_policy_mixture_uncertainty_damping.py` now records RNG-neutral runtime diagnostics without changing behavior:

```text
mean/max epsilon
mean disagreement
max raw epsilon before cap
cap-hit fraction
fraction epsilon >= .10
fraction epsilon >= .25
```

### Size8 candidates

Still executing physically after build/regression/smoke PASS:

```text
size8 no damping             workflow 31446308103
size8 + temporal w50         workflow 31448623827
size8 + uncertainty s1.0     workflow 31449546648
```

No additivity is assumed; each composition must earn its own 5×64 result.

## Independent baseline reproducibility

The existing fresh-run comparator requires two physical evidence commits for both the paired 256 result and the five-iteration 320 baseline. The 256 target already has two independent commits; the 320 target previously had only one.

Workflow `31451518046` is now running a second independent authoritative size4 5×64 baseline after build/regression/smoke PASS. When its evidence is persisted, `tools/check_r7_3_fresh_run_reproducibility.py` can compare exact structure/sample/node counters and cross-seed metrics at tolerance `1e-9`.

This baseline reproducibility check is separate from winner-specific checkpoint/resume certification.

## Candidate checkpoint/resume readiness

R7.2 certifies base checkpoint `SPINCORE_R7_CHECKPOINT_V2`. R7.3 ensembles contain extra behavior state, so `python/spincore/r7_candidate_checkpoint.py` adds:

```text
SPINCORE_R7_CANDIDATE_BEHAVIOR_V1
```

through the unchanged base checkpoint `extra` field. It preserves side Advantage members, previous temporal ensemble, wrapper parameters and fit generation, reuses the authoritative restored primary as member zero and fails closed on primary mismatch.

A hidden determinism defect was found and fixed: constructing side PyTorch modules during restore would initialize temporary weights and consume global torch RNG before `load_state_dict()`. Side reconstruction is now isolated with `torch.random.fork_rng(devices=[])`.

Main regression `31450903801`:

```text
C++ regression PASS
Python 33 passed
side-model restore torch-RNG neutral PASS
```

This is serialization/readiness evidence. The exact winning candidate still requires physical continuous-vs-stop/restore/continue recertification.

## Winner certification pipeline — prepared before the winner exists

To avoid a new implementation gap after a five-iteration PASS, the winner path is now fail-closed and staged.

`tools/freeze_r7_3_candidate_semantics.py` consumes only a deliberate `SPINCORE_R7_3_WINNER_SELECTION_V1` record and refuses to freeze a candidate unless its 5×64 evidence actually has all fit gates PASS, `cross_seed_pass=true`, `r7_3_pass=true`, the authoritative deck/RNG contract and unchanged frozen gates. It records exact source commit plus Git object identities for the relevant source trees, workflow and runners. Output schema:

```text
SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1
```

Workflow: `.github/workflows/r7_3_freeze_candidate_semantics.yml`.

After a freeze, `.github/workflows/r7_3_frozen_candidate_fresh_repro.yml` runs `tools/run_r7_3_frozen_candidate_fresh_repro.py`. It creates a detached worktree at the exact source commit that generated the winner, rebuilds that source, runs C++ and Python regressions, reruns the exact candidate with the frozen hyperparameters and recursively compares the complete evidence, ignoring only wall-clock timestamp/duration and allowing numeric delta at most `1e-9`.

Output schema:

```text
SPINCORE_R7_3_FROZEN_CANDIDATE_FRESH_REPRO_V1
```

Neither semantic freeze nor fresh-process reproducibility sets `ready_for_640=true`; physical checkpoint/resume recertification remains mandatory first.

## Automatic evidence consolidation

Base matrix remains:

```text
SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4
15 candidate rows + 1 baseline
```

Expanded matrix is now:

```text
SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V4
23 candidate rows + 1 baseline = 24 rows
```

The supplemental rows include six size4 uncertainty scale/cap calibrations plus size8 temporal-w50 and size8 uncertainty-s10. Ranking is evidence only and never promotes production semantics automatically.

## Residual downstream layer

Final AveragePolicy size4 ensemble remains reserved as a downstream residual layer:

```text
mean TV = 0.138377
p95 TV  = 0.368730
```

It is considered only after an upstream durable winner is known; no multiplicative/additive benefit is assumed.

## Closed / deprioritized primary branches

- raw root scaling beyond 1280;
- independent x8/x16 path multiplication as standalone fix;
- common-path RNG;
- antithetic x4;
- exhaustive opponent expectation;
- merely raising Advantage optimizer ceiling;
- behavior-aware MSE auxiliary objective;
- exact duplicate aggregation as standalone fix;
- behavior-aware multistart selection;
- raw Advantage ensemble 2/4 standalone;
- legal common-mode centering;
- robust median/trimmed policy aggregation;
- card/suit rewrite as dominant explanation;
- ordinary Direct Behavior as durable solution;
- aggregated-regret Direct Behavior as durable solution;
- regret-floor as primary durable mechanism;
- direct size4 policy-mixture 640 escalation.

## Promotion rule

Before any mechanism advances to 640 it must:

1. PASS every frozen per-seed fit gate;
2. materially improve both mean and p95 versus `0.266591 / 0.567002` at 5×64;
3. clear both frozen cross-seed gates `0.15 / 0.35` at the same five-iteration horizon;
4. have the exact winning behavior/execution semantics frozen against its source commit;
5. survive exact-source fresh-process reproducibility;
6. pass deterministic continuous-vs-stop/restore/continue candidate checkpoint recertification;
7. remain the smallest/interpretable mechanism among statistically comparable winners;
8. keep all frozen gates unchanged.

Only then may a 640 acceptance-scale run be authorized. A 640 pass still does not imply table readiness: R7.4 HU+3H, then R8–R12, remain ahead.

`READY FOR TABLES = NO`.
