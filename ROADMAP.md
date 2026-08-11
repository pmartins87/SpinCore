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
  - R7.4 larger HU + 3H pilot — TODO after R7.3 acceptance
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Frozen R7.3 acceptance contract

```text
Advantage weighted NRMSE <= 0.75
AveragePolicy weighted mean TV <= 0.12
cross-seed mean TV <= 0.15
cross-seed p95 TV <= 0.35
algorithm seeds = 20260829, 20260807
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
global_root continuous across iterations
partial-exact opponent level = 2
primary RNG = one persistent live bundle.batch_rng
training device = cpu
learning rate = 0.001
```

No gate is relaxed. Side ensemble fits do not advance the authoritative primary RNG stream.

## Confirmed R7.3 failure mechanism

```text
Advantage approximation -> nonlinear regret map -> behavior
-> next trajectories -> next strategy targets
```

Iteration-1 shared strategy targets are identical under initial uniform behavior. Large shared-target divergence appears only after the first fitted Advantage behavior feeds back into iteration 2. Later fitted-policy replacement keeps regenerating instability. Support fragmentation and exact shared-state disagreement are both material; off-support extrapolation is not the sole cause.

Three mechanisms have independent empirical support:

1. **Advantage policy ensembling** reduces approximation/sign variance. Size8 is the first current-generation 2×128 candidate to clear both cross-seed gates.
2. **Temporal inertia** reduces repeated feedback amplification.
3. **Uncertainty-adaptive damping** is the strongest completed five-iteration mechanism; it damps only states where independently fitted regret policies disagree.

## Mandatory durability baseline

Authoritative paired size4 policy mixture:

```text
2×128: mean 0.171940 / p95 0.413605, fits PASS
5×64:  mean 0.266591 / p95 0.567002, fits PASS
```

A short-horizon win never authorizes 640. Five-iteration durability is mandatory.

## Completed 5×64 results

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

Uncertainty-s10 improves the baseline by about 36.9% in mean and 37.1% in p95 but still misses the hard gates by:

```text
mean gap = 0.018098
p95 gap  = 0.006780
```

Regret-floor e10 improves over baseline but is worse than uncertainty-s10 in both metrics, so regret-floor is no longer a primary route.

## Active physical durability program

All current candidates below have already passed build/regression/smoke and are in the physical 5×64 stage.

```text
size4 uncertainty s1.25 cap .50      workflow 31450032347
size4 uncertainty s1.50 cap .50      workflow 31450032347
size4 uncertainty s1.75 cap .50      workflow 31451592073
size4 uncertainty s2.00 cap .50      workflow 31451592073
size4 uncertainty s1.50 cap .65      workflow 31451592073
size4 uncertainty s1.50 cap .80      workflow 31451592073
size8 no damping                      workflow 31446308103
size8 + temporal w50                  workflow 31448623827
size8 + uncertainty s1.0 cap .50     workflow 31449546648
```

The scale/cap sweep distinguishes stronger state-local damping from simple clipping at the `.50` cap. The uncertainty runner now records RNG-neutral diagnostics: mean/max epsilon, mean disagreement, max raw epsilon, cap-hit fraction and fractions with epsilon >= `.10` and `.25`.

Selection policy: prefer the **smallest/interpretable** fit-valid five-iteration candidate that clears both cross gates, unless a more complex candidate has materially stronger margin that changes the robustness decision.

## Size8 short-horizon milestone

```text
2×128 size8
mean 0.139615 PASS
p95  0.329689 PASS
fits PASS
```

This remains only a short-horizon milestone until its five-iteration variants finish.

## Independent baseline reproducibility

The existing comparator needs two independent evidence commits at both 256 and 320. The 256 target already has two. Workflow `31451518046` is now physically executing the second authoritative size4 5×64 baseline after build/regression/smoke PASS. When persisted, the existing comparator can require exact structure/sample/node agreement and cross-seed metric delta <= `1e-9`.

This baseline determinism check is separate from winner-specific certification.

## Candidate checkpoint state

R7.2 base checkpoint remains `SPINCORE_R7_CHECKPOINT_V2`. R7.3 ensemble state is serialized through its unchanged `extra` field using:

```text
SPINCORE_R7_CANDIDATE_BEHAVIOR_V1
```

It preserves side Advantage models, prior temporal ensemble, parameters and fit generation; restores the authoritative primary as member zero; and fails closed on primary mismatch.

A hidden determinism bug was fixed: naive construction of restored side networks consumed global torch RNG before loading their state. Side-model construction is now isolated with `torch.random.fork_rng(devices=[])`.

The physical checkpoint worker is no longer merely theoretical. Main regression workflow `31452600161`, commit `2423e1c2f22dee0557d0c7f280ac1f5e542f9e20`:

```text
C++ regression PASS
Python 40 passed
uncertainty tiny continuous-vs-restore exact PASS
temporal tiny continuous-vs-restore exact PASS
```

Those tests exercise actual collection/training/checkpoint/restore/continue paths, including preservation of the previous temporal ensemble.

## Winner certification pipeline — fully prepared through 640

A numerical 5×64 PASS now has a complete fail-closed path before any larger acceptance run.

### 1. Deliberate winner selection

A selected candidate is represented by `SPINCORE_R7_3_WINNER_SELECTION_V1`. There is intentionally no winner-selection file yet.

### 2. Immutable semantic/execution freeze

`tools/freeze_r7_3_candidate_semantics.py` and `.github/workflows/r7_3_freeze_candidate_semantics.yml` output:

```text
SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1
```

The freezer rejects anything that is not an actual fit-valid 5×64 gate pass. It pins:

- exact algorithm seeds `20260829, 20260807`;
- exact evidence commit and byte-identical evidence SHA-256;
- source workflow run/path and exact source head;
- authoritative deck/RNG contract;
- ensemble size and behavior parameters;
- full 5×64 execution hyperparameters;
- Git object identities for source trees, workflow and runner dependencies.

The source head must be an ancestor of the evidence commit. Workflow overrides of frozen seeds/lr/device are rejected.

### 3. Exact-source fresh-process reproducibility

`tools/run_r7_3_frozen_candidate_fresh_repro.py` and `.github/workflows/r7_3_frozen_candidate_fresh_repro.yml` output:

```text
SPINCORE_R7_3_FROZEN_CANDIDATE_FRESH_REPRO_V1
```

They create a detached worktree at the exact source commit, rebuild it, run C++ and Python regressions, rerun the exact frozen candidate and recursively compare the immutable original evidence with the fresh evidence. Only `generated_at_unix` and `duration_seconds` are ignored; numeric tolerance is `1e-9`.

### 4. Physical continuous-vs-stop/restore/continue recertification

`tools/r7_3_frozen_candidate_checkpoint_worker.py`, `tools/run_r7_3_frozen_candidate_checkpoint_recert.py` and `.github/workflows/r7_3_frozen_candidate_checkpoint_recert.yml` output:

```text
SPINCORE_R7_3_CANDIDATE_CHECKPOINT_RECERT_V1
```

Fresh reproducibility must pass first. The orchestrator uses the exact frozen algorithm source and overlays only the checkpoint serialization helper. The physical run shares a common prefix, checkpoints after iteration 3, then compares continuous execution against restore+continue through iteration 5 and final AveragePolicy fit.

The final equality gate includes counters, both reservoirs and reservoir RNGs, `bundle.batch_rng`, global torch RNG, primary Advantage, AveragePolicy, both optimizers, every current side model, prior temporal models, fit generation, shared cross-seed observation corpus and final cross-seed metrics. Any mismatch blocks 640.

### 5. Exact-source certified-winner 640 acceptance

`tools/run_r7_3_frozen_candidate_640_acceptance.py` and `.github/workflows/r7_3_frozen_candidate_640_acceptance.yml` output:

```text
SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1
```

This stage cannot run unless freeze, exact-source fresh reproducibility and checkpoint recertification all pass for the same source/evidence provenance. It reuses the exact frozen source and semantics, changing only roots per iteration from `64` to `128`:

```text
5 × 128 = 640 roots/seed
```

The same frozen cross-seed and fit gates apply. A 640 pass only marks R7.3 ready to advance to R7.4. It never sets table readiness.

## Evidence consolidation

```text
base:     SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4
          15 candidates + baseline

extended: SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V4
          23 candidates + baseline = 24 rows
```

The expanded matrix includes the six size4 uncertainty scale/cap calibrations plus size8 temporal-w50 and size8 uncertainty-s1.0. Ranking is evidence only; it never changes semantics automatically.

## Residual downstream layer

Final AveragePolicy size4 ensemble remains reserved as a residual layer:

```text
mean 0.138377
p95  0.368730
```

It is considered only if the best durable upstream mechanism still leaves a small residual. No additive or multiplicative benefit is assumed without a physical combined test.

## Closed / deprioritized primary branches

Raw root scaling, x8/x16 path multiplication as standalone fix, common-path RNG, antithetic x4, exhaustive opponent expectation, simply raising Advantage optimizer capacity, behavior-aware MSE auxiliary objective, duplicate-target aggregation, multistart selection, raw Advantage ensemble 2/4 standalone, legal common-mode centering, robust median/trimmed aggregation, card/suit rewrite as dominant explanation, ordinary Direct Behavior, aggregated-regret Direct Behavior, regret-floor as primary mechanism and direct plain-size4 640 escalation are closed or deprioritized.

## R7.3 promotion rule

Before R7.3 can advance to R7.4, the selected mechanism must:

1. PASS all frozen per-seed fit gates at 5×64;
2. clear both cross-seed gates `mean <= 0.15`, `p95 <= 0.35` at 5×64;
3. be frozen against its immutable evidence and exact source commit;
4. reproduce from that exact source in a fresh process;
5. pass exact continuous-vs-stop/restore/continue recertification;
6. pass the same gates at exact-source 5×128 = 640 acceptance scale;
7. keep all frozen gates unchanged.

Then and only then does **R7.3 become PASS and R7.4 begin**. R7.4 HU+3H, followed by R8–R12, still stand between us and table use.

`READY FOR TABLES = NO`.
