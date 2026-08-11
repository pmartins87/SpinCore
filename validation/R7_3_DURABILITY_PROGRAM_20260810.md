# R7.3 five-iteration durability program — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates remain unchanged.

## Optimization target

The authoritative paired size4 policy mixture is strong over two CFR iterations but degrades at the mandatory five-iteration horizon:

```text
2×128: mean 0.171940 / p95 0.413605
5×64:  mean 0.266591 / p95 0.567002
```

Every upstream candidate must therefore prove **feedback-depth durability** before acceptance scaling.

The confirmed causal chain is:

```text
Advantage approximation -> nonlinear regret map -> behavior
-> next trajectories -> next strategy targets
```

Shared strategy targets are identical in iteration 1 under the initial uniform behavior. The first fitted Advantage feedback produces large shared-target divergence in iteration 2, and subsequent fitted-policy replacement keeps regenerating the instability.

## Frozen gates

```text
Advantage weighted NRMSE <= 0.75
AveragePolicy weighted mean TV <= 0.12
cross-seed mean TV <= 0.15
cross-seed p95 TV <= 0.35
```

No gate is relaxed.

## Completed 5×64 results

All rows are 5 CFR iterations × 64 roots = 320 roots/seed.

| candidate | mean TV | p95 TV | fit | interpretation |
|---|---:|---:|---|---|
| size4 no damping | `0.266591` | `0.567002` | PASS | authoritative baseline |
| size4 decay tremble e15 | `0.231886` | `0.475154` | PASS | improvement |
| size4 decay tremble e30 | `0.217853` | `0.457102` | PASS | stronger |
| size4 decay tremble e45 | `0.211607` | `0.448567` | PASS | best global tremble |
| size4 temporal w75 | `0.222885` | `0.481673` | PASS | improvement |
| size4 temporal w50 | `0.179915` | `0.395478` | PASS | strong repeated stabilization |
| size4 first-transition e30 | `0.239409` | `0.512631` | PASS | one-shot intervention insufficient |
| size4 uncertainty s05 | `0.224640` | `0.471332` | PASS | adaptive damping helps |
| **size4 uncertainty s10** | **`0.168098`** | **`0.356780`** | **PASS** | **best completed durable row** |
| size4 regret-floor e05 | `0.201896` | `0.427662` | PASS | below baseline, worse than uncertainty |
| size4 regret-floor e10 | `0.183337` | `0.384941` | PASS | below baseline, worse than uncertainty |
| size1 no damping | `0.438845` | `0.878729` | — | poor |
| size1 decay tremble e30 | `0.395333` | `0.798287` | — | damping helps independently |
| Direct Behavior control | `0.276185` | `0.828670` | PASS | closed |
| Direct Behavior aggregated regret | `0.307350` | `0.914166` | PASS | closed |

The current best completed candidate, uncertainty-s10, improves the baseline by approximately 36.9% in mean TV and 37.1% in p95 TV. Its remaining miss is very small in p95 but still hard:

```text
mean gap = 0.018098
p95 gap  = 0.006780
```

Regret-floor e10 does not dominate uncertainty-s10 on either metric, so the regret-floor branch is deprioritized as a primary route.

## Uncertainty-adaptive behavior

The mechanism is:

```text
disagreement = mean member TV to ensemble-mean regret policy
epsilon = min(cap, scale * disagreement)
behavior = (1-epsilon) * ensemble_mean + epsilon * legal_uniform
```

Completed evidence:

```text
scale .50 cap .50: mean 0.224640 / p95 0.471332
scale 1.00 cap .50: mean 0.168098 / p95 0.356780
```

This supports ensemble disagreement as a useful state-local proxy for where feedback should be damped.

Commit `b0887eabb201fb28214b47fb3959733896992ada` instrumented the same behavior without changing its returned policy or consuming RNG. New evidence records mean/max epsilon, mean disagreement, maximum raw epsilon before cap, cap-hit fraction, and fractions of fitted-behavior calls with epsilon at least `.10` and `.25`. This lets the cap sweep distinguish genuine scale benefit from clipping at `.50`.

## Active local calibration

Workflow `31450032347` is physically running after build/regression/smoke PASS:

```text
size4 s1.25 cap .50
size4 s1.50 cap .50
```

Workflow `31451592073` has also passed build/regression/smoke for all four jobs and is physically running:

```text
size4 s1.75 cap .50
size4 s2.00 cap .50
size4 s1.50 cap .65
size4 s1.50 cap .80
```

The selection rule favors the **smallest** fit-valid five-iteration candidate that clears both cross-seed gates. The cap variants are not assumed superior; they exist specifically to test whether `.50` clips the unstable tail.

## Size8 program

Short-horizon size8 2×128 remains the first current Generation-2 candidate to clear both cross gates:

```text
mean 0.139615 PASS
p95  0.329689 PASS
fits PASS
```

Mandatory five-iteration runs remain active:

```text
size8 no damping          workflow 31446308103
size8 + temporal w50      workflow 31448623827
size8 + uncertainty s1.0  workflow 31449546648
```

All three have passed build/regression/smoke and are executing the physical 5×64 stage. No interaction between mechanisms is assumed until measured.

## Independent baseline reproducibility

The existing comparator `tools/check_r7_3_fresh_run_reproducibility.py` requires two evidence commits for both its 256 and 320 targets. The 256 target already has two independent physical commits; the authoritative 320 durability baseline previously had one.

Commit `a67dcf651862d04b9590072ca92e6bbe1a4b65cf` enabled explicit retriggering of the 320 baseline workflow. This launched workflow `31451518046`, which passed build/regression/smoke and is physically executing the second 5×64 baseline. Its evidence will allow exact structure/sample/node and cross-seed comparison at tolerance `1e-9`.

This is a baseline determinism check, not a substitute for winner-specific checkpoint/resume certification.

## Candidate checkpoint state

`SPINCORE_R7_CANDIDATE_BEHAVIOR_V1` is layered through the existing `SPINCORE_R7_CHECKPOINT_V2` extra field. It preserves side Advantage members, previous temporal ensemble, parameters and fit generation while reusing the authoritative restored primary model.

The restore path was hardened so constructing side models cannot advance global torch RNG. Main regression `31450903801` passed:

```text
C++ PASS
Python 33 passed
candidate side restore torch-RNG neutral PASS
```

Physical continuous-vs-stop/restore/continue recertification remains blocked until a winner exists.

## Winner certification prepared in advance

A five-iteration numerical pass no longer leaves an undefined next step.

`tools/freeze_r7_3_candidate_semantics.py` is fail-closed. It accepts only `SPINCORE_R7_3_WINNER_SELECTION_V1` and refuses to freeze evidence unless all fit gates pass, `cross_seed_pass=true`, `r7_3_pass=true`, the 5×64 horizon and exact opponent level are correct, the generation-2 deck/RNG contract is preserved, and no gate was changed. It freezes exact source commit, source workflow, execution hyperparameters and Git object identities. Output:

```text
SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1
```

Workflow: `.github/workflows/r7_3_freeze_candidate_semantics.yml`.

Then `tools/run_r7_3_frozen_candidate_fresh_repro.py` creates a detached worktree at that exact source commit, rebuilds it, runs C++ and Python regressions, reruns the frozen candidate and compares the complete deterministic evidence recursively. Only `generated_at_unix` and `duration_seconds` are excluded; numeric tolerance is `1e-9`.

Workflow: `.github/workflows/r7_3_frozen_candidate_fresh_repro.yml`.

Report schema:

```text
SPINCORE_R7_3_FROZEN_CANDIDATE_FRESH_REPRO_V1
```

This stage still leaves `ready_for_640=false`. Candidate-specific checkpoint/resume recertification is required next.

## Evidence consolidation

The original base matrix remains:

```text
SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4
15 candidates + baseline
```

The expanded matrix is now:

```text
SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V4
23 candidates + baseline = 24 rows
```

Supplemental calibration/composition rows are size4 uncertainty s1.25, s1.50, s1.75, s2.00, s1.50-cap.65, s1.50-cap.80, size8 temporal-w50 and size8 uncertainty-s1.0. Ranking is evidence only and cannot change production semantics.

## Causal conclusions retained

- Advantage approximation/sign variance is material.
- Policy-mixture ensembling is material.
- Global damping has independent benefit.
- Continued temporal stabilization is stronger than one-shot first-transition damping.
- State-local ensemble-disagreement damping is the strongest completed durable mechanism.
- Exact shared-state disagreement remains material; off-support extrapolation is not the sole source.
- Regret-floor helps but is dominated by uncertainty-s10 at the tested values.
- Ordinary and aggregated-regret Direct Behavior are closed as durable solutions.
- Final AveragePolicy ensembling remains only a downstream residual layer.

## Promotion rule

No candidate moves to 640 merely because it ranks first. Promotion requires:

1. all frozen per-seed fit gates PASS;
2. material improvement versus `0.266591 / 0.567002` at 5×64;
3. full cross-seed gate clearance `mean <= 0.15` and `p95 <= 0.35` at five iterations;
4. fail-closed semantic/execution freeze against the exact source commit that generated the pass;
5. exact-source fresh-process reproducibility;
6. deterministic continuous-vs-stop/restore/continue candidate checkpoint recertification;
7. the smallest/interpretable winner among statistically comparable mechanisms;
8. frozen gates unchanged.

Only then may acceptance-scale 640 run. R7.4 and R8–R12 still remain before table use.
