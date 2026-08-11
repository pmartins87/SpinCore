# R7.3 five-iteration durability program — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates remain unchanged.

## Optimization target and failure mechanism

The authoritative paired size4 policy mixture is strong over two iterations but degrades under repeated feedback:

```text
2×128: mean 0.171940 / p95 0.413605
5×64:  mean 0.266591 / p95 0.567002
```

The confirmed causal chain is:

```text
Advantage approximation -> nonlinear regret map -> behavior
-> next trajectories -> next strategy targets
```

Shared targets are identical in iteration 1 under initial uniform behavior. The first fitted Advantage feedback creates large iteration-2 shared-target divergence, and later fitted-policy replacement keeps regenerating it. Every primary candidate must therefore prove five-iteration durability before acceptance scaling.

## Frozen contract

```text
Advantage weighted NRMSE <= 0.75
AveragePolicy weighted mean TV <= 0.12
cross-seed mean TV <= 0.15
cross-seed p95 TV <= 0.35
algorithm seeds = 20260829, 20260807
partial-exact opponent level = 2
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
global_root continuous across iterations
primary RNG = one persistent live bundle.batch_rng
lr = 0.001
device = cpu
```

No gate is relaxed.

## Completed 5×64 evidence

| candidate | mean TV | p95 TV | fit | interpretation |
|---|---:|---:|---|---|
| size4 no damping | `0.266591` | `0.567002` | PASS | authoritative baseline |
| size4 tremble e15 | `0.231886` | `0.475154` | PASS | improvement |
| size4 tremble e30 | `0.217853` | `0.457102` | PASS | stronger |
| size4 tremble e45 | `0.211607` | `0.448567` | PASS | best global tremble |
| size4 temporal w75 | `0.222885` | `0.481673` | PASS | improvement |
| size4 temporal w50 | `0.179915` | `0.395478` | PASS | strong repeated stabilization |
| size4 first-transition e30 | `0.239409` | `0.512631` | PASS | one-shot intervention insufficient |
| size4 uncertainty s05 | `0.224640` | `0.471332` | PASS | adaptive damping helps |
| **size4 uncertainty s10** | **`0.168098`** | **`0.356780`** | **PASS** | **best completed durable row** |
| size4 regret-floor e05 | `0.201896` | `0.427662` | PASS | dominated by uncertainty s10 |
| size4 regret-floor e10 | `0.183337` | `0.384941` | PASS | dominated by uncertainty s10 |
| size1 no damping | `0.438845` | `0.878729` | — | poor |
| size1 tremble e30 | `0.395333` | `0.798287` | — | damping helps independently |
| Direct Behavior | `0.276185` | `0.828670` | PASS | closed |
| Direct Behavior aggregated regret | `0.307350` | `0.914166` | PASS | closed |

Uncertainty-s10 improves the baseline by about 36.9% in mean and 37.1% in p95. It still fails hard gates by:

```text
mean gap = 0.018098
p95 gap  = 0.006780
```

## Uncertainty-adaptive mechanism and active calibration

```text
disagreement = mean member TV to ensemble-mean regret policy
epsilon = min(cap, scale * disagreement)
behavior = (1-epsilon) * ensemble_mean + epsilon * legal_uniform
```

Completed:

```text
scale .50 cap .50 -> 0.224640 / 0.471332
scale 1.00 cap .50 -> 0.168098 / 0.356780
```

The runner now records RNG-neutral runtime statistics without changing behavior: fitted behavior calls, mean/max epsilon, mean disagreement, max raw epsilon before cap, cap-hit fraction and fractions epsilon >= `.10`/`.25`.

Physical 5×64 calibration now running after build/regression/smoke PASS:

```text
workflow 31450032347
  s1.25 cap .50
  s1.50 cap .50

workflow 31451592073
  s1.75 cap .50
  s2.00 cap .50
  s1.50 cap .65
  s1.50 cap .80
```

The cap variants specifically test whether `.50` is clipping the unstable tail. The selection policy prefers the smallest/interpretable gate-clearing candidate unless a more complex alternative has materially stronger robustness margin.

## Active size8 program

Short-horizon size8 2×128 already passed both cross gates:

```text
mean 0.139615
p95  0.329689
fits PASS
```

Mandatory five-iteration runs remain physically active:

```text
size8 no damping          workflow 31446308103
size8 + temporal w50      workflow 31448623827
size8 + uncertainty s1.0  workflow 31449546648
```

No interaction is assumed additive; every composition earns its own result.

## Independent baseline reproducibility

Workflow `31451518046` is physically executing the second independent authoritative size4 5×64 baseline after build/regression/smoke PASS. The existing comparator will then have two evidence commits at both its 256 and 320 targets and can require exact structural/sample/node equality plus cross-seed metric delta <= `1e-9`.

This is a baseline determinism check, separate from winner certification.

## Candidate checkpoint state — physically smoke-tested

R7.3 ensemble state is layered through the unchanged `SPINCORE_R7_CHECKPOINT_V2` extra field as:

```text
SPINCORE_R7_CANDIDATE_BEHAVIOR_V1
```

It preserves current side models, prior temporal models, behavior parameters and fit generation and reuses the restored authoritative primary as member zero.

A restore determinism bug was fixed: constructing temporary side networks would consume global torch RNG before loading their checkpoint weights. Side construction is now isolated with `torch.random.fork_rng(devices=[])`.

Main regression `31452600161`, commit `2423e1c2f22dee0557d0c7f280ac1f5e542f9e20`:

```text
C++ PASS
Python 40 passed
tiny uncertainty collect/train/checkpoint/restore/continue exact PASS
tiny temporal collect/train/checkpoint/restore/continue exact PASS
```

The temporal smoke explicitly verifies prior-ensemble preservation.

## Full winner certification pipeline

### Stage 1 — selection and immutable freeze

A deliberate `SPINCORE_R7_3_WINNER_SELECTION_V1` is required; none exists yet.

`tools/freeze_r7_3_candidate_semantics.py` is fail-closed and outputs:

```text
SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1
```

It refuses anything except an actual fit-valid 5×64 pass. It pins algorithm seeds, exact evidence commit and byte-identical SHA-256, source workflow/run/head, ensemble/behavior parameters, full execution hyperparameters, authoritative deck/RNG contract and Git object identities. The source head must be an ancestor of the evidence commit. Workflow overrides of frozen seeds, lr or device are rejected.

Workflow: `.github/workflows/r7_3_freeze_candidate_semantics.yml`.

### Stage 2 — exact-source fresh-process reproducibility

`tools/run_r7_3_frozen_candidate_fresh_repro.py` creates a detached worktree at the exact frozen source commit, rebuilds it, runs C++ and Python regressions, reruns the candidate and compares against the immutable evidence commit recursively.

Ignored fields:

```text
generated_at_unix
duration_seconds
```

Numeric tolerance: `1e-9`. Output:

```text
SPINCORE_R7_3_FROZEN_CANDIDATE_FRESH_REPRO_V1
```

Workflow: `.github/workflows/r7_3_frozen_candidate_fresh_repro.yml`.

### Stage 3 — physical checkpoint/resume recertification

Tools:

```text
tools/r7_3_frozen_candidate_checkpoint_worker.py
tools/run_r7_3_frozen_candidate_checkpoint_recert.py
```

Workflow:

```text
.github/workflows/r7_3_frozen_candidate_checkpoint_recert.yml
```

Output schema:

```text
SPINCORE_R7_3_CANDIDATE_CHECKPOINT_RECERT_V1
```

Fresh reproducibility must pass first. The exact frozen algorithm source runs a common prefix through iteration 3; one branch continues directly and the other restores a real checkpoint and continues. Exact final equality is required for:

```text
counters
Advantage reservoir state/order/RNG
Strategy reservoir state/order/RNG
bundle.batch_rng
global torch RNG
primary Advantage model
AveragePolicy model
both optimizer states
all current side Advantage models
previous temporal ensemble
fit generation
final fit/progress report
shared cross-seed observation corpus
cross-seed metrics
```

Any mismatch blocks acceptance scaling.

### Stage 4 — exact-source 640 acceptance

Tool:

```text
tools/run_r7_3_frozen_candidate_640_acceptance.py
```

Workflow:

```text
.github/workflows/r7_3_frozen_candidate_640_acceptance.yml
```

Output schema:

```text
SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1
```

It cannot execute unless the freeze, exact-source fresh reproducibility and checkpoint recertification all pass for the same provenance. It uses the exact frozen algorithm source and changes only the root scale:

```text
5 × 64  = 320 roots/seed durability
       ->
5 × 128 = 640 roots/seed acceptance
```

All frozen fit and cross-seed gates stay unchanged. A 640 pass marks `r7_3_ready_to_advance_to_r7_4=true`; table readiness remains false.

## Evidence consolidation

```text
SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4
15 candidates + baseline

SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V4
23 candidates + baseline = 24 rows
```

The expanded matrix includes six size4 uncertainty scale/cap calibrations plus size8 temporal-w50 and size8 uncertainty-s1.0. Ranking is evidence only.

## Promotion rule

R7.3 becomes PASS only after the selected candidate:

1. passes all fit gates at 5×64;
2. clears `mean <= 0.15` and `p95 <= 0.35` at 5×64;
3. is immutably frozen against its exact evidence/source provenance;
4. reproduces from the exact source in a fresh process;
5. passes exact continuous-vs-stop/restore/continue recertification;
6. passes the same frozen gates at exact-source 5×128 = 640;
7. keeps every acceptance threshold unchanged.

Then R7.4 larger HU+3H may begin. R8–R12 still remain before table use.
