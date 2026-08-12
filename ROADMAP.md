# SpinCore finite roadmap — canonical state 2026-08-12

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate through R12 passes and the deferred R7.3 exact-reproducibility debt is closed.

## Canonical roadmap status

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
  - R7.2 LCFR weighting / checkpoint+resume infrastructure / fresh-process worker — **PASS REBUILT**
  - R7.3 selected strategy quality at 640 roots/seed — **PASS FOR PROVISIONAL R7.4 ADVANCEMENT**
  - R7.3 exact fresh-process reproducibility — **OPEN RELEASE/CERTIFICATION DEBT; NOT PASS**
  - R7.4 SPINRULESET-4 source invariance — **PASS**
  - R7.4 structural HU/3H preflight — **PASS**
  - R7.4 held-out HU 640 — **PASS**
  - R7.4 held-out 3H 320 — **IN PROGRESS VIA DETERMINISTIC STAGED RESUME**
  - R7.4 held-out 3H 640 confirmation — **BLOCKED UNTIL 3H320 PASS**
- R8 Production training — **BLOCKED UNTIL FINITE R7.4 FINAL PASS**
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

No intermediate success authorizes table use.

## Frozen strategic contract

```text
selected behavior = size4_uncertainty_s175
behavior semantic = SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1
ensemble size = 4
epsilon scale = 1.75
epsilon cap = 0.50

Advantage weighted NRMSE <= 0.75
AveragePolicy weighted mean TV <= 0.12
cross-seed mean TV <= 0.15
cross-seed p95 TV <= 0.35

R7.3 selection seeds = 20260829, 20260807
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
global_root continuous across iterations
partial-exact opponent level = 2
primary RNG = one persistent live bundle.batch_rng
production utility = ICM_EXACT_V1 explicit payout delta
thread contract = SOURCE_WORKFLOW_NO_EXPLICIT_THREAD_OVERRIDE
```

No strategic threshold has been relaxed.

## R7.3 selected winner and 640 strategy bridge

Original frozen 5×64 winner:

```text
roots/seed = 320
mean TV = 0.1329178512096405       PASS
p95 TV  = 0.2854667007923126       PASS
all per-seed fit gates             PASS
source workflow = 31451592073
source head = 01edcb4697ae07f8f379d79b0b4b8e43e309d65e
evidence commit = 05c0976e8311874ea9a55f5c899a088abe3b4f00
evidence SHA256 = 39bc31e0198df1ba8b6b5033271ae8da839ec32a80cd42a22b09647b9b1e130e
```

The unchanged 5×128 provisional strategy-quality bridge subsequently passed at 640 roots/seed:

```text
workflow = 31579597855
evidence commit = 872f53a053ac83160be54977715ba1ceae4d8b25
mean TV = 0.13625219464302063       PASS
p95 TV  = 0.3153517544269562        PASS
seed 20260829 Advantage NRMSE = 0.48132333159446716
seed 20260829 Policy TV = 0.0904657244682312
seed 20260807 Advantage NRMSE = 0.4868704676628113
seed 20260807 Policy TV = 0.09239403158426285
```

This is the strategy-quality prerequisite that authorizes provisional R7.4 engineering.

## R7.3 exact-reproducibility debt

Strict fresh-process recertification remains unresolved:

```text
fresh_process_reproducible = false
difference_count = 734 report fields
numeric tolerance = 1e-9
strict run = 31565565329
```

The failure is not reclassified as PASS and is not hidden by tolerance or thread hacks. Decision record:

```text
validation/R7_3_EXACT_REPRO_DEFERRED_DECISION_20260812.md
```

Controlled deferral means:

1. exact reproducibility does not block R7.4/R8 engineering while unchanged strategy-quality gates continue to pass;
2. exact reproducibility remains explicit release/certification debt;
3. the debt **must close before `READY FOR TABLES`**;
4. action-level canonical/extreme sentinels are also required before table use.

## R7.4 accepted rules source and structural gate

R7.4 extends the frozen training implementation with `SPINRULESET-4` at source head:

```text
e43b2cfea31f927393cf2751485d712902d6f02d
```

HU invariance is based on byte identity of the selected R7.3 training components plus extension regression, not on pretending the unresolved historical numeric reproduction passed.

Authoritative accepted evidence:

```text
validation/R7_4_RULESET_ACCEPTANCE.json
hu_invariance_pass = true
selected_training_components_byte_identical = true
historical_numeric_evidence_reproduction_evaluated = false
historical_exact_reproducibility_debt_preserved = true
```

Structural preflight also passed:

```text
corrected workflow = 31600267534
case_count = 15
HU cases = 6
all_chip_zero_sum = true
all_clone_neural_exact = true
all_icm_zero_sum_within_1e12 = true
```

## R7.4 finite physical sequence

The precommitted design remains:

```text
A. accepted SPINRULESET-4 source invariance + structural preflight
B. held-out HU:  5 × 128 = 640 roots/seed
C. held-out 3H:  5 ×  64 = 320 roots/seed screen
D. only if B+C PASS: held-out 3H 5 × 128 = 640 roots/seed confirmation
E. R7.4 PASS only if B + C + D all PASS
```

Held-out seeds are derived mechanically from the immutable winner evidence hash and never reuse the R7.3 selection seeds. Current pair:

```text
1954132610
372483540
```

Scenario cycles remain fixed at 6 TRUE_HEADS_UP variants and 15 THREE_HANDED stack/dealer variants.

### Current physical evidence

HU640 passed unchanged gates:

```text
validation/R7_4_HELDOUT_HU_640.json
mean TV = 0.12478918582201004
p95 TV = 0.2732357978820801
r7_4_domain_stability_pass = true
```

The original combined 3H320 job was **cancelled by the GitHub-hosted runner's 6-hour ceiling while still progressing normally**. That cancellation is infrastructure, not a strategic failure.

Runtime consistency relevant to the accepted 640 bridge and current held-out execution is:

```text
Ubuntu 24.04
Python 3.11.15
PyTorch 2.13.0+cpu
```

To preserve the exact frozen strategic test without reducing roots, changing seeds, or relaxing gates, 3H execution is now split at deterministic post-Advantage-fit iteration boundaries using the existing authoritative checkpoint state:

```text
SPINCORE_R7_CHECKPOINT_V2
SPINCORE_R7_CANDIDATE_BEHAVIOR_V1
SPINCORE_R7_4_STAGED_CHECKPOINT_V1
```

The checkpoint preserves both reservoirs and reservoir RNGs, live `bundle.batch_rng`, global torch RNG, models, optimizers, counters, ensemble side models, behavior diagnostics, global-root position, scenario counts and iteration reports.

Before staged 3H evidence is accepted, a physical continuous-versus-resumed THREE_HANDED regression must report exact equality:

```text
SPINCORE_R7_4_STAGED_RESUME_EQUIVALENCE_V1
all_exact = true
```

Current staged repair workflow:

```text
.github/workflows/r7_4_three_handed_staged_repair.yml
run = 31637418697
status = IN PROGRESS
```

The 3H640 confirmation workflow has already been converted to the same deterministic staged mechanism and will gate on a genuine HU640 + 3H320 screen PASS.

## Remaining finite path to table use

```text
R7.4 staged-resume equivalence PASS
-> R7.4 3H320 unchanged strategic gates PASS
-> R7.4 3H640 unchanged strategic gates PASS
-> R7.4 FINAL PASS
-> R8 production training
-> R9 strategic audit
-> R10 OpenHoldem runtime
-> R11 safe exploitation
-> R12 operational homologation
-> close R7.3 exact-reproducibility release debt
-> complete required action-level sentinels
-> READY FOR TABLES
```

R7.4 PASS authorizes **R8 only**, never table use.

`READY FOR TABLES = NO`.
