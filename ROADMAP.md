# SpinCore finite roadmap — canonical state 2026-08-12

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate through R12 passes and every release debt, including the deferred R7.3 exact-reproducibility debt, is closed.

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
  - R7.4 staged-resume equivalence — **PASS EXACT**
  - R7.4 held-out HU 640 — **PASS**
  - R7.4 held-out 3H 320 screen — **PASS**
  - R7.4 held-out 3H 640 confirmation — **IN PROGRESS; STAGE 1 ACTIVE FOR BOTH SEEDS**
  - R7.4 final gate — **PENDING 3H640**
- R8 Production training — **OFFICIAL TRAINING BLOCKED UNTIL R7.4 FINAL + R8.0 EXACT PROFILE**
  - R8.0 production-profile acquisition/validation pipeline — **INFRASTRUCTURE PASS; EXACT SELECTED-STATE DATA BLOCKED**
  - R8.1 deterministic production infrastructure — **PASS INFRASTRUCTURE**
  - R8.2 Ryzen calibration selector/precommit — **PASS INFRASTRUCTURE; PHYSICAL CALIBRATION NOT RUN**
  - R8.3–R8.5 official training/freeze — **BLOCKED**
- R9 Strategic audit — **FINITE GATE DESIGN FROZEN; EXECUTION BLOCKED UNTIL R8.5**
- R10 OpenHoldem runtime — **FINITE GATE DESIGN FROZEN; EXECUTION BLOCKED UNTIL R9 PASS**
- R11 Safe exploitation — **FINITE GATE DESIGN FROZEN; EXECUTION BLOCKED UNTIL R10 PASS**
- R12 Operational homologation — **FINITE FINAL GATE DESIGN FROZEN; EXECUTION BLOCKED UNTIL R11 PASS**

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
R7.4 held-out seeds = 1954132610, 372483540
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
global_root continuous across iterations
partial-exact opponent level = 2
primary RNG = one persistent live bundle.batch_rng
production utility = ICM_EXACT_V1 explicit payout delta
thread contract = SOURCE_WORKFLOW_NO_EXPLICIT_THREAD_OVERRIDE
```

No strategic threshold has been relaxed.

## R7.3 strategy-quality prerequisite and exact-reproducibility debt

The original frozen 5×64 winner remains:

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

The unchanged provisional 5×128 strategy-quality bridge passed at 640 roots/seed:

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

Strict fresh-process recertification remains unresolved:

```text
fresh_process_reproducible = false
difference_count = 734 report fields
numeric tolerance = 1e-9
strict run = 31565565329
```

This remains explicit debt, not PASS. It does not block the controlled R7.4/R8 engineering path, but **must be resolved before R12 can emit `READY FOR TABLES = YES`**. No tolerance, seed, gate or thread hack may be used to relabel the debt.

## R7.4 accepted physical evidence

Rules/invariance and structural preflight are accepted under `SPINRULESET-4` at extension source head:

```text
e43b2cfea31f927393cf2751485d712902d6f02d
```

Staged continuation has an exact physical mechanism proof:

```text
validation/R7_4_STAGED_RESUME_EQUIVALENCE.json
all_exact = true
```

### Held-out HU640 — PASS

```text
validation/R7_4_HELDOUT_HU_640.json
roots/seed = 640
mean TV = 0.12478918582201004
p95 TV = 0.2732357978820801
r7_4_domain_stability_pass = true
```

### Held-out 3H320 — PASS

Authoritative repaired staged workflow:

```text
run = 31637418697
validation/R7_4_HELDOUT_3H_320.json
validation/R7_4_HELDOUT_SCREEN_SUMMARY.json
roots/seed = 320
```

Final per-seed gates:

```text
seed 1954132610:
  Advantage NRMSE = 0.5119520425796509   PASS
  Policy mean TV  = 0.1004391759634018   PASS

seed 372483540:
  Advantage NRMSE = 0.5032332539558411   PASS
  Policy mean TV  = 0.1074957475066185   PASS
```

Cross-seed screen:

```text
mean TV = 0.10584357380867004    PASS
p95 TV  = 0.2369937151670456     PASS
max TV  = 0.6491326093673706     diagnostic only
all scenarios exercised          PASS
r7_4_heldout_screen_pass          true
```

### Held-out 3H640 — ACTIVE

The successful 3H320 evidence automatically triggered:

```text
workflow = .github/workflows/r7_4_three_handed_640_confirmation.yml
run = 31661899987
gate = PASS
stage 1 = ACTIVE for seeds 1954132610 and 372483540
roots/iteration = 128
iterations = 5
roots/seed = 640
```

The confirmation runtime is explicitly frozen and regression-guarded:

```text
Ubuntu 24.04
Python 3.11.15
PyTorch 2.13.0+cpu
heartbeat every 300 s
```

R7.4 final PASS remains false until the 3H640 confirmation itself passes the unchanged per-seed and cross-seed gates.

## R8 preparation already accepted without starting official training

R8.0 has a fail-closed production-profile schema/evidence acquisition pipeline, but exact first-party selected-state `buy-in × multiplier` mappings for all state-dependent stack/blind/payout semantics are still missing. Pilot constants are forbidden substitutes.

R8.1 production infrastructure has accepted deterministic independent-stream scheduling, central Algorithm-R state, durable scheduler checkpoints and integrated semantic transactions. Same-stream root-level parallelism remains forbidden because it would alter the persistent live RNG contract.

R8.2 has an accepted calibration selector/precommit. Candidate concurrency is eligible only if it reproduces the exact validated R8.1 transaction-generation identities; among semantically exact error-free candidates, highest throughput wins and exact ties prefer lower concurrency. CPU utilization is telemetry, not an acceptance target. Physical Ryzen calibration is not yet authorized.

## Strategic sentinels and finite downstream gates

Action-level sentinel infrastructure is accepted:

```text
python/spincore/strategic_sentinel.py
python/spincore/sentinel_state_catalog.py
validation/STRATEGIC_ACTION_SENTINEL_GATE_DESIGN_20260812.md
framework regression = 31661499555 PASS
state-catalog regression = 31661639444 PASS
```

This is infrastructure only. The production sentinel set, exact integrity baselines and numerical strategic plausibility bounds are intentionally not frozen until exact production profiles and policy identities exist. Integrity-only evidence can never substitute for strategic plausibility.

Finite downstream gate designs are now frozen:

```text
validation/R9_STRATEGIC_AUDIT_GATE_DESIGN_20260812.md
validation/R10_OPENHOLDEM_RUNTIME_GATE_DESIGN_20260812.md
validation/R11_SAFE_EXPLOITATION_GATE_DESIGN_20260812.md
validation/R12_OPERATIONAL_HOMOLOGATION_GATE_DESIGN_20260812.md
```

R12.9 is the only gate allowed to emit `READY FOR TABLES = YES`, and only after all earlier gates pass and all release debts — specifically including R7.3 exact reproducibility — are closed.

## Remaining finite path to table use

```text
R7.4 3H640 confirmation ACTIVE
-> R7.4 FINAL PASS
-> R8.0 exact production profiles
-> R8.2 physical Ryzen calibration
-> R8.3 official HU training
-> R8.4 official 3H training
-> R8.5 immutable production-policy freeze
-> R9 strategic audit
-> R10 OpenHoldem runtime integration
-> R11 safe exploitation
-> R12 operational homologation
-> close every release debt including R7.3 exact reproducibility
-> R12.9 READY FOR TABLES gate
```

R7.4 PASS authorizes **R8 only**, never table use.

`READY FOR TABLES = NO`.
