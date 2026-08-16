# SpinCore finite roadmap — canonical state 2026-08-16

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate through R12 passes and every release debt, including the deferred R7.3 exact-reproducibility debt, is closed.

## Canonical roadmap status

- R0 Foundation / canonical repository — **PASS REBUILT**
- R1 Complete poker engine — **PASS REBUILT**
- R2 Canonical infoset + neural encoder — **PASS REBUILT; V1 IS NOW CONTROL/FALLBACK FOR R7.5, NOT PRODUCTION FREEZE**
- R3 Tournament continuation value (`ICM_EXACT_V1`, explicit payout) — **PASS REBUILT**
- R4 Neural infrastructure — **PASS REBUILT**
- R5 CFR correctness oracle — **PASS REBUILT**
- R6 Deep CFR integration on authoritative `SpinTraversalState` — **PASS REBUILT**
- R7 Pilot / performance / statistical stability — **R7.4 PASS; REPRESENTATION REVIEW ADDED BEFORE PRODUCTION**
  - R7.0 approximation metrics / full-reservoir audit — **PASS REBUILT**
  - R7.1 native own-reach frontier — **PASS REBUILT**
  - R7.2 LCFR weighting / checkpoint+resume infrastructure / fresh-process worker — **PASS REBUILT**
  - R7.3 selected strategy quality at 640 roots/seed — **PASS FOR PROVISIONAL ADVANCEMENT**
  - R7.3 exact fresh-process reproducibility — **OPEN RELEASE/CERTIFICATION DEBT; NOT PASS**
  - R7.4 SPINRULESET-4 source invariance — **PASS**
  - R7.4 structural HU/3H preflight — **PASS**
  - R7.4 staged-resume equivalence — **PASS EXACT**
  - R7.4 held-out HU 640 — **PASS**
  - R7.4 held-out 3H 320 screen — **PASS**
  - R7.4 held-out 3H 640 confirmation — **PASS**
  - R7.4 final gate — **PASS; READY TO ADVANCE TO R8 ENGINEERING**
- R7.5 Strategic representation & action abstraction — **IN PROGRESS; FINITE-CLOSURE POLICY ACTIVE**
  - R7.5.0 legacy evidence + architecture precommit — **PASS AS DESIGN PRECOMMIT ONLY**
  - R7.5.1 recover/regenerate + audit flop mappings — **PASS AS DIAGNOSTIC/STRUCTURAL INPUT; LEGACY 184 DEFECT ESTABLISHED; SPNNIV3 SUCCESSOR IMPLEMENTED**
  - R7.5.2 representation semantics/integration — **SPNNIV3 STRUCTURAL/SEMANTIC INTEGRITY IMPLEMENTED; V2 IS NOT THE PRODUCTION MIGRATION TARGET**
  - R7.5.3 frozen representation admission/selection — **IN PROGRESS; H2/H3 LOCAL TRAINING PASS; CROSS-SEED BLOCKER DIAGNOSED AS DECK/CHANCE DOMINANT; X4 CHANCE-COVERAGE READMISSION IN PROGRESS**
  - R7.5.4 frozen action-abstraction ablation — **PENDING PROVISIONAL REPRESENTATION WINNER**
  - R7.5.5 production representation/action freeze — **PENDING**
- R8 Production training — **OFFICIAL TRAINING BLOCKED UNTIL R7.5.5 FINAL + R8.0 EXACT PROFILE**
  - R8.0 production-profile acquisition/validation pipeline — **INFRASTRUCTURE PASS; EXACT SELECTED-STATE DATA BLOCKED**
  - R8.1 deterministic production infrastructure — **PASS INFRASTRUCTURE**
  - R8.2 Ryzen calibration selector/precommit — **PASS INFRASTRUCTURE; GENERIC FROZEN LOCAL-RUN EVIDENCE WRAPPER ADDED; PHYSICAL CALIBRATION NOT RUN**
  - R8.3–R8.5 official training/freeze — **BLOCKED BY R7.5.5 + R8.0**
- R9 Strategic audit — **FINITE GATE DESIGN FROZEN; EXECUTION BLOCKED UNTIL R8.5**
- R10 OpenHoldem runtime — **FINITE GATE DESIGN FROZEN; EXECUTION BLOCKED UNTIL R9 PASS**
- R11 Safe exploitation — **FINITE GATE DESIGN FROZEN; EXECUTION BLOCKED UNTIL R10 PASS**
- R12 Operational homologation — **FINITE FINAL GATE DESIGN FROZEN; EXECUTION BLOCKED UNTIL R11 PASS**

No intermediate success authorizes table use.

## Frozen R7.3/R7.4 strategic contract

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

No R7.3/R7.4 strategic threshold has been relaxed. R7.5 does not retroactively reinterpret those gates; it determines whether the recovered V1 representation/action abstraction is suitable for production or should be replaced before official training.

## R7.3 exact-reproducibility debt

The frozen strategy-quality evidence passed, including the provisional 640 roots/seed bridge, but strict fresh-process exact reproduction remains unresolved:

```text
fresh_process_reproducible = false
difference_count = 734 report fields
numeric tolerance = 1e-9
strict run = 31565565329
```

This remains explicit debt, not PASS. It does not block controlled R7.5/R8 engineering, but **must be resolved before R12 can emit `READY FOR TABLES = YES`**. No tolerance, seed, gate or thread hack may be used to relabel the debt.

## R7.4 final physical evidence — PASS

Final gate:

```text
validation/R7_4_FINAL_GATE.json
r7_4_pass = true
r7_4_ready_to_advance_to_r8 = true
ready_for_tables = false
```

Held-out evidence:

```text
HU640: PASS
3H320: PASS
3H640: PASS
```

3H640 confirmation used 640 roots/seed and passed all unchanged per-seed/coverage gates. Cross-seed confirmation:

```text
mean TV = 0.08999575674533844    PASS
p95 TV  = 0.20019790530204773    PASS
max TV  = 0.4369678199291229     diagnostic only
all scenarios exercised          PASS
```

R7.4 authorizes further engineering only. It does not prove that the current neural representation/action abstraction is the best production design and never authorizes table use.

## R7.5 — strategic representation & action abstraction

Legacy Spin & Go attempts were audited before production training. Canonical evidence includes the historical audits plus the SPNNIV3 integrity/admission chain. The old 184/V2 investigations are evidence feeding R7.5; they are not additional permanent roadmap branches.

Key frozen decisions:

```text
- exact poker/traversal state stays exact;
- lossy compression is allowed only at neural-observation boundary;
- NeuralInputV1 is control/fallback, not a production freeze;
- V2 is an intermediate attempt, not the current production migration target;
- SPNNIV3 is the intended successor lineage;
- H2 and H3 are the current SPNNIV3 admission candidates;
- exact stack/pot/SPR facts remain available;
- actor-aware and sizing-aware complete action history is required;
- richer postflop action sets must justify every extra branch by strategic gain vs Ryzen cost.
```

### Recovered 184 mapping and ontology evidence

The missing historical `184Flops.json` was recovered. It covered all 22,100 physical flops and 184 representatives, but 40 exact suit-isomorphic classes were split by legacy suit spelling, so the historical map is not eligible unchanged. This defect is preserved as evidence; SPNNIV3 no longer relies on that map as its primary representation.

The regression-proven C++ postflop ontology scaffold decomposes semantic lines such as c-bet, donk, probe, float, delayed lines and raises into compositional state facts rather than copying old strategy rules. Exact continuous geometry remains available.

### R7.5.3 current admission state — 2026-08-16

SPNNIV3 structural integrity is implemented with complete variable-length structured history, universal deduplicated action slots, exact HU/3H geometry, exact suit/rank invariance at the neural boundary, and an action width of 10. H2 is the exact relational/history candidate; H3 adds objective poker semantics.

The frozen Phase-2 local training gates passed for all eight H2/H3 × HU/3H × training-seed cells. The first complete strategic admission did **not** pass because every required cross-seed policy-stability row exceeded the unchanged `mean TV <= 0.15` and `p95 TV <= 0.35` gates. Independent parity auditing reproduced that failure and ruled out the covered evaluator/state/action identity mismatch path.

Winner-independent decomposition then isolated the dominant variance source without selecting H2/H3:

```text
same strategy memory / different final-policy learner: about 0.107 mean TV / 0.282 p95 TV

different strategy memory / common final learner:     about 0.243 mean TV / 0.624 p95 TV
=> upstream strategy-memory generation dominant

learning/memory RNG sensitivity:                       about 0.148 / 0.401
sampling/traversal RNG sensitivity:                     about 0.239 / 0.629
=> sampling/traversal family dominant

deck/chance sensitivity:                               0.239170 / 0.624240
traversal-action-sampling sensitivity:                  about 0.14986 / 0.39321
=> DECK_CHANCE_DOMINANT under the frozen 1.20x rule
```

The active primary remediation increases independent chance coverage from 64 to 256 roots per iteration while preserving the two independent training seeds, `deck_seed`, global-root semantics, scenario cycle, model/training budget and all hard gates. A mechanical zero-root reporting division in the first GitHub execution was frozen and corrected without changing the experiment; the corrected recovery is the same x4 remediation, not a new roadmap stage or extra strategic attempt.

Finite closure is now authoritative under:

```text
validation/R7_5_FINITE_CLOSURE_AND_COMPUTE_POLICY_20260816.md
```

R7.5.3 permits the active x4 remediation plus **at most one** final winner-independent chance-variance remediation if x4 fails. If x4 passes, one complete frozen strategic Phase-2 evaluation decides H2/H3. If the final permitted remediation also fails, R7.5.3 closes FAIL/BLOCKED rather than spawning R7.5.3D/E/F. No gate may be relaxed. If both admitted candidates remain strategically inconclusive after passing, the already-frozen smaller/faster H2 tie-break applies.

Only after a provisional representation winner exists may R7.5.4 strategic action/sizing revalidation begin. Only R7.5.5 may freeze the production encoder/action abstraction.

## R8 preparation already accepted without starting official training

R8.0 has a fail-closed production-profile schema/evidence acquisition pipeline, but exact first-party selected-state `buy-in × multiplier` mappings for all state-dependent stack/blind/payout semantics are still missing. Pilot constants are forbidden substitutes.

R8.1 production infrastructure has accepted deterministic independent-stream scheduling, central Algorithm-R state, durable scheduler checkpoints and integrated semantic transactions. Same-stream root-level parallelism remains forbidden because it would alter the persistent live RNG contract.

R8.2 has an accepted calibration selector/precommit. Candidate concurrency is eligible only if it reproduces the exact validated R8.1 transaction-generation identities; among semantically exact error-free candidates, highest throughput wins and exact ties prefer lower concurrency. CPU utilization is telemetry, not an acceptance target. **Physical Ryzen calibration has not run and is not marked PASS.**

Heavy CPU-bound experiments and official training are now assigned to the Ryzen when their scale makes GitHub runner chaining inefficient. GitHub remains the frozen-contract/referee/certification environment. `tools/spincore_ryzen_frozen_runner.py` records exact commit, tracked-worktree state, contracts, runtime, command/log and SHA-256 artifact inventory for future heavy local executions; this infrastructure does not itself authorize R8 training.

R8.0 data acquisition and other non-strategic engineering may proceed while R7.5 executes. R8.3/R8.4 official production training may not start until both R7.5.5 and R8.0 prerequisites are satisfied.

## Strategic sentinels and finite downstream gates

Action-level sentinel infrastructure is accepted:

```text
python/spincore/strategic_sentinel.py
python/spincore/sentinel_state_catalog.py
validation/STRATEGIC_ACTION_SENTINEL_GATE_DESIGN_20260812.md
```

This is infrastructure only. The production sentinel set, exact integrity baselines and numerical strategic plausibility bounds are not substitutes for the R7.5 representation/action selection.

Finite downstream gate designs remain:

```text
validation/R9_STRATEGIC_AUDIT_GATE_DESIGN_20260812.md
validation/R10_OPENHOLDEM_RUNTIME_GATE_DESIGN_20260812.md
validation/R11_SAFE_EXPLOITATION_GATE_DESIGN_20260812.md
validation/R12_OPERATIONAL_HOMOLOGATION_GATE_DESIGN_20260812.md
```

R12.9 is the only gate allowed to emit `READY FOR TABLES = YES`, and only after all earlier gates pass and all release debts — specifically including R7.3 exact reproducibility — are closed.

## Remaining finite path to table use

```text
R7.4 FINAL PASS
-> R7.5.3 representation admission/selection [active; finite]
-> R7.5.4 action/sizing audit
-> R7.5.5 production representation/action freeze
-> R8.0 exact production profiles
-> R8.2 physical Ryzen calibration under selected R7.5 architecture
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

`READY FOR TABLES = NO`.
