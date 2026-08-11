# SpinCore finite roadmap — canonical recovery generation 2

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate through R12 passes.

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
  - R7.3 multi-seed stability — **5×64 WINNER SELECTED; CERTIFICATION ACTIVE**
  - R7.4 larger held-out HU + 3H pilot — **FINITE GATE PRECOMMITTED; BLOCKED BY R7.3**
- R8 Production training — TODO after R7.4 PASS
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## R7.3 frozen gates and execution contract

```text
Advantage weighted NRMSE <= 0.75
AveragePolicy weighted mean TV <= 0.12
cross-seed mean TV <= 0.15
cross-seed p95 TV <= 0.35
selection seeds = 20260829, 20260807
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
global_root continuous across iterations
partial-exact opponent level = 2
primary RNG = one persistent live bundle.batch_rng
training device = cpu
learning rate = 0.001
```

No gate has been relaxed.

## R7.3 causal resolution

The confirmed instability chain is:

```text
Advantage approximation
-> nonlinear regret matching
-> divergent behavior
-> divergent next trajectories
-> divergent strategy targets
-> repeated CFR feedback amplification
```

Iteration-1 shared strategy targets are identical under initial uniform behavior. The first fitted Advantage feedback is the first physically confirmed transition to large shared-target divergence in iteration 2. Support fragmentation and exact shared-state disagreement are both material.

The strongest durable control mechanism is state-local **uncertainty-adaptive damping** of a policy mixture. It uses disagreement among independently fitted regret policies as a proxy for epistemic instability and mixes only uncertain states toward legal uniform behavior.

## R7.3 selected winner

The completed uncertainty sweep produced several genuine 5×64 gate passes. Deliberate selection chose the smallest robust winner rather than automatically selecting the numerically lowest mean.

### Selected: `size4_uncertainty_s175`

```text
ensemble size = 4
epsilon scale = 1.75
epsilon cap = 0.50
5 CFR iterations × 64 roots = 320 roots/seed
mean TV = 0.1329178512       PASS
p50 TV  = 0.1155516654
p95 TV  = 0.2854667008       PASS
max TV  = 0.5646659136
all per-seed fit gates       PASS
R7.3 durability gate         PASS
```

Margins:

```text
mean gate margin = 0.0170821488
p95 gate margin  = 0.0645332992
```

Authoritative provenance:

```text
source workflow run = 31451592073
source head         = 01edcb4697ae07f8f379d79b0b4b8e43e309d65e
evidence commit     = 05c0976e8311874ea9a55f5c899a088abe3b4f00
evidence file       = validation/R7_3_POLICY_MIXTURE_UNCERTAINTY_DAMPING_s175_320.json
selection file      = validation/R7_3_WINNER_SELECTION.json
```

A size8 uncertainty candidate achieved slightly lower mean TV but required twice as many Advantage models and had worse p95/max behavior. The size4 s175 candidate therefore won the complexity/robustness decision.

**This 5×64 PASS does not by itself make R7.3 PASS.** Certification remains mandatory.

## R7.3 corrected certification chain

The selected winner is frozen by `SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1`, including exact evidence bytes, source commit/tree identities, hyperparameters, deck/RNG contract and behavior parameters.

A certification audit uncovered two ways a nominal “exact-source” rerun could silently cease to be exact:

1. certification workflows had injected PyTorch/OpenMP/MKL thread-count overrides that the source evidence workflow never set;
2. the checkpoint worker could execute from current-main `tools`, causing imports to resolve current-main training modules instead of the frozen source tree.

Both paths are now rejected. The corrected contract is:

```text
thread_environment_contract = SOURCE_WORKFLOW_NO_EXPLICIT_THREAD_OVERRIDE
certifier injects no thread-count overrides
checkpoint helper + worker are overlaid into detached frozen-source worktree
worker executes from frozen-source worktree/tools
all behavior/training imports resolve there
```

There is an older workflow run `31468467215` using the pre-hardening snapshot. It is explicitly **NONAUTHORITATIVE**.

Because old and corrected Actions can write the same canonical evidence paths, certification now uses `tools/r7_3_certification_evidence.py`: current HEAD evidence is accepted only if it satisfies immutable provenance; otherwise Git history is searched newest-first for the newest provenance-valid version. If none exists, certification fails closed. A stale legacy overwrite can therefore no longer silently replace corrected evidence.

### Corrected active cascade

Workflow `31469278146`:

```text
semantic freeze                    PASS
exact-source fresh reproduction    PHYSICAL IN PROGRESS
checkpoint recertification         waits for fresh PASS
5×128 = 640 acceptance             waits for checkpoint PASS
```

Fresh reproduction requires recursive evidence equality with numeric tolerance `1e-9`, ignoring only `generated_at_unix` and `duration_seconds`. Checkpoint recertification requires exact final state across counters, both reservoirs and their RNGs, live batch RNG, global torch RNG, primary/side networks, AveragePolicy, optimizers and wrapper state.

Only a corrected exact-source 640 PASS changes R7.3 to PASS and authorizes R7.4.

## Candidate checkpoint determinism

Base checkpoint remains `SPINCORE_R7_CHECKPOINT_V2`. Ensemble behavior state is stored in its existing `extra` field under:

```text
SPINCORE_R7_CANDIDATE_BEHAVIOR_V1
```

Side model reconstruction is global-torch-RNG neutral via `torch.random.fork_rng(devices=[])`, fixing a hidden restart divergence mechanism. The physical recert worker supports the selected uncertainty mechanism and temporal candidates, but only the selected winner will receive full certification.

## R7.4 gate is now finite and precommitted

The full design is frozen before observing any R7.4 strategic result in:

```text
validation/R7_4_GATE_DESIGN_20260811.md
```

R7.4 carries the exact accepted s175 algorithm forward. It does **not** search for another algorithm or tune thresholds after seeing domain results.

### Held-out seed policy

R7.3 selection seeds are forbidden. Two R7.4 seeds are derived deterministically from the immutable frozen evidence SHA-256:

```text
SHA256("SpinCore|R7.4|heldout|index|" + evidence_sha256)
-> positive 31-bit seed
-> reject zero, duplicates, and R7.3-seed collisions
```

The same held-out seed pair is used throughout R7.4.

### Scenario coverage

TRUE_HEADS_UP uses six deterministic stack/dealer variants:

```text
(0,750,750), (0,500,1000), (0,1000,500)
× dealer 1/2
```

THREE_HANDED uses fifteen deterministic variants:

```text
(500,500,500)
(250,500,750)
(250,750,500)
(500,250,750)
(750,250,500)
× dealer 0/1/2
```

Every scenario must be exercised. Scenario selection is deterministic by global root index and the authoritative deck formula is retained.

### R7.4 finite physical sequence

```text
A. corrected accepted-source structural HU/3H preflight
B. held-out HU:  5 × 128 = 640 roots/seed
C. held-out 3H:  5 ×  64 = 320 roots/seed screen
D. only if B+C PASS: held-out 3H 5 × 128 = 640 roots/seed confirmation
E. R7.4 PASS only if B + C + D all PASS
```

The 3H 320 stage is only a compute filter and can never finish R7.4 by itself.

R7.4 reuses the same fit and cross-seed thresholds as R7.3. No easier 3H-specific gate exists.

Prepared automation:

```text
.github/workflows/r7_4_domain_preflight.yml
.github/workflows/r7_4_heldout_domain_screen.yml
.github/workflows/r7_4_three_handed_640_confirmation.yml

tools/run_r7_4_domain_preflight.py
tools/run_r7_4_stability_pilot.py
tools/r7_4_stability_pilot_worker.py
tools/summarize_r7_4_heldout_screen.py
tools/finalize_r7_4_gate.py
```

Final schema:

```text
SPINCORE_R7_4_FINAL_GATE_V1
```

R7.4 PASS authorizes **R8 only**, never table use.

## Regression state

Main regression workflow `31470937877`, commit `b0aadb263be39c8ed83d6c8673c99ae2fcb80705`:

```text
C++ regression PASS
Python 65 passed
```

The Python suite includes exact-source/provenance rejection, stale-evidence Git-history recovery, checkpoint RNG neutrality, R7.4 held-out seed derivation, HU/3H scenario coverage, seed-local uncertainty diagnostics and the finite R7.4 final gate.

## Closed / deprioritized primary branches

Raw root scaling, independent x8/x16 path multiplication as standalone fix, common-path RNG, antithetic x4, exhaustive opponent expectation, simply increasing optimizer capacity, behavior-aware MSE auxiliary objective, duplicate-target aggregation, multistart selection, raw Advantage ensemble standalone, common-mode centering, robust median/trimmed aggregation, card/suit rewrite as dominant explanation, ordinary Direct Behavior, aggregated-regret Direct Behavior, regret-floor as primary mechanism and direct plain-size4 640 escalation are closed or deprioritized.

## Remaining finite path to table use

```text
R7.3 corrected fresh PASS
-> R7.3 checkpoint recert PASS
-> R7.3 exact-source 640 PASS
-> R7.4 structural PASS
-> R7.4 held-out HU640 + 3H320 PASS
-> R7.4 held-out 3H640 PASS
-> R8 production training
-> R9 strategic audit
-> R10 OpenHoldem runtime
-> R11 safe exploitation
-> R12 operational homologation
-> READY FOR TABLES
```

No intermediate success bypasses a later stage.

`READY FOR TABLES = NO`.
