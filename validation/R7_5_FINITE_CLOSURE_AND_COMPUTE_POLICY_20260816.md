# R7.5 finite-closure and compute-execution policy — 2026-08-16

Status: **ACTIVE GOVERNANCE POLICY**  
Roadmap effect: **no new roadmap stage**  
`READY FOR TABLES = NO`

## Purpose

R7.5 is a finite production-engineering gate, not an open-ended research program. Internal diagnostics may explain a failed gate, but their names (`3C`, RNG factorial, deck/traversal split, runtime recovery, etc.) do **not** create new roadmap nodes and do not extend the project indefinitely.

The canonical roadmap remains R0 through R12. The current roadmap item is **R7.5.3 — production representation admission/selection**.

## Finite exit contract for R7.5.3

The representation candidates remain H2 and H3 under the already frozen Phase-2 contracts and thresholds. There are only two permitted winner-independent stability-remediation attempts from the present state:

1. **Primary remediation — independent chance coverage x4.** This is the currently frozen experiment: 256 roots/iteration, 3 iterations, independent training seeds, unchanged `deck_seed`, unchanged scenario cycle and unchanged hard gates.
2. **Final contingency remediation — at most one further winner-independent chance-variance correction.** It may be executed only if the x4 remediation fails the unchanged hard stability gates. Its algorithm, seeds, budget, acceptance rule and output interpretation must be frozen before candidate outputs are inspected.

A mechanical rerun/recovery of the *same* frozen experiment does not consume an additional remediation attempt when it changes no strategic algorithm, seed, coverage target or gate. The x4 zero-root reporting correction frozen in `R7_5_3C_CHANCE_COVERAGE_X4_RUNTIME_CORRECTION_FREEZE_20260816.json` is such a mechanical recovery.

### Mandatory exits

If x4 passes all local-training and cross-seed stability gates, run the complete already-frozen H2/H3 Phase-2 strategic evaluation on the stabilized policies. Apply the existing selection rules. If both pass and strategic evidence remains materially inconclusive, use the already-frozen H2 size/speed tie-break. **R7.5.3 then ends.**

If x4 fails, execute at most the one final contingency remediation above. If that remediation passes, run the complete Phase-2 strategic evaluation and decide H2/H3. If it also fails, **R7.5.3 ends as FAIL/BLOCKED**. Do not create R7.5.3D/E/F or equivalent attempts to avoid the decision.

A FAIL/BLOCKED exit requires an explicit architecture/fallback decision before downstream production training; it never authorizes gate relaxation.

## Downstream finite path

After R7.5.3 closes successfully:

`R7.5.3 representation -> R7.5.4 action/sizing -> R7.5.5 production freeze -> R8 production -> R9 strategic audit -> R10 OpenHoldem runtime -> R11 safe exploitation -> R12 operational homologation -> READY FOR TABLES gate`

H4 remains forbidden until the current representation-admission contract explicitly authorizes it. R7.5.4 strategic revalidation cannot begin before a provisional representation winner exists. R8 official production training cannot begin before its existing prerequisites are satisfied.

## Compute-placement policy: GitHub referee, Ryzen compute

Long CPU-bound work should no longer be forced into GitHub Actions merely because the orchestration exists there.

### GitHub Actions is preferred for

- frozen-contract verification and hash checks;
- build/test/CI and small-to-medium deterministic regressions;
- heldout/reference evaluation and independent certification;
- publishing durable evidence and validating returned artifacts;
- short mechanical reproductions needed to prove parity.

### Ryzen 9 is preferred for

- CFR/traversal jobs expected to take several hours per cell;
- hundreds/thousands of roots per independent seed;
- large chance-coverage experiments and multi-seed ablations;
- physical R8 calibration and official R8 production training;
- any workload that would otherwise require many chained six-hour GitHub jobs solely to evade runner limits.

The older AOF operational architecture already used the same separation of responsibilities: control/publication on the master machine and heavy calculation on Ryzen. SpinCore adopts that principle without inheriting old strategy semantics.

## Local-compute evidence contract

Running on Ryzen must not turn a frozen experiment into a local black box. Before a result is admissible:

1. The exact source commit, experiment contract, seeds, budgets and gates must be committed/frozen first.
2. The local repository must start from that exact commit and a clean tracked worktree.
3. The run must record machine/OS, Python, PyTorch, thread environment, command line, start/end times and return code.
4. Checkpoints/resume must preserve the experiment's frozen RNG/global-root semantics.
5. Result files must have SHA-256 inventory and an immutable run manifest.
6. Returned Ryzen artifacts must be certified against the frozen contract in GitHub before they can close a gate.
7. A local performance difference may choose an execution configuration only where the corresponding calibration precommit permits it; it may never change a strategic acceptance threshold.

`tools/spincore_ryzen_frozen_runner.py` is the generic evidence wrapper for future heavy local executions. It orchestrates and records a frozen command; it does not alter the poker/CFR algorithm being executed.

## Non-negotiable gates

Current Phase-2 hard thresholds remain unchanged:

- Advantage weighted NRMSE <= 0.75 for every required iteration;
- final AveragePolicy weighted mean TV <= 0.12;
- cross-seed mean TV <= 0.15;
- cross-seed p95 TV <= 0.35;
- all required representations/domains/seeds/evaluation rows must be retained;
- dropping a failed domain/seed or post-hoc changing a threshold is forbidden.

This policy limits *how many remediation attempts are allowed*; it does not make PASS easier.
