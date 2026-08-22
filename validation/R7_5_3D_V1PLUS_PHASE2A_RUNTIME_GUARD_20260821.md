# R7.5.3D — Phase 2A mechanical runtime guard

Date: 2026-08-21
Status: RECORDED_BEFORE_PHASE2A_OUTPUTS
READY FOR TABLES: NO
Production training authorized: NO

The first implementation audit of Phase 2A found two mechanical issues before any Phase 2A training output existed.

## Zero-root x4 fit reporting

Phase 2A intentionally follows the already-admitted x4 execution pattern: collect four contiguous 64-root chunks and then perform one unchanged Advantage fit with a temporary `roots_per_iteration=0` configuration. The shared `run_one_phase2_v3_iteration` still divides temporary reporting rates by `roots_added`, which is zero on this fit-only call.

Phase 2A therefore uses the existing audited helper `_fit_only_iteration` from `tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py`. That helper executes the same Advantage reset, primary fit, side-member fits, audits, behavior update and persistent stage-state update. Its only difference is that temporary root-normalized reporting placeholders are `0.0`; Phase 2A immediately replaces those fields with the actual four-chunk 256-root totals.

This is the same mechanical correction already used by the successful x4 lineage and does not change training semantics.

## Authoritative policy audit seed

The first base implementation gave each capacity arm a capacity-specific local policy-audit sample seed. That would introduce avoidable audit-sampling noise between capacity arms.

Before any Phase 2A output, the runtime guard instead pins every capacity/learner arm for a given training seed to the authoritative Phase-2 final-policy audit seed:

`training_seed ^ 0x71A5BEEF`

The audit size remains 2,048 and the hard local weighted-mean-TV gate remains 0.12.

## Guarded child processes

The parent Phase 2A runner launches the two independent training-seed cells in child Python processes. The runtime guard explicitly launches those children through the same guarded entrypoint, ensuring the zero-root correction and authoritative audit seed cannot be bypassed in subprocesses.

## Power-loss recovery

The atomic resume checkpoint is authoritative. If power is lost after the Strategy stream and resume checkpoint are committed but before the small last-stage JSON marker is written, the guard permits exactly that final marker to be reconstructed from metadata already stored inside the checkpoint. Earlier missing stage artifacts still fail closed.

## Scientific contract unchanged

The guard changes none of:

- H2 representation;
- THREE_HANDED domain;
- training/evaluation seeds;
- PF0 control action candidate;
- x4 root/deck/scenario order;
- Advantage memory, fit or ensemble semantics;
- Strategy capacity arms;
- learner budgets;
- heldout states;
- hard local/cross-seed gates;
- Phase 2A causal classification rules.

No Phase 2A result existed when this correction was recorded.