# R7.5.3D — V1+ Phase 2A implementation freeze

Date: 2026-08-21
Status: FROZEN_BEFORE_PHASE2A_OUTPUTS
READY FOR TABLES: NO
Production training authorized: NO

This file refines the already-frozen Phase 2A Strategy-memory-capacity causal ablation with exact implementation details before any Phase 2A training output exists.

## Scientific identity

- representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`
- domain: `THREE_HANDED`
- training seeds: `1342191342`, `1801739323`
- action candidate: `PF0_CONTROL_33_75_AI`
- iterations: 3
- x4 chance coverage: 4 contiguous chunks × 64 roots = 256 roots/iteration
- roots/seed: 768
- exact-opponent levels: unchanged authoritative Phase-2 value
- Advantage reservoir: unchanged 100,000
- Advantage fit / four-member behavior ensemble / deck-seed / scenario-cycle semantics: unchanged from the admitted x4 path
- policy steps: 16,384
- batch size: 256
- learning rate: 0.001
- heldout: the same frozen THREE_HANDED evaluation seeds `2029384436` and `1150634112`, first 1,024 states each
- hard stability gates: mean TV <= 0.15 and p95 TV <= 0.35
- local Advantage NRMSE gate and local AveragePolicy weighted-mean-TV gate remain unchanged.

## Strategy stream capture

The authoritative 100k control reservoir remains the existing `bundle.pol_mem` and therefore retains its historical uniform-reservoir RNG semantics.

During every 64-root chunk, every `StrategySample` passed to the collector is also captured read-only to an append-only per-chunk stream artifact. Capture occurs after no policy decision and does not consume or modify:

- bundle batch RNG;
- deck RNG;
- Advantage RNG;
- control reservoir RNG beyond the authoritative control `add` itself;
- solver state;
- behavior policy.

Chunk stream files are atomic artifacts and are used only after the upstream x4 trajectory is complete.

Before fitting any capacity arm, replaying all captured stream chunks through a fresh 100k reservoir using the historical control seed `training_seed ^ 0x5A5A5A5A` must reproduce the authoritative final control reservoir exactly (`seen`, retained items, and RNG state). Failure invalidates the experiment.

## Passive capacity arms

The exact captured Strategy stream is replayed in order into:

- `S100K_CONTROL`: authoritative final `bundle.pol_mem`, capacity 100,000;
- `S400K`: fresh uniform reservoir, capacity 400,000, seed `training_seed ^ 0x40040040`;
- `S800K`: fresh uniform reservoir, capacity 800,000, seed `training_seed ^ 0x80080080`.

Shadow-reservoir replay happens only after traversal and Advantage training are complete, so shadow capacities cannot alter upstream trajectories.

## Learner modes

### COMMON_LEARNER

For every capacity and both training seeds:

- policy initialization seed: `0x13579BDF`;
- minibatch RNG seed: `0x2468ACE013579BDF`.

This removes final-learner seed differences as far as mechanically possible while leaving the retained Strategy target distribution as the changing factor.

### NATIVE_LEARNER

For each training seed:

- policy initialization is exactly the current H2 policy factory seed `(training_seed ^ 0x5DEECE66D) & 0x7FFFFFFF`;
- every capacity fit starts from an independent clone of the final upstream `bundle.batch_rng` state, i.e. the exact RNG state that the existing final AveragePolicy fit would consume for the control memory.

Thus capacity arms within the same training seed share the same native learner starting RNG state; capacities do not consume each other's RNG.

## Fit audit

Each fitted policy is audited on its own retained Strategy memory with the frozen audit size 2,048 and the unchanged local policy weighted-mean-TV maximum 0.12.

A capacity-causality conclusion is invalid if any of the two upstream seeds fails an Advantage NRMSE gate or if the required COMMON_LEARNER fits used in the causal comparison fail their local policy-fit gate.

## Cross-seed evaluation

For each `(capacity, learner_mode, evaluation_seed)`, evaluate the two training-seed policies on exactly the same 1,024 frozen heldout states and record the full aligned per-state TV vector plus mean/p50/p95/max and hard-gate boolean.

Pooled mean TV is the equal-weight average of the two evaluation-seed means.

## Paired bootstrap

Capacity improvements are defined statewise as:

`improvement(A -> B) = TV_A - TV_B`

Positive values favor the larger capacity.

Use the repository's deterministic equal-group stratified bootstrap, with the two evaluation seeds as equal-weight groups, 2,000 replicates and 95% confidence. Seed keys include `R7.5.3D`, `PHASE2A`, learner mode, and the capacity contrast.

Required contrasts for COMMON_LEARNER:

- S100K_CONTROL -> S400K
- S400K -> S800K
- S100K_CONTROL -> S800K

Also compute S100K_CONTROL -> S800K for NATIVE_LEARNER.

## Exact causal classification

First require all upstream Advantage gates and all COMMON_LEARNER local policy-fit gates to pass. Otherwise status is `PHASE2A_INVALID_LOCAL_GATES`.

Define `base = pooled COMMON_LEARNER mean TV for S100K_CONTROL` and `large = pooled COMMON_LEARNER mean TV for S800K`.

A statistically supported capacity effect requires all of:

1. COMMON S100K->S800K paired-bootstrap `ci_low > 0`;
2. both individual heldout evaluation seeds have `mean_tv(S800K) <= mean_tv(S100K)`;
3. no individual heldout seed degrades by more than 0.01 mean TV (guard retained explicitly);
4. capacity curve coherence:
   - ideal case: pooled `S100K >= S400K >= S800K`;
   - otherwise any adjacent reversal must be <= 0.005 absolute **and** the corresponding paired-bootstrap CI must include zero;
5. NATIVE S100K->S800K pooled improvement is non-negative.

If these are not all true, status is `CAPACITY_EFFECT_NOT_SUPPORTED`.

If statistically supported, practical materiality is satisfied when either:

- `base - large >= 0.02`, or
- `(base - large) / base >= 0.10`.

If statistical support passes but materiality does not, status is `CAPACITY_EFFECT_REAL_BUT_INSUFFICIENT`.

If both statistical support and materiality pass, status is `CAPACITY_EFFECT_MATERIALLY_SUPPORTED`.

Reaching the hard x4 stability gate is reported separately and is not required to prove a capacity effect; x4 is diagnostic, not production admission.

## Governance after Phase 2A

- No H2/H3 winner is selected by Phase 2A.
- No capacity arm becomes a production candidate from stability alone.
- If capacity is materially supported, the next causal step may combine the proven memory remedy with an upstream 3H sampling/variance remedy if needed.
- If capacity is real but insufficient, prioritize upstream sampling/variance reduction rather than another blind capacity escalation.
- If capacity is not supported, stop memory escalation and prioritize upstream trajectory stabilization.
- Representation/history compression remains a later independent ablation if memory/sampling repair does not fully solve stability.
- Any eventual stable V1+ candidate must still pass a separate strategic-strength gate against the certified stable V1 control.