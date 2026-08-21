# R7.5.3D — V1+ Phase 2A Strategy-memory capacity causal ablation

Date: 2026-08-21
Status: FROZEN_BEFORE_PHASE2A_OUTPUTS
READY FOR TABLES: NO
Production training authorized: NO

## Why this experiment is next

Phase 1 and Phase 1B establish four facts:

1. final H2/H3 policy instability is materially worse in THREE_HANDED than TRUE_HEADS_UP;
2. 3H Strategy streams contain ~3.2M–4.0M samples at x16 while the reservoir remains fixed at 100,000, retaining only ~2.5–3.1%;
3. 3H cross-seed geometry and categorical-history support are already much less aligned than HU, so exact continuous V3 history is not the sole/root cause; and
4. Strategy memory does not drive traversal behavior, so its capacity can be changed passively without changing the upstream Advantage/chance trajectory.

This makes Strategy-memory capacity the cleanest first causal factor to isolate. The experiment is diagnostic/admission work, not a production training run.

## Scope

Use only:

- representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`;
- domain: `THREE_HANDED`;
- training seeds: `1342191342`, `1801739323`;
- frozen `PF0_CONTROL_33_75_AI` action candidate;
- the same x4 chance-coverage schedule used in R7.5.3C: 3 iterations × 4 chunks × 64 roots = 768 roots per seed;
- the same authoritative deck-seed function, scenario cycle, exact-opponent level, Advantage training algorithm, Advantage reservoir capacity 100,000, model architecture, optimizer family, batch size and hard heldout evaluation states.

H2 is used first because Phase 1 showed H3 semantics are not the primary cause. No conclusion about final H2 versus H3 selection is permitted from Phase 2A.

## Passive Strategy-memory arms

For each training seed, one and only one upstream traversal/Advantage trajectory is generated.

Every Strategy sample produced by that trajectory is copied passively into three independent uniform reservoirs:

- `S100K_CONTROL`: capacity 100,000;
- `S400K`: capacity 400,000;
- `S800K`: capacity 800,000.

The existing 100k control must preserve the current uniform-reservoir semantics. Shadow reservoirs must use frozen deterministic seeds and must never feed the behavior policy, Advantage memory, traversal RNG, deck RNG, scenario order, or child selection.

Therefore all three capacity arms see the same generated Strategy stream for a given training seed; only retained Strategy support differs.

## Training budgets

Upstream H2 3H x4 trajectory:

- iterations: 3;
- roots per iteration: 256;
- 4 contiguous 64-root chunks per iteration;
- exact opponent levels: unchanged frozen Phase-2 value;
- Advantage reservoir: 100,000 unchanged;
- Advantage fit: unchanged frozen Phase-2/x4 schedule;
- Advantage ensemble/behavior: unchanged.

Final AveragePolicy fitting for each Strategy-memory arm:

- same H2 architecture;
- policy steps: 16,384;
- batch size: 256;
- learning rate: 0.001;
- no architecture or observation change.

## Two learner readouts

To distinguish memory-distribution effects from final learner randomness, each capacity arm produces two readouts:

### COMMON_LEARNER

For the two independent training-seed memories at the same capacity, use the same frozen policy initialization seed and same frozen batch-RNG seed derivation. Cross-seed TV in this readout therefore isolates differences in retained Strategy target distributions as far as mechanically possible.

### NATIVE_LEARNER

Use the existing training-seed-derived policy initialization/batch semantics. This is the end-to-end readout closest to the current Phase-2 training path.

No arm can be selected from either readout alone.

## Evaluation

Evaluate every final policy on the same two frozen heldout corpora used by R7.5.3C:

- THREE_HANDED / evaluation seed 2029384436;
- THREE_HANDED / evaluation seed 1150634112.

Record for every capacity × learner mode × evaluation seed:

- cross-training-seed mean TV;
- cross-training-seed p95 TV;
- per-state TV vector or sufficient paired summary;
- legal-action-conditioned action-slot disagreement;
- local policy-fit audit metrics;
- Strategy samples seen, retained and retention fraction.

The unchanged hard admission gates remain mean TV <= 0.15 and p95 TV <= 0.35. Passing these gates at x4 is informative but is not itself production admission.

## Causal decision rule

Strategy-memory capacity is considered a materially supported remedy only if all of the following hold:

1. `S800K` improves pooled COMMON_LEARNER mean TV versus `S100K_CONTROL` with paired bootstrap 95% confidence interval strictly above zero for the improvement;
2. both frozen evaluation seeds move in the same direction for mean TV;
3. no frozen evaluation seed degrades by more than 0.01 absolute mean TV;
4. the capacity curve is directionally coherent: `S400K` lies between control and `S800K` or is statistically indistinguishable from one adjacent arm rather than reversing materially; and
5. the NATIVE_LEARNER readout does not contradict the COMMON_LEARNER direction.

A practical continuation threshold is additionally frozen: the pooled `S800K` mean-TV improvement must be at least 0.02 absolute or 10% relative versus `S100K_CONTROL`. Smaller improvements are classified as `CAPACITY_EFFECT_REAL_BUT_INSUFFICIENT` and trigger a sampling/variance-reduction ablation rather than another blind memory escalation.

If `S800K` materially improves stability but does not reach the hard gate, a later experiment may combine the proven capacity remedy with a sampling/variance remedy. Do not jump directly to multi-million capacity.

If capacity has little/no effect, do not increase memory further; prioritize upstream 3H trajectory/sampling stabilization.

## Representation consequence

Phase 2A does not alter SPNNIV3 observation semantics. Exact history compression is deliberately held constant so that Strategy-memory capacity is causally identifiable.

Only after memory/sampling causality is resolved may a compressed-history V1+ arm be tested. Phase 1B already shows that continuous history can amplify fragmentation, so such an arm remains plausible, but it is not the first intervention.

## Strategic-strength safeguard

Improved stability is necessary but never sufficient.

No Phase 2A arm becomes the V1+ winner. Any later full candidate that passes stability must separately demonstrate strategic non-inferiority or improvement against the certified stable V1 control under paired common-deal evaluation and precommitted confidence intervals.

A stable but strategically weaker candidate fails.

## Governance

- no threshold relaxation;
- no seed shopping;
- no dropping THREE_HANDED;
- no H2/H3 winner declared;
- no production training authorization;
- no representation compression mixed into this capacity experiment;
- no action-abstraction change mixed into this capacity experiment;
- no claim that larger memory alone is the final solution before Phase 2A results.