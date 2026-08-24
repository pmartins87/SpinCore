# R7.5 Architecture Reset — Phase2C2 Range-Reach Target-Kernel Causal Pilot

Date: 2026-08-24

## Trigger and finite scope

Phase2C1 passed every frozen exact range/reach transition gate. Its returned result SHA256 is `62ad2352c807a3b046bc84df2cbdf66cc8e0217e3422d01f2bcd9ddeafe7875b`, with status `EXACT_RANGE_REACH_TRANSITION_PROTOTYPE_FEASIBLE`. The maximum incremental-vs-direct reach error was `5.551115123125783e-17`, the maximum joint-normalizer relative error versus Phase2C0 was `2.1541587260971804e-16`, the maximum effective-support relative error was `3.8543787362290213e-16`, all positive-joint counts matched exactly, and raw two-opponent reach storage stayed at 39,200 bytes.

The B15/B16 estimator-repair line remains closed. Phase2C2 is the **single bounded structural causal pilot** permitted by Phase2C1. It is not a K sweep and does not reopen importance weighting, rejection sampling, clipping, tempering, MCMC, SIR, or generic posterior-estimator tuning.

This architecture-reset branch remains finite:

* Phase2C2 PASS -> permit one full x4 hard-stability confirmation of the same frozen structural candidate.
* Phase2C2 FAIL -> select the certified stable V1 fallback and close the V1+ architecture reset.
* A Phase2C2 PASS alone never selects an architecture winner and never authorizes production training.

## Scientific question

Does using the exact Phase2C1 public-action reach state to construct a low-discrepancy posterior private-card target kernel for a guaranteed preflop continuation sample produce a **causal reduction in final AveragePolicy cross-seed instability**, when both arms spend the same auxiliary target compute and keep the already-supported root IID64 correction and 25% preflop-continuation behavior floor?

## Frozen source behavior

* domain: `THREE_HANDED`
* representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`
* source/bootstrap behavior: exact final Phase2B13 `IID64_MEAN_CANDIDATE` four-member Advantage ensemble, separately for training seeds `1342191342` and `1801739323`
* source Phase2C1 result SHA256: `62ad2352c807a3b046bc84df2cbdf66cc8e0217e3422d01f2bcd9ddeafe7875b`
* source Phase2C1 status: `EXACT_RANGE_REACH_TRANSITION_PROTOTYPE_FEASIBLE`
* source Phase2B13 result SHA256: `6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80`
* source Phase2B14 result SHA256: `7cd1886596d345abdcdef479775498eddf7e014205de86e44afb5bb0ea291f86`
* frozen 25% uniform behavior floor on preflop continuation only; root/postflop/heldout inference floor remain zero

## Pilot trajectory

The pilot intentionally starts with fresh Advantage/Strategy reservoirs and optimizer state but bootstraps iteration-1 behavior from the exact final Phase2B13 four-member ensemble. This avoids diluting a small structural intervention inside the already-large B13 reservoirs while still testing causal propagation through behavior -> new traversal -> AveragePolicy.

Frozen budget per arm/seed:

* 2 CFR iterations;
* 64 logical roots per iteration;
* 128 logical roots total;
* exact existing Phase-2 opponent action expansion level remains 2;
* existing Advantage/Policy fit sizes, network architecture, reservoir capacities, learning rate, legal-action semantics, and heldout evaluation are unchanged.

## Equal-compute arms

Two arms are trained from the same source behavior and deterministic root schedule:

1. `RANGE1_EQUAL_COMPUTE_CONTROL`
2. `RANGE64_MEAN_CANDIDATE`

For **both arms and every logical root**:

* compute the same existing B13 root IID64 target set and insert its arithmetic mean at the initial-root actor sample;
* deterministically select one guaranteed nonterminal **two-action preflop continuation path** using the frozen action preference order `CHECK_CALL -> aggressive slots ascending -> FOLD` and exact legality;
* construct the exact Phase2C1 opponent reach tables at that continuation from the current frozen behavior;
* create exactly 64 posterior private-card joint assignments using the same randomized 8x8 stratified range kernel;
* sample one independent future board for each joint assignment;
* use one fixed traversal RNG across the 64 target traversals for that continuation so the auxiliary average integrates private/public chance rather than traversal RNG;
* perform all 64 auxiliary continuation target traversals in both arms.

The control inserts the **first** of the 64 structurally generated continuation targets. The candidate inserts the arithmetic mean of the **same 64** targets. Therefore the causal comparison is equal-compute.

Exactly one root sample and exactly one depth-2 preflop continuation sample must be replaced per logical root. Missing, duplicate, or identity-drifting replacements abort the pilot.

## Structural private-card kernel

At the selected continuation, the current actor's private cards are fixed. Two opponent reach vectors over the exact 2,450 ordered hands are propagated through the two observed public actions using the Phase2C1 update rule.

Let `wA(hA)` and `wB(hB)` be those reach weights. The legal joint posterior is proportional to

`wA(hA) * wB(hB) * I(no card collision)`.

The 64 private assignments are produced without rejection or self-normalized importance weights:

1. compute the exact collision-adjusted marginal over seat A;
2. divide its CDF into 8 equal-probability strata and draw one deterministic seeded uniform inside each stratum;
3. for each selected seat-A hand, compute the exact conditional CDF of seat B under card removal;
4. draw 8 deterministic seeded stratified seat-B samples;
5. the 8x8 product yields exactly 64 legal posterior assignments.

The random-within-stratum seeds are frozen namespaces independent of arm. Both arms receive byte-identical assignment/board/traversal proposals.

## Local validity gates

Every arm/seed must satisfy:

* exactly 2 iterations and 128 logical roots;
* exactly 128 root IID64 replacements;
* exactly 128 structural continuation replacements;
* exactly `128 * 64 = 8,192` root auxiliary target traversals;
* exactly `128 * 64 = 8,192` continuation auxiliary target traversals;
* every selected continuation remains preflop, nonterminal, has at least two non-forced preflop events, and is visited exactly once as a matching Advantage sample;
* all required Advantage ensemble weighted NRMSE values `<=0.75`;
* all final AveragePolicy fit weighted mean TV values `<=0.12`;
* no source/checkpoint/heldout/result identity drift.

Any local-validity failure invalidates the causal comparison and forces audit; it is not a candidate FAIL.

## Frozen causal PASS gates

The primary learner is `COMMON_LEARNER`. All of the following are required:

1. pooled candidate cross-seed mean TV improves over equal-compute control by at least `0.020` absolute **or** `10%` relative;
2. equal-group stratified bootstrap 95% CI for `control TV - candidate TV` has lower bound `>0`;
3. candidate mean TV improves on both frozen heldout evaluation seeds;
4. candidate p95 TV may not worsen by more than `0.020` on either heldout;
5. preflop continuation-2plus mean TV improves in both heldouts;
6. root mean TV may not worsen by more than `0.020` on either heldout;
7. `NATIVE_LEARNER` pooled mean TV must not worsen, and neither heldout may worsen by more than `0.010` mean TV.

Hard stability (`mean <=0.15` and `p95 <=0.35` on both COMMON heldouts) is recorded but is **not required** for this small pilot. If the causal gates pass, the only next route is a full x4 confirmation of the identical structural intervention.

## Decision

* local validity failure -> `STOP_AND_AUDIT_PHASE2C2_LOCAL_VALIDITY`
* causal PASS -> `PRECOMMIT_FULL_X4_STRUCTURAL_RANGE_REACH_CONFIRMATION`
* causal FAIL -> `SELECT_CERTIFIED_STABLE_V1_FALLBACK_AND_CLOSE_V1PLUS_ARCHITECTURE_RESET`

Phase2C2 itself never authorizes architecture winner selection, production training, R8, or table use.

## Prohibitions

No K sweep; no K128/K256; no alternate stratification side after outputs; no posterior clipping/tempering/importance/rejection/MCMC/SIR; no extra continuation target count after outputs; no threshold relaxation; no seed shopping; no heldout/scenario dropping; no change to the 25% continuation floor; no production training; no ready-for-tables claim.
