# R7.5 Architecture Reset — Phase2C1 Exact Range-Reach Solver Prototype

Date: 2026-08-24

## Trigger and finite scope

Phase2C0 passed every frozen structural factorization gate. Its returned result SHA256 is
`55e83be4fd8776e0fcdb63e7d4400ed05aff8c48213898ad8f1abe3713a35876`, with status
`STRUCTURAL_REACH_FACTORIZATION_FEASIBLE`. The largest observed factorization error was
`5.551115123125783e-17`, filler/board-independence error was exactly zero, all 16 tasks had positive finite
normalizers, and every task stayed within the 4,901 table-evaluation budget.

The estimator-repair line remains closed. Phase2C1 is not another Monte-Carlo target estimator. It prototypes an
explicit range/reach state transition kernel for the public preflop history.

This architecture-reset branch remains finite:

* Phase2C1 PASS permits exactly one separately precommitted bounded range/reach target-kernel causal pilot.
* Phase2C1 FAIL selects the certified stable V1 fallback and closes the V1+ architecture reset.
* No B15/B16 estimator repair may be reopened.

## Scientific question

Can the exact opponent private-card reach represented by Phase2C0 be propagated **incrementally through the public
action path** as explicit per-seat reach tables, while reproducing the direct full-history likelihood and Phase2C0
joint posterior invariants to numerical precision?

A usable structural solver must not recompute an opaque posterior estimator from scratch at every continuation.
It must be able to carry a deterministic reach state forward through public actions.

## Frozen sources

* domain: `THREE_HANDED`
* representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`
* source behavior: exact final Phase2B13 `IID64_MEAN_CANDIDATE` four-member Advantage ensemble, separately for
  training seeds `1342191342` and `1801739323`
* behavior includes the frozen 25% uniform floor on preflop continuations only
* source Phase2C0 result SHA256:
  `55e83be4fd8776e0fcdb63e7d4400ed05aff8c48213898ad8f1abe3713a35876`
* source Phase2C0 status: `STRUCTURAL_REACH_FACTORIZATION_FEASIBLE`
* same 8 deterministic Phase2C0 anchors
* same audited Windows canonical suit-isomorphic heldout reconstruction
* two behavior seeds -> 16 prototype tasks

## Exact ordered-hand state

No new hand-order symmetry is assumed in Phase2C1. For the current actor's two fixed cards, each opponent seat keeps
an explicit vector over the same 2,450 ordered two-card candidates used by Phase2C0. The two opponent vectors are
coupled only through exact card removal when a joint posterior quantity is computed.

The core structural state is therefore two 2,450-entry reach vectors plus the public state/history. Two float64
vectors require 39,200 bytes of raw reach storage; this is the frozen memory reference for the prototype.

## Incremental propagation algorithm

For each anchor/behavior task:

1. reconstruct the exact target actor's private-card orbit and canonical explicit deal using the audited Phase2B15
   Windows correction;
2. initialize each opponent's 2,450 ordered-hand reach vector to one and the current actor's known-hand reach scalar
   to one;
3. replay the stored public preflop action path event by event;
4. when an opponent seat acts, evaluate the frozen behavior probability of the exact observed action for each of that
   seat's 2,450 candidate hands at that public prefix and multiply that seat's reach vector elementwise;
5. when the final current actor previously acted in the path, multiply only the known-hand scalar by that frozen
   action probability;
6. advance the authoritative public solver state with the exact observed universal action;
7. at the target continuation, require byte-identical SPNNIV3, actor, active mask, and legal slots.

No hidden hand is sampled. No posterior normalization is required during updates. No future board is sampled. The
reach vectors are deterministic functions of frozen behavior plus public history.

## Direct-likelihood parity checks

For every prototype task and each opponent seat, 128 deterministic candidate hands are checked against the original
Phase2C0 direct full-history seat likelihood. The incremental reach entry and direct likelihood must agree to
absolute tolerance `1e-12`.

The known current-actor reach scalar is checked against the direct current-actor likelihood to the same tolerance.

## Joint posterior parity against Phase2C0

Using the final incremental opponent reach vectors and the identical collision mask, recompute:

* joint posterior normalizer;
* effective joint support;
* positive joint assignments.

For the matching Phase2C0 row:

* normalizer relative error must be `<=1e-12`;
* effective-support relative error must be `<=1e-12`;
* positive-joint-assignment count must match exactly.

This is the primary proof that the structural transition state is mathematically equivalent to the already-passed
factorized posterior, not merely correlated with it.

## Frozen feasibility gates

Phase2C1 passes only if all hold:

1. all 16 tasks complete;
2. final target SPNNIV3/actor/active-mask/legal identity is exact for every task;
3. maximum incremental-vs-direct opponent-hand reach error `<=1e-12`;
4. maximum current-actor scalar parity error `<=1e-12`;
5. maximum Phase2C0 joint-normalizer relative error `<=1e-12`;
6. maximum Phase2C0 effective-support relative error `<=1e-12`;
7. every positive-joint-assignment count matches Phase2C0 exactly;
8. raw two-opponent reach-vector storage is `<=39,200` bytes;
9. per-task structural policy-evaluation count is `<=4,902`.

Wall-clock time is recorded but is not a strategic gate because it is machine-dependent. No result may relax the
frozen exactness or evaluation-count gates after inspection.

## Decision

PASS -> `PRECOMMIT_SINGLE_BOUNDED_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT`.

FAIL -> `SELECT_CERTIFIED_STABLE_V1_FALLBACK_AND_CLOSE_V1PLUS_ARCHITECTURE_RESET`.

A PASS does not authorize production training, full-x4 confirmation, architecture winner selection, or table use.
The next causal pilot, if permitted, must be separately frozen and is the only remaining structural pilot before a
fallback decision or hard-stability confirmation route.

## Prohibitions

No target averaging; no rejection/importance sampling; no K sweep; no K128/K256; no clipping/tempering; no MCMC or
SIR; no seed shopping; no anchor/scenario dropping; no threshold relaxation; no hidden-card Monte Carlo inside the
reach update; no new hole-order symmetry assumption; no training from Phase2C1; no H2/H3 R7.5.3 readmission; no
production training; no ready-for-tables claim.
