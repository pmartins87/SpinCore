# R7.5 Architecture Reset — Phase2C0 Structural Reach-Factorization Feasibility Audit

Date: 2026-08-24

## Trigger and finite closure

Phase2B16 is locally valid but failed the frozen direct-posterior mean-TV ceiling: exact rejection posterior sampling reached pooled mean TV 0.2517340306655365 against the <=0.24 gate. Phase2B15/B16 therefore close the estimator-repair path. No K128/K256 escalation, importance-weight tuning, clipping, tempering, MCMC, SIR, or additional posterior-sampler variant is authorized.

The architecture-reset decision now has only two branches: (A) one bounded structural reach-support route, or (B) certified stable V1 fallback. Because Phase2B16 improved materially over SNIS on every other frozen gate and missed only the mean-TV ceiling, this precommit chooses one bounded structural feasibility audit before falling back to V1.

Phase2C0 is not another target estimator and performs no Advantage target traversal. It asks whether preflop action-history reach can be represented exactly and cheaply enough to justify a solver-level range/reach prototype.

## Scientific question

For a fixed current actor and public preflop history, does the frozen behavior-path likelihood factor exactly into:

`constant(current-actor observed actions) * product(seat-specific likelihood factors for each opponent's private ordered two-card hand)`

subject only to card-removal coupling between opponent hands?

If yes, the exact joint posterior over opponent private cards can be constructed from two seat-specific 2,450-hand tables plus a card-exclusion mask, rather than by stochastic rejection/importance weighting. This would support a genuinely structural range/reach solver path.

## Frozen inputs

* domain: THREE_HANDED
* representation: H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL
* source behavior: exact final Phase2B13 IID64_MEAN_CANDIDATE checkpoints, separately for seeds 1342191342 and 1801739323
* source Phase2B16 result SHA256: `3b5e71c3cc92ed530589877f6790333b1f94b579bb39e7c687082787693d958c`
* source Phase2B16 status: `EXACT_POSTERIOR_STILL_TOO_UNSTABLE_CLOSE_ESTIMATOR_REPAIR_PATH`
* heldout reconstruction: inherited audited Phase2B15 Windows canonical suit-isomorphic explicit-deal reconstruction
* anchors: deterministic first 2 state-index anchors from each (evaluation seed × continuation region), total 8 anchors
* behavior seeds: 2, total 16 independent anchor/behavior tasks
* continuation regions: PREFLOP_CONTINUATION_1 and PREFLOP_CONTINUATION_2PLUS

## Ordered private-hand state space

The current actor's two cards are fixed by the authoritative SPNNIV3 observation. Each opponent hand is an ordered pair from the remaining 50 cards, so each seat has 50×49 = 2,450 ordered candidates. The two opponent seats are coupled only by card exclusion. The exact joint prior support contains 50×49×48×47 = 5,527,200 ordered assignments.

Ordered pairs are deliberate because SPNNIV3 retains the two private-card slots separately; no unproven hole-order symmetry is imposed.

## Exact factorization checks

For each of the 16 tasks:

1. compute the current-actor action-path likelihood component once;
2. enumerate all 2,450 ordered private hands for opponent seat A and compute only probabilities of observed actions taken by seat A;
3. enumerate all 2,450 ordered private hands for opponent seat B analogously;
4. verify on a deterministic sample of 128 valid joint opponent assignments that full frozen path likelihood equals the product of the three seat components to absolute tolerance 1e-12;
5. verify on 32 deterministic candidate hands per opponent that the seat-specific component is invariant to alternative filler cards and future board to absolute tolerance 1e-12;
6. compute exact collision-masked joint posterior normalizer and effective support from the two seat tables without target traversal.

Every explicit replay must end at the exact authoritative target SPNNIV3, actor, active mask, and legal slots.

## Frozen feasibility gates

Structural reach factorization is feasible only if all hold:

* all 16 tasks complete;
* maximum factorization absolute error <= 1e-12;
* maximum filler/board-independence absolute error <= 1e-12;
* every task has finite positive posterior normalizer;
* no task requires more than 4,901 seat-policy table evaluations (1 actor constant + 2×2,450 opponent ordered hands; validation replays are recorded separately);
* no source/result/heldout/checkpoint identity drift.

No stability or strategic winner is claimed from this audit.

## Decision

PASS -> `PRECOMMIT_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE`.

FAIL -> `SELECT_CERTIFIED_STABLE_V1_FALLBACK_AND_CLOSE_V1PLUS_ARCHITECTURE_RESET`.

Phase2C0 itself authorizes no training, no x4 confirmation, no architecture winner, no production training, and no ready-for-tables claim.

## Prohibitions

No estimator repair reopening; no K escalation; no posterior approximation tuning; no threshold relaxation; no seed shopping; no anchor dropping; no hole-order symmetry assumption; no training from Phase2C0; no H2/H3 R7.5.3 readmission.
