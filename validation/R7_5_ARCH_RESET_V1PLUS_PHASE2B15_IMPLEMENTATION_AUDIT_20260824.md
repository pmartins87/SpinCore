# R7.5 Architecture Reset — Phase2B15 Implementation Audit

Date: 2026-08-24

## Scope

Audit of the frozen read-only `Phase2B15 Posterior-Weighted Preflop-Continuation Chance Screen` implementation
before any Ryzen output exists.

## Source identity

The tool requires the exact Phase2B13 result SHA
`6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80`
and the exact Phase2B14 result SHA
`7cd1886596d345abdcdef479775498eddf7e014205de86e44afb5bb0ea291f86`.

It also loads only the final Phase2B13 `IID64_MEAN_CANDIDATE` resume checkpoints for the two frozen training
seeds.  Checkpoint schema, execution SHA, H2/3H identity, action candidate, iteration=3, global_root=384,
stage_index=6, arm, K=64, and four-member final behavior ensemble are validated before use.

## Posterior semantics

For each preflop continuation anchor, the current actor's two private cards are fixed.  Opponent private cards
and all five future board cards are proposed IID from the card prior conditional on those two cards.

The exact observed universal-action path is replayed under each proposed private-card deal.  Its importance
likelihood is the product of the exact frozen behavior probabilities assigned to those already-observed actions.

This is the relevant Bayesian correction because opponent actions carry information about opponent private cards.
Future board cards are not yet observed at these anchors and therefore remain independent of the path likelihood.

The implementation does not add a numerical action-probability floor.  Exact zero support produces zero
importance weight.  All-zero support for an anchor/block is a hard failure rather than being silently repaired.

## Infoset leakage guard

After replay under every alternative deal, the target state must reproduce the exact heldout current-actor
SPNNIV3 bytes, actor, active mask, and legal universal slots.  This ensures that hidden-card resampling changes
only hidden chance, not the observable target infoset.

## Equal-compute pairing

Each block computes 64 raw target traversals once.  The arithmetic K64 estimator and posterior-weighted K64
estimator are then formed from the same 64 targets.  Therefore estimator differences cannot be attributed to a
different traversal budget.

Both source behavior seeds use the same proposal deals.  Traversal RNG is fixed within each anchor across all
proposals and both independent blocks.

## Workload and parallelism

Frozen workload:

* 64 balanced anchors;
* 2 behavior seeds;
* 2 blocks;
* 64 target traversals per block;
* 16,384 total continuation target traversals.

The launcher permits up to 30 one-thread worker processes.  No Torch training is performed.  Completed
anchor/block partial JSON files are persisted and validated on resume.

## Read-only property

The Phase2B15 code performs no:

* optimizer step;
* Advantage fit;
* AveragePolicy fit;
* reservoir add or mutation;
* production training;
* full-x4 confirmation.

Existing Phase2B13 checkpoints and heldouts are read-only inputs.

## Synthetic tests

The deterministic test suite covers:

* equal log weights reproducing the arithmetic mean and ESS=64;
* a one-sample concentrated posterior yielding ESS=1;
* regret-matching TV/sign/dominant-action helpers;
* deterministic chance/traversal seed construction;
* PASS classification and weight-degeneracy classification;
* governance flags remaining false.

The Windows launcher additionally runs `py_compile`, exact prerequisite/hash checks, frozen H2/3H contract
validation, candidate checkpoint identity checks, an x64 solver rebuild, ABI/explicit-deal checks, and the existing
explicit-deal round-trip tests before scientific work starts.

## Known interpretation boundary

This is a feasibility/stability screen for self-normalized posterior weighting under a frozen learned behavior
likelihood.  It is not a proof that this finite-K estimator is unbiased, and it is not a production posterior
sampler.  A PASS only permits a separately precommitted small causal training pilot.

## Audit conclusion

The implementation matches the frozen Phase2B15 question and preserves the no-scale-up governance from
Phase2B13/Phase2B14.  No Ryzen PASS is claimed here; runtime and scientific result remain pending.
