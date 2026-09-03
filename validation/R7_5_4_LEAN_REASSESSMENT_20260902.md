# R7.5.4 lean reassessment — dense 3H recovery

Date: 2026-09-02

## Finding

The historical R7.5.4 action-abstraction precommit explicitly marks `PF_DENSE_REFERENCE` as `eligible_to_win: false`. Its role is `REFEREE_UPPER_COVERAGE_REFERENCE`; the production-eligible postflop candidates are PF0 through PF4.

The original referee protocol makes the dense model expensive because it uses dense-generated held-out states and dense continuation, repeated over every training seed and paired evaluation seed, plus large crossplay and bootstrap counts. This is useful for publication-grade controlled comparison but it is not itself a production candidate.

The historical Actions run already contains final artifacts for the eligible postflop candidates in both HU and 3H across the frozen training seeds, while the missing final cells are the dense-reference 3H cells now being mechanically recovered.

## Decision-relevant interpretation

Completing three full five-iteration dense 3H models can only improve the final agent indirectly, by enabling the originally frozen referee protocol. It cannot directly become the selected production action abstraction under the historical candidate eligibility rule.

Therefore weeks of dense-only recovery are not justified by matrix completeness alone.

## Recommended lean replacement

1. Stop dense 3H recovery at the next safe interruption point and preserve all completed checkpoints.
2. Analyze the already completed eligible PF0-PF4 HU/3H final models.
3. Run a direct adaptive comparison among eligible candidates using common deals / common random numbers and exact action identity.
4. Use small-to-medium crossplay and exact-action omission screens first. Eliminate clearly dominated candidates early.
5. Increase hands / held-out states only for candidates whose practical EV difference remains close enough that more evidence can change the winner.
6. Use the existing dense HU evidence as diagnostic context, not as a mandatory production gate.
7. Resume dense 3H training only if the eligible-candidate comparison remains materially ambiguous and a dense continuation/referee is likely to resolve that ambiguity.

## Quality argument

This change does not intentionally reduce playing quality. It redirects compute from a non-winning referee model to comparisons among models that can actually be selected and then to production-scale training of the winner. Exact poker semantics, legal actions, HU/3H coverage, common-random-number comparison, held-out play, and strategic EV evaluation remain protected.

No claim is made that dense 3H can never provide useful information. The claim is narrower: it should be computed only when that information is decision-relevant, rather than because a historical 36/36 matrix required it.
