# R7.5 Architecture Reset — Phase2B15 Posterior-Weighted Preflop-Continuation Chance Screen

Date: 2026-08-24

## Trigger

Phase2B14 reproduced the Phase2B13 heldout metrics exactly and classified the remaining COMMON instability as
`PREFLOP_CONTINUATION_RESIDUAL_DOMINANT_AFTER_ROOT_IID64`.

The Phase2B13 IID64 root intervention produced a consistent root improvement in both heldouts, but its overall
effect was below the precommitted materiality gate.  Phase2B14 therefore routes to a continuation-specific
conditional-chance diagnostic rather than a full-x4 Phase2B13 confirmation.

## Scientific question

At a preflop continuation infoset, hidden opponent cards are not distributed as the unconditional card prior:
the already-observed actions contain information about those cards.  Does conditioning candidate hidden-card
deals on the observed action path, using the frozen learned behavior policy as the likelihood model, produce a
material and more stable Advantage target estimator at equal K64 compute?

## Frozen scope

* representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`
* domain: `THREE_HANDED`
* source behavior: exact final Phase2B13 `IID64_MEAN_CANDIDATE` Advantage ensemble, separately for training seeds
  `1342191342` and `1801739323`
* behavior policy includes the already-accepted 25% uniform floor only on preflop continuations
* target iteration: 3
* exact opponent levels: 2
* regions only:
  * `PREFLOP_CONTINUATION_1`
  * `PREFLOP_CONTINUATION_2PLUS`
* root and postflop are excluded from this screen
* 16 deterministic scenario-round-robin anchors per region per heldout evaluation seed
* heldout evaluation seeds: `2029384436`, `1150634112`
* total anchors: 64
* 2 independent chance blocks per anchor
* K=64 conditional-IID deals per block
* both source behavior seeds evaluate the same chance proposals
* traversal RNG is fixed within each anchor across both blocks and all proposals
* total target traversals: 64 anchors × 2 behavior seeds × 2 blocks × 64 = 16,384

## Equal-compute estimators

Every block computes the same 64 raw Advantage targets once.

`UNWEIGHTED_IID64` is their arithmetic mean.

`POSTERIOR_WEIGHTED_IID64` is their self-normalized importance mean.  The proposal is uniform opponent private
cards plus uniform future board conditional on the current actor's two hole cards.  The importance weight is
proportional to the product of the frozen behavior-policy probabilities of the exact already-observed preflop
actions when that candidate private-card deal is replayed.

Future board cards do not enter the likelihood because no board card is visible at a preflop continuation.
They remain IID from the card prior after removing the sampled private cards.

No arbitrary probability floor may be added to the likelihood.  If the exact frozen behavior assigns zero
support to every proposal for an anchor/block, the screen is invalid/support-degenerate and must not be repaired
post hoc.

## Identity guards

For every alternative deal, replaying the frozen public action path must end at exactly the same:

* SPNNIV3 bytes for the current actor,
* actor,
* active mask,
* legal universal slots.

Any mismatch aborts.

The source Phase2B13 result SHA is
`6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80`.

The source Phase2B14 result SHA is
`7cd1886596d345abdcdef479775498eddf7e014205de86e44afb5bb0ea291f86`.

## Primary readouts

For independent block A vs block B, compare unweighted and posterior-weighted estimates using:

* regret-matching policy TV,
* legal-target mean absolute difference,
* positive-sign disagreement,
* dominant legal-action mismatch,
* fraction with TV >= 0.35.

Also record the within-block policy shift caused by posterior conditioning, effective sample size (ESS), maximum
normalized importance weight, zero-weight count, and weight-span diagnostics.

## Frozen gates

Posterior weighting is considered supported only if all of the following hold:

1. local validity and complete task coverage;
2. pooled posterior shift mean TV >= 0.03;
3. pooled block-to-block TV improves by >= 0.03 absolute OR >= 15% relative;
4. sign disagreement improves by >= 0.02 absolute OR >= 10% relative;
5. TV>=0.35 tail rate improves by >= 10% relative;
6. both source behavior seeds have positive TV improvement;
7. neither continuation region degrades by more than 0.01 absolute mean TV;
8. posterior weights remain usable: median ESS >= 16, p10 ESS >= 8, and p95 maximum normalized weight <= 0.35.

## Decision hierarchy

* invalid -> `PHASE2B15_INVALID_STOP_AUDIT`
* poor weight health -> `POSTERIOR_IMPORTANCE_WEIGHT_DEGENERACY`
* posterior shift below 0.03 -> `POSTERIOR_CONDITIONING_EFFECT_SMALL`
* all support gates pass -> `POSTERIOR_WEIGHTED_CONTINUATION_ESTIMATOR_SUPPORTED`
* otherwise -> `POSTERIOR_WEIGHTING_MATERIAL_BUT_STABILITY_NOT_SUPPORTED`

A supported screen permits only a separately precommitted small training pilot.  This screen itself authorizes
no training.

## Prohibitions

No K sweep, no K128/K256 escalation, no threshold relaxation, no seed shopping, no scenario dropping, no
higher behavior floor, no Huber retry, no lagged-policy retry, no arbitrary post-history uniform hidden-card
resampling, no full-x4 confirmation, no architecture winner, no production training, and no ready-for-tables
claim.
