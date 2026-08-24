# R7.5 Architecture Reset — Phase2B16 Exact-Rejection Posterior Continuation Screen

Date: 2026-08-24

## Trigger

Phase2B15 was locally valid and showed a material posterior shift with healthy importance weights, but self-normalized posterior weighting materially worsened block-to-block stability in both behavior seeds and both continuation regions. The failure was not weight degeneracy. Before abandoning posterior-conditioned continuation targets, run one final estimator-level test that removes self-normalized-importance-estimator noise entirely.

## Scientific question

At the same preflop continuation infosets and under the same frozen Phase2B13 behavior policies, does drawing opponent hidden cards **directly from the exact action-history posterior** by rejection sampling materially reduce the instability seen with Phase2B15 self-normalized importance weighting?

For a prior hidden-card proposal h with likelihood L(h)=product of the frozen behavior probabilities of the already-observed preflop actions, L(h) is in [0,1]. Sampling h from the prior and accepting with probability L(h) gives an exact draw from p(h | observed action path), up to Monte Carlo randomness, without normalized importance weights.

Future board cards are drawn independently from the remaining deck only after a private-card proposal is accepted. No board card is visible in the tested continuation states, so the action-history likelihood is independent of the future board.

## Frozen sources and scope

* representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`
* domain: `THREE_HANDED`
* source behavior: exact final Phase2B13 `IID64_MEAN_CANDIDATE` four-member Advantage ensemble, separately for training seeds `1342191342` and `1801739323`
* behavior includes the frozen 25% uniform floor only on preflop continuations
* exact same 64 balanced anchors used by Phase2B15
* regions: `PREFLOP_CONTINUATION_1`, `PREFLOP_CONTINUATION_2PLUS`
* evaluation seeds: `2029384436`, `1150634112`
* 2 independent blocks per anchor
* K=64 accepted posterior deals per block
* fixed traversal RNG within an anchor across blocks and accepted posterior deals
* no network fit, optimizer step, reservoir mutation, AveragePolicy fit, x4 confirmation, architecture selection, production training, or ready-for-tables claim

Phase2B15 result SHA256: `0e4f0a5bf2d48fb7f48b2763f8a65e3093d879aa50729f5d8a80d28fa9578f6a`.

## Windows heldout reconstruction

Use the already-audited Phase2B15 Windows runtime correction: reconstruct the acting player's private cards from authoritative SPNNIV3 ranks plus same-suit relation, create a canonical suit-isomorphic explicit deal, replay the frozen public action path, and require byte-identical SPNNIV3/actor/active-mask/legal identity before sampling.

The historical `deck_seed` must not be used to reconstruct a Linux-generated heldout deal on Windows.

## Exact posterior rejection sampler

For each anchor/behavior/block:

1. sample opponent private cards from the uniform prior conditional on the current actor's private cards;
2. replay the frozen observed action path with those cards and compute the exact frozen behavior likelihood L(h);
3. draw U~Uniform[0,1); accept iff U <= L(h);
4. after acceptance, sample a fresh future board uniformly from the remaining cards;
5. compute one raw Advantage target for the accepted deal;
6. repeat until exactly 64 accepted targets are obtained; average them arithmetically.

No likelihood floor, clipping, tempering, rescaling, MCMC, SIR, or posterior approximation is allowed.

A deterministic proposal/acceptance namespace is frozen and independent across blocks. Both behavior seeds use the same prior proposal seed sequence and acceptance uniforms; only their frozen behavior likelihoods differ.

## Runtime feasibility guard

This is an exact sampler, but some heldout paths may have very low absolute reach probability. Each task is capped at 50,000 prior private-card proposals for 64 accepted posterior draws. Hitting the cap is not repaired; the screen is classified `EXACT_POSTERIOR_REJECTION_COMPUTE_INFEASIBLE` and no target-training pilot is authorized.

Record acceptance rate, proposal count, minimum/median/max accepted likelihood, and seconds per task.

## Frozen comparison

Phase2B16 is compared to the exact successful Phase2B15 runtimefix partials and final result. Phase2B15 partial coverage and aggregate metrics must reproduce exactly before scientific interpretation.

Primary reference values:

* Phase2B15 unweighted pooled mean TV: `0.21994731031322912`
* Phase2B15 SNIS posterior pooled mean TV: `0.3157316176926827`
* Phase2B15 SNIS posterior sign disagreement: `0.25846354166666663`
* Phase2B15 SNIS posterior tail TV>=0.35: `0.328125`

## Frozen support gates

Exact rejection posterior sampling is considered supported only if all hold:

1. complete local validity and exact reproduction of the Phase2B15 runtimefix baseline/partial aggregate;
2. no rejection task hits the 50,000-proposal cap;
3. exact-posterior pooled mean TV improves vs Phase2B15 SNIS by >=0.05 absolute OR >=20% relative;
4. exact-posterior pooled mean TV <=0.24;
5. exact-posterior sign disagreement improves vs Phase2B15 SNIS by >=0.03 absolute OR >=15% relative;
6. exact-posterior tail TV>=0.35 is <=0.28;
7. both behavior seeds improve mean TV vs their Phase2B15 SNIS values;
8. neither continuation region is worse than its Phase2B15 SNIS mean TV by >0.01;
9. exact-posterior dominant-action mismatch does not exceed 0.28.

These thresholds are deliberately stronger than merely beating the failed SNIS estimator: a posterior method that remains substantially less stable than the unweighted continuation estimator is not sufficient to justify training.

## Decision hierarchy

* invalid/reproduction failure -> `PHASE2B16_INVALID_STOP_AUDIT`
* proposal cap hit -> `EXACT_POSTERIOR_REJECTION_COMPUTE_INFEASIBLE`
* all gates pass -> `EXACT_REJECTION_POSTERIOR_CONTINUATION_SUPPORTED`
* otherwise -> `EXACT_POSTERIOR_STILL_TOO_UNSTABLE_CLOSE_ESTIMATOR_REPAIR_PATH`

A PASS permits only a separately precommitted small continuation-target training pilot. A FAIL closes the current estimator-level posterior-repair line: no K128/K256 escalation and no additional importance-weight/SIR/MCMC tuning. The next architectural route is then explicit solver/representation support for reach-conditioned continuation information or fallback to the certified stable V1 control.

## Prohibitions

No threshold relaxation, no seed shopping, no anchor/scenario dropping, no higher behavior floor, no K sweep, no K128/K256, no Huber retry, no lagged-policy retry, no SNIS tuning, no weight clipping/tempering, no MCMC/SIR substitution after seeing results, no training from a failed screen, no full-x4 confirmation, no architecture winner, no production training.
