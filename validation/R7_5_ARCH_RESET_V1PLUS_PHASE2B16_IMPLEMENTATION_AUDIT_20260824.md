# R7.5 Architecture Reset — Phase2B16 Implementation Audit

Date: 2026-08-24
Status: IMPLEMENTED / RYZEN TESTS NOT YET RUN

## Scope audited

Phase2B16 is a read-only final estimator-level continuation diagnostic. It does not alter solver training, networks, optimizer state, reservoirs, AveragePolicy, the Phase2B6 25% continuation behavior floor, the frozen H2 representation, or any historical evidence.

## Scientific invariants

* Same H2/THREE_HANDED source behavior checkpoints as Phase2B15.
* Same 64 Phase2B15 continuation anchors and two evaluation seeds.
* Same K=64 accepted posterior targets per block and two independent blocks.
* Same fixed traversal RNG namespace per anchor.
* Posterior private-card draws are exact rejection samples from the uniform private-card prior with acceptance probability equal to the product of frozen behavior probabilities along the already-observed public preflop path.
* Future board is sampled only after private-card acceptance and does not enter the preflop path likelihood.
* No weight floor, clipping, tempering, MCMC, SIR, K sweep, K128/K256, or post-result tuning.

## Rejection correctness review

The behavior path likelihood is a product of legal action probabilities and therefore lies in [0,1]. A proposal from the uniform hidden-card prior accepted with probability L(h) has accepted density proportional to prior(h)*L(h), exactly the desired action-history posterior. The implementation compares log(U) with log L to avoid numerical underflow and draws U strictly inside (0,1).

Private-card proposal, probe-board filler, acceptance uniform, and accepted future-board seeds use separate deterministic namespaces. Behavior seed is deliberately absent from those random namespaces, so both source behaviors see the same prior proposal/acceptance random numbers while their different frozen policies determine different accept/reject outcomes.

For every proposal and accepted target, replay must end at byte-identical SPNNIV3/actor/active-mask/legal identity. Accepted-deal likelihood is recomputed after replacing the probe board with an independently sampled future board and must match to 1e-12, directly checking the preflop board-independence assumption.

## Windows replay correction inheritance

Phase2B16 does not reuse historical heldout `deck_seed` to reconstruct Linux deals on Windows. It inherits the Phase2B15 audited canonical suit-isomorphic SPNNIV3 reconstruction and launcher preflights all 64 anchors before the screen.

## Phase2B15 baseline integrity

The runner requires the exact Phase2B15 result SHA256 `0e4f0a5bf2d48fb7f48b2763f8a65e3093d879aa50729f5d8a80d28fa9578f6a` and the successful runtimefix partials. Before B16 scientific interpretation, the partials must reproduce the frozen Phase2B15 aggregate unweighted/posterior TV, sign-disagreement, and tail metrics to absolute tolerance 1e-12.

## Resume and failure semantics

Each B16 task is stored atomically under `ryzen_v1plus_phase2b16/partials`. Cached rows bind to the current execution SHA and exact task identity. A task stops after 50,000 prior proposals if fewer than 64 posterior samples were accepted. Such a cap hit is classified as compute infeasibility; it is not repaired by changing thresholds or proposal mechanics.

## Decision firewall

PASS only permits a separately precommitted small causal continuation-target training pilot. FAIL closes the current estimator-level posterior-repair line. Neither outcome authorizes full x4, architecture selection, production training, or table use.

## Files

* `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B16_EXACT_REJECTION_POSTERIOR_CONTINUATION_PRECOMMIT_20260824.md`
* `tools/r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation.py`
* `tools/test_r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation.py`
* `tools/run_r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation_ryzen.ps1`
* `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B15_RESULT_EVIDENCE_20260824.json`

No claim is made that the real Windows/Ryzen synthetic or solver preflights have passed yet; those are launcher gates for the next execution.
