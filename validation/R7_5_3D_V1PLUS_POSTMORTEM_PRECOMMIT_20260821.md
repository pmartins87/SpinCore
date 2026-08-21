# R7.5.3D — V1+ forensic post-mortem precommit

Date: 2026-08-21
Status: FROZEN_BEFORE_POSTMORTEM_OUTPUTS
READY FOR TABLES: NO
Production training authorized: NO

## Purpose

Determine, before any further large training run, why the stable SPNNIV1 lineage and the richer SPNNIV3 H2/H3 lineage differ so strongly in cross-seed stability, and identify a representation/training path that preserves strategic information without carrying the observed V3 instability into production.

The target is not "return to V1" and not "keep V3 and train longer". The target is a V1+ successor that must satisfy two independent requirements:

1. hard cross-seed stability; and
2. strategic non-inferiority / improvement versus the stable V1 control under paired evaluation.

A candidate that is stable but strategically weak is ineligible. A candidate that is strategically attractive but unstable is also ineligible.

## Evidence entering this phase

- The historical SPNNIV1 lineage has certified cross-seed stability at the relevant R7.4 gate.
- SPNNIV3 H2/H3 preserves substantially more observable state, especially complete structured quantitative public history.
- The x4 R7.5.3C chance-coverage remediation left all eight H2/H3 cross-seed rows outside the hard gates despite all eight local training cells passing.
- The final x16 experiment materially improved HU stability, with H3 HU approximately at the hard boundary, while 3-handed stability improved only modestly and remained materially outside the gates.
- Previous RNG decomposition identified chance/deck variance as the dominant source of policy divergence, with action/traversal sampling secondary and final learner variation materially smaller.

These facts justify a causal post-mortem. They do not authorize selecting V1, H2, H3, or a new architecture in advance.

## Frozen hypotheses to test

### HYP-A — history-fragmentation hypothesis

The exact, unbounded, actor-aware and quantitatively sized V3 history fragments statistically similar poker situations into too many low-support contexts, especially in 3-handed play. V1's coarse last-32 street/action token stream forces stronger generalization and therefore lower variance, but may alias strategically distinct situations.

Prediction: cross-seed TV should rise with history complexity, quantitative sizing diversity, action-path length, and the number of exact-history variants that collapse to the same V1-like history projection.

### HYP-B — reservoir-pressure hypothesis

The V3 state space is too diverse for the frozen 100,000-sample reservoirs. Independent seeds may retain materially different subsets once `seen >> capacity`, causing different learned targets even when optimizer fit quality is locally good.

Prediction: high replacement pressure, low exact/projection overlap between seed reservoirs, and stronger instability in cells with greater unique-observation density.

### HYP-C — model-capacity / regularization hypothesis

The richer V3 network may have enough capacity to preserve seed-specific sampling noise that the V1 architecture smooths away.

Prediction: instability will remain high even within comparable history-complexity/memory-support strata and will correlate weakly with reservoir fragmentation. This hypothesis requires a later controlled capacity ablation; it must not be inferred merely from parameter count.

### HYP-D — H3-semantic amplification hypothesis

Objective semantic features may improve learnability in smaller HU state space but amplify partitioning or over-specialization in 3H.

Prediction: paired H3-minus-H2 state-level TV should be neutral/favorable in HU but systematically worse in specific 3H strata.

### HYP-E — action-output hypothesis

The universal 10-slot output layer may contribute to instability through specific aggressive action slots, but it is not presumed to be the primary cause because R7.5.3 used the frozen PF0 control action candidate.

Prediction: if true, most cross-seed L1 mass should concentrate in a small subset of legal aggressive slots across otherwise diverse states.

## Phase 1 — zero-new-training forensic readout

Use only the already-paid final x16 checkpoints and frozen heldout bundles. No model retraining, no threshold changes, no seed changes.

Required state-level diagnostics for every H2/H3 × HU/3H × evaluation-seed row:

- exact per-state total-variation distance between the two frozen training seeds;
- TV distribution by street;
- TV by public-history length and action-path length;
- TV by legal-action count;
- TV by current pot, to-call, current bet, stack geometry and SPR bins;
- TV by forced/non-forced history counts, actor diversity, last actor/action type, and action-type composition;
- TV by quantitative historical sizing statistics derived from paid/pot-before and commitment/pot-before where defined;
- action-slot decomposition of absolute policy disagreement;
- top high-TV heldout states with enough decoded context for manual inspection;
- paired H3-minus-H2 TV on the exact same heldout state identities.

## Phase 1 — V1-like projection diagnostics

From every V3 observation, derive read-only coarse history projections without modifying the trained network:

- V1-like history projection: last 32 `(street, action_type)` events, ignoring actor and quantitative event fields;
- structured-categorical projection: actor + street + action type + forced flag, ignoring quantitative event fields;
- exact V3 history identity.

Measure how many exact histories collapse into each coarser projection and whether state-level cross-seed TV increases with that refinement factor.

This is a diagnostic projection only. It is not a claim that the original V1 observation can be reconstructed byte-for-byte from V3, because V1 and V3 use different card encodings.

## Phase 1 — reservoir diagnostics

For every final checkpoint and both Advantage/Strategy reservoirs record:

- capacity;
- total samples seen;
- samples retained;
- seen/capacity saturation factor;
- exact observation uniqueness among retained samples;
- duplicate fraction;
- exact-observation overlap and Jaccard overlap between the two training seeds;
- V1-like history-projection overlap between seeds;
- structured-categorical-history overlap between seeds;
- distribution by iteration when available.

The purpose is to distinguish "too much information for the current memory budget" from "network optimization instability".

## Phase 1 decision rules

No architecture is selected from a single correlation. The post-mortem must combine state-level, reservoir-level, domain-level and H2/H3 evidence.

- If history richness/refinement and reservoir fragmentation jointly explain the 3H instability, prioritize a V1+ compressed structured-history representation.
- If reservoir pressure dominates independently of representation strata, prioritize a variance/memory remedy before changing observation semantics.
- If H3 semantics are helpful or neutral in HU but harmful only in identifiable 3H strata, test semantics separately rather than discarding the entire H3 idea.
- If disagreement is concentrated in specific action slots, carry that evidence into R7.5.4; do not silently attribute it to representation.
- If none of the above explains the divergence, run a small precommitted model-capacity/regularization ablation before any large successor training.

## Phase 2 — causal ablation after forensic evidence

Only after Phase 1 output is persisted, freeze a small controlled ablation matrix. Candidate dimensions may include:

- coarse V1 history;
- actor-aware categorical history without continuous sizing;
- compressed sizing features;
- exact full V3 history;
- compact versus current V3 network capacity;
- H3 semantics off/on.

The ablation must hold deals, training seeds, action candidate, optimizer family, evaluation states and hard gates fixed as far as mechanically possible. One factor at a time must be identifiable.

## V1+ design principle

Preserve information whose removal creates strategic aliasing; compress information whose exactness creates variance without demonstrated strategic value.

Strong default candidates to preserve from V3 include:

- suit/rank symmetry-aware card representation;
- exact current pot/stack/commitment geometry;
- explicit relative actor/position information;
- primitive legality;
- strategically meaningful history structure.

The full continuous event-by-event quantitative history is not pre-admitted. It must earn inclusion through the forensic and causal ablation evidence.

## Final admission rule

A V1+ successor is eligible for downstream production training only if both are true:

1. it passes the unchanged hard stability gates in both TRUE_HEADS_UP and THREE_HANDED; and
2. paired strategic evaluation shows non-inferiority or material improvement versus the stable V1 control, using common deals/states and precommitted confidence-interval rules.

Stability is an eligibility gate, not a strategic ranking metric. Strategic strength is a separate gate, not inferred from stability.

## Governance

- No threshold relaxation.
- No seed shopping.
- No dropping 3H because HU is easier.
- No production training while V1+ admission is unresolved.
- No claim that more optimizer steps or more roots alone will solve the problem without causal evidence.
- No representation winner is declared by this precommit.
