# R7.5 Architecture Reset — Phase2B10 Private/Public Chance Decomposition

Status: **FROZEN BEFORE PHASE2B10 OUTPUTS**  
Date: 2026-08-24

## 1. Why this diagnostic exists

Phase2B9 rejected robust Advantage regression as the next stabilization mechanism. The paired Huber beta=0.02 fit slightly improved source-memory Advantage NRMSE for both training seeds, but worsened cross-seed heldout behavior mean TV on both heldouts and worsened pooled mean TV from `0.30567460296341376` to `0.3161210369628588`. No Huber trajectory pilot is authorized.

The strongest upstream evidence remains Phase2B1: with the acting player's exact root SPNNIV3 observation held fixed, changing hidden/future chance produced K1 regret-matching target-policy TV about `0.51537`, while changing traversal/action-sampling RNG on the same full deal produced only about `0.05194`.

Phase2B1 intentionally bundled two distinct poker chance components:

1. opponents' private hole cards;
2. future public board runout.

Before modifying training again, Phase2B10 separates those components exactly at the preflop root. The result decides whether the next variance-reduction design should target public board chance, private-hole chance, or both.

## 2. Frozen source

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED` only.
- Source behavior trajectories: exact completed Phase2B6.
- Training behavior seeds: `1342191342`, `1801739323`.
- Phase2B6 result SHA-256: `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`.
- Phase2B9 result SHA-256: `71f77b6921597c7b1d048f8fb3e448f5fce74a974b247ac4ca88383fece5c64a`.
- Phase2B1 result SHA-256: `f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef`.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Exact opponent levels: `2`.
- Target iteration: `3`.
- Phase2B6 continuation behavior contract remains active during diagnostic traversals: root native; preflop continuation uses the frozen 25% uniform floor; postflop native.
- No model fit, optimizer step, reservoir insertion, Strategy collection, or checkpoint mutation is permitted.

## 3. Additive diagnostic solver capability

The current solver predeals all private cards and the entire five-card board from one deck seed at root. Existing ABI-v2 exposes creation by seed and state cloning, but it does not expose a way to hold one chance component fixed while changing another.

Phase2B10 therefore permits one **additive read-only/diagnostic ABI-v2 extension**:

- export the exact root deal as card ids;
- create a fresh root state from an explicitly supplied valid deal.

The existing seed-based constructor and all existing solver/training entry points remain unchanged. Explicit-deal creation must validate card ids, live/dead seats, uniqueness, and exact round-trip identity. The diagnostic launcher must build a fresh solver and pass synthetic round-trip tests before scientific work begins.

This extension does not authorize explicit-deal production training by itself.

## 4. Frozen experimental geometry

Reuse all 15 exact-root Phase2B1 collision groups. For each scenario use the first **4** stored collision deck seeds as independent anchor deals. Every anchor is known to reproduce the same exact acting-player root SPNNIV3 observation within its scenario.

For each anchor generate **8** deterministic replicates for each arm below. All resampling is conditional on the actor's two physical hole cards and uses deterministic precommitted seed namespaces; there is no result-dependent seed selection.

### A. `TRAVERSAL_ONLY`

- hold the entire explicit deal fixed;
- vary only traversal RNG across the eight replicates.

This is the action/traversal-noise reference.

### B. `PRIVATE_ONLY`

- hold actor hole cards fixed;
- hold the complete five-card board fixed to the anchor;
- uniformly resample the four opponent-hole card slots from all cards not occupied by actor hole cards or the fixed board;
- hold traversal RNG fixed.

This isolates opponents' private-card chance conditional on one fixed public runout.

### C. `PUBLIC_ONLY`

- hold all six player hole cards fixed to the anchor;
- uniformly resample the ordered five-card board from all remaining cards;
- hold traversal RNG fixed.

This isolates future public-board chance conditional on one fixed private deal.

### D. `COMBINED`

- hold actor hole cards fixed;
- uniformly resample both four opponent-hole slots and the ordered five-card board from the remaining 50 cards;
- hold traversal RNG fixed.

This reproduces the conditional chance family measured more coarsely by Phase2B1 while making the two components factorable.

Each explicit-deal variant must preserve exact root actor, exact root SPNNIV3 observation, and exact universal legal mask. Any violation aborts the diagnostic.

## 5. Exact work

Per behavior seed:

`15 scenarios × 4 anchors × 4 arms × 8 replicates = 1,920 root target traversals`.

Across both frozen Phase2B6 behavior seeds:

`3,840 root target traversals`.

Workers may be parallelized aggressively because tasks are independent. Each worker must use one Torch/OMP/MKL thread. Worker count is a compute-only parameter and does not change statistical semantics.

## 6. Readouts

For every `(behavior seed, scenario, anchor, arm)` collect eight raw ten-slot root Advantage target vectors. Compute deterministic non-overlapping K1 pairs `(0,1), (2,3), (4,5), (6,7)` and report:

- legal target mean-absolute difference;
- legal positive/non-positive sign-disagreement fraction;
- regret-matching policy TV;
- dominant-action mismatch.

Report pooled and by-behavior-seed summaries for all four arms.

Define excess chance TV relative to traversal reference:

- `private_excess = PRIVATE_ONLY_K1_TV - TRAVERSAL_ONLY_K1_TV`;
- `public_excess = PUBLIC_ONLY_K1_TV - TRAVERSAL_ONLY_K1_TV`;
- `combined_excess = COMBINED_K1_TV - TRAVERSAL_ONLY_K1_TV`.

Negative excess values are retained in evidence but treated as zero only for dominance ratios.

## 7. Frozen classification rule

A component is considered materially active if its pooled excess TV is at least `0.10`.

Classification:

- `PUBLIC_BOARD_CHANCE_DOMINANT` if public excess is material and is at least `1.5 ×` private excess, with PUBLIC_ONLY > PRIVATE_ONLY for both source behavior seeds.
- `PRIVATE_HOLE_CHANCE_DOMINANT` if private excess is material and is at least `1.5 ×` public excess, with PRIVATE_ONLY > PUBLIC_ONLY for both source behavior seeds.
- otherwise `MIXED_PRIVATE_PUBLIC_CHANCE` if combined excess is material.
- otherwise `CHANCE_COMPONENT_DECOMPOSITION_UNRESOLVED`.

Also report an interaction diagnostic:

`interaction_excess = COMBINED_ONLY_K1_TV - max(PRIVATE_ONLY_K1_TV, PUBLIC_ONLY_K1_TV)`.

No interaction threshold changes the primary classification post hoc.

## 8. Frozen routing

- `PUBLIC_BOARD_CHANCE_DOMINANT` -> `PRECOMMIT_PUBLIC_CHANCE_SAMPLING_OR_STRATIFIED_BOARD_DIAGNOSTIC`.
- `PRIVATE_HOLE_CHANCE_DOMINANT` -> `PRECOMMIT_PRIVATE_HAND_STRATIFIED_CHANCE_DIAGNOSTIC`.
- `MIXED_PRIVATE_PUBLIC_CHANCE` -> `PRECOMMIT_FACTORIZED_PRIVATE_PUBLIC_CHANCE_VARIANCE_REDUCTION_DIAGNOSTIC`.
- unresolved -> `REASSESS_REPRESENTATION_SUPPORT_AND_CHANCE_INTERACTION_BEFORE_TRAINING`.

A Phase2B10 classification does **not** authorize a training pilot. The next intervention must be independently precommitted after the component source is known.

## 9. Guardrails

- no Huber beta tuning;
- no Phase2B8 lag-weight tuning;
- no higher uniform floor;
- no seed shopping;
- no threshold relaxation;
- no dropped scenario or source behavior seed;
- no explicit-deal production training in this phase;
- no reservoir insertion;
- no optimizer step;
- no AveragePolicy fit;
- no architecture winner selection;
- `READY FOR TABLES = NO`.

## 10. Strategic firewall

Stability remains only one admission dimension. Even if later chance-variance work satisfies the historical cross-seed gates, the candidate must still pass a separately precommitted strategic-strength comparison against the certified stable V1 control before architecture selection.
