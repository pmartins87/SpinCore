# R7.5.3D — V1+ Phase 1B projection findings

Date: 2026-08-21
Status: PHASE1B_FINDINGS_PERSISTED / NO_ARCHITECTURE_SELECTED
READY FOR TABLES: NO
Production training authorized: NO

## Provenance

Input: `R7_5_3D_V1PLUS_PHASE1B_PROJECTION_DECOMPOSITION.json`

- schema: `SPINCORE_R7_5_3D_V1PLUS_PHASE1B_PROJECTION_DECOMPOSITION_V1`
- status: `PHASE1B_COMPLETE_NO_ARCHITECTURE_SELECTED`
- x16 training execution SHA: `f44e05513721b59f63ed5c61f37de2c115c67315`
- diagnostic execution SHA: `d56404b665b9c613833cd97578621ca70bec9267`
- representations: H2/H3 final
- domains: TRUE_HEADS_UP / THREE_HANDED
- memories: Advantage / Strategy
- training seeds: 1342191342 / 1801739323

No training, traversal, model mutation, reservoir mutation, threshold change, seed change, or architecture selection occurred in Phase 1B.

## Main result

Phase 1B rejects a simple explanation in which the final x16 failure is caused primarily by exact card identity or by the continuous numeric fields of V3 history.

The strongest domain split is a combination of:

1. radically greater 3H Strategy-memory pressure;
2. much poorer cross-seed support alignment for 3H current geometry and public action paths; and
3. a likely feedback loop in which independent training seeds visit different 3H trajectories before the final AveragePolicy fit.

The Strategy reservoir is therefore a high-priority causal bottleneck, but it cannot by itself be the entire upstream root cause because Strategy memory is passive: it is populated by `collect_strategy_own_reach` and is used to fit the final average policy, while behavior during traversal is driven by the Advantage policy. Different upstream Advantage/chance trajectories can therefore create different Strategy streams before reservoir truncation acts.

## Strategy-memory saturation remains the largest resource asymmetry

All final reservoirs have capacity 100,000.

- HU Strategy seen: ~147k–158k; retained fraction ~63–68%.
- 3H Strategy seen: ~3.22M–4.02M; retained fraction ~2.49–3.10%.
- HU Advantage seen: ~1.42M–1.47M.
- 3H Advantage seen: ~1.15M–1.32M.

Thus the pronounced 3H/HU asymmetry is specific to Strategy-memory volume, not generic memory pressure. A uniform 100k Strategy reservoir is an unbiased sample of its stream, so low retention alone is not proof of insufficiency; the causal question is whether 100k is enough for the much higher-dimensional 3H Strategy distribution. That must be tested directly.

## Strategy-support decomposition

Cross-seed Jaccard overlap of retained Strategy observations:

| projection | H2 HU | H2 3H | H3 HU | H3 3H |
|---|---:|---:|---:|---:|
| cards_only | 0.01706 | 0.01133 | 0.01630 | 0.01152 |
| geometry_only | 0.80635 | 0.37176 | 0.80920 | 0.36817 |
| fixed_state_no_history | 0.02336 | 0.00196 | 0.02214 | 0.00176 |
| history_exact | 0.50730 | 0.09957 | 0.51262 | 0.09929 |
| history_structured | 0.72331 | 0.15555 | 0.73073 | 0.15878 |
| history_v1_like | 0.72331 | 0.15735 | 0.73073 | 0.16061 |
| no_cards_plus_exact_history | 0.47535 | 0.06015 | 0.47764 | 0.05911 |
| no_cards_plus_structured_history | 0.49563 | 0.06227 | 0.49678 | 0.06134 |

### Cards are not the discriminator

`cards_only` exact-support overlap is very low in both HU and 3H. 3H is lower, but only modestly relative to the enormous difference in policy stability. Since HU can approach the hard stability gate despite similarly tiny exact-card Jaccard, exact same-card overlap is not sufficient to explain the 3H failure.

### Current geometry is materially less aligned in 3H

`geometry_only` falls from ~0.81 in HU to ~0.37 in 3H for Strategy memory, and a similar HU→3H drop appears in Advantage memory. Independent seeds therefore visit substantially different stack/pot/position/legal-state support in 3H.

### Public action-path support also collapses in 3H

For Strategy memory, `history_structured`/`history_v1_like` overlap falls from ~0.72–0.73 HU to ~0.156–0.161 3H. This collapse remains even after removing all continuous event sizing fields.

Therefore exact quantitative history is not the root cause.

### Continuous history is a secondary amplifier

Within Strategy memory, replacing `history_exact` with `history_structured` raises 3H Jaccard from ~0.099 to ~0.156–0.159. Numeric event detail therefore fragments history support materially.

However, once current no-card geometry is combined with history, the difference between exact and structured history is small:

- H2 3H: 0.06015 exact vs 0.06227 structured;
- H3 3H: 0.05911 exact vs 0.06134 structured.

This shows that the dominant joint-support problem is already present in the geometry × categorical-path combination. Compressing only continuous history cannot plausibly solve the whole instability.

### Actor-aware structured history is not the main issue

`history_structured` and `history_v1_like` are almost identical in overlap. Removing actor/forced detail and using the V1-like `(street, action_type)` projection barely changes the support overlap. Therefore a blind return to V1 history semantics is not justified by Phase 1B.

## Evidence of upstream trajectory divergence

The total number of samples generated by the two independent seeds differs much more in 3H than HU.

Approximate relative seed differences in samples seen:

- H2 HU Strategy: 0.34%; H2 3H Strategy: 11.7%.
- H3 HU Strategy: 2.3%; H3 3H Strategy: 21.9%.
- H2 HU Advantage: 2.3%; H2 3H Advantage: 9.0%.
- H3 HU Advantage: 0.8%; H3 3H Advantage: 13.5%.

Because Strategy memory does not drive traversal behavior, this indicates that independent 3H training trajectories are already diverging upstream, consistent with the earlier chance/deck-dominant RNG decomposition. The 100k Strategy reservoir can then amplify the final AveragePolicy difference by retaining only a small sample of two already-different high-dimensional streams.

## Refined causal model

The best-supported mechanism after Phase 1B is:

`chance / Advantage-path variation -> different 3H geometry and action-path visitation -> multi-million-sample Strategy streams -> fixed 100k Strategy reservoir -> under-resolved AveragePolicy target distribution -> large cross-seed final-policy TV`.

Exact quantitative V3 history adds fragmentation on top of this chain but is not required for the failure.

H3 semantic features remain a secondary factor, not a primary cause.

## Decision

Do not compress V3 history first and do not return to V1 representation wholesale.

The first training ablation must isolate the Strategy-memory-capacity bottleneck while leaving traversal behavior, Advantage memory, cards, representation, action candidate, optimizer family, training seeds and chance schedule unchanged.

A clean design is to fork the same generated Strategy sample stream into multiple passive reservoirs of different capacities. Because the Strategy reservoir is not used by the behavior policy, these shadow reservoirs can be populated in one traversal without changing the upstream trajectory. Separate AveragePolicy fits at the end then measure whether larger retained Strategy support reduces cross-seed TV.

Only after this capacity effect is identified should a separate sampling/variance-reduction ablation be introduced. Representation compression remains a later arm if memory/sampling repair is insufficient.

Stability remains an eligibility gate. Strategic strength remains a separate selection gate; no capacity arm can be selected for production merely because its cross-seed TV improves.