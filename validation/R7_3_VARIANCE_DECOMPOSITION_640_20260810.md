# SpinCore R7.3 — 640-root variance decomposition

Physical workflow: `31354594794`

Evidence commit: `efd732288cc0d1414cfe791208fb23433f608be1`

Status: **R7.3 FAIL / ACTIVE**. `READY FOR TABLES = NO`.

## Why this diagnostic was run

The corrected 640-root run satisfied both per-seed approximation gates but failed cross-seed stability. Doubling to 1280 roots left cross-seed mean TV essentially flat (`0.477649 -> 0.473190`) and caused AveragePolicy fit to exceed the frozen `0.12` gate on both seeds. Root scaling was therefore paused so the source of instability could be decomposed instead of spending more compute blindly.

This diagnostic trained two controlled AveragePolicyNet replicas on each of the two independently generated 640-root CFR strategy memories. All four replicas used the recovered 152,438-parameter architecture and were trained up to 8,192 optimizer steps with the frozen policy-fit gate unchanged.

## Physical result

All four controlled policy fits passed the frozen policy-fit gate (`weighted mean TV <= 0.12`).

Comparing two independently initialized/optimized policy networks trained on the **same CFR memory** produced:

- memory A mean TV: `0.2531768084`
- memory B mean TV: `0.2321299911`
- average within-memory mean TV: `0.2426533997`

Comparing policy networks with the **same optimizer/init seed but trained on different CFR memories** produced:

- controlled pair 0 mean TV: `0.4699736536`
- controlled pair 1 mean TV: `0.4698107839`
- average across-memory mean TV: `0.4698922187`

The across-memory / within-memory mean-TV ratio is `1.9364749032`.

The persisted diagnostic classification is:

`CFR_MEMORY_VARIANCE_DOMINANT`

## Important interpretation limit

The within-memory comparisons above were evaluated on a common observation set formed from both memories. Therefore the measured `0.24265` within-memory disagreement includes policy extrapolation onto observations that may be outside the training memory's own support. It must **not** be interpreted as pure optimizer variance without a support-conditioned follow-up.

The across-memory result is still decisive enough to show that policy fitting alone cannot explain the observed ~`0.47` cross-seed disagreement: even when optimizer initialization is controlled and all four policy-fit gates pass, changing the CFR memory leaves almost the full instability intact.

## Next diagnostic

Before changing production semantics or resuming brute-force root scaling, run a controlled **shared-deck, support-conditioned** experiment:

1. give both algorithm seeds the same hidden-card/future-board deck stream;
2. preserve independent model/reservoir/action-sampling randomness;
3. fit two controlled AveragePolicy replicas per memory;
4. evaluate same-memory and across-memory TV separately on memory-A support, memory-B support, and their union.

This separates two remaining hypotheses:

- card/chance coverage is a major source of CFR-memory divergence;
- algorithmic/CFR dynamics remain unstable even under common random cards.

It also separates on-support optimizer disagreement from off-support extrapolation.

## Frozen gates remain unchanged

- Advantage weighted normalized RMSE <= `0.75`
- Average-policy weighted mean TV <= `0.12`
- Cross-seed mean TV <= `0.15`
- Cross-seed p95 TV <= `0.35`

No gate was relaxed and this diagnostic is not itself an R7.3 acceptance run.
