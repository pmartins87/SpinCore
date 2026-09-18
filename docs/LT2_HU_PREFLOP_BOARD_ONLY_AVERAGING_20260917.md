# SpinCore — LT2 HU preflop board-only averaging audit

Date: 2026-09-17
Status: **ACTIVE — READ-ONLY IMPLEMENTATION-FEASIBILITY GATE**

## Why this gate exists

The completed target-estimator sweep established that:

- exact0 plus more independent hidden deals is decisively more target-MSE-efficient at matched node cost than exact1;
- exact1 has no reproducible policy-space advantage large enough to justify its ~2.1x node cost;
- future-board variance is the largest single hidden/chance component;
- full opponent-hand posterior resampling is materially harder to integrate into production training than future-board resampling.

The next question is therefore practical:

**How much of the full hidden-deal averaging benefit can be captured by averaging only future boards while keeping the currently sampled opponent hand fixed?**

## Source

Stage B checkpoint:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Read only. No training roots and no optimizer steps.

## Anchors

Same deterministic HU-preflop anchor strata:
- 16 root;
- 32 one-action continuation;
- 16 two-or-more-action continuation.

FACING_ALL_IN remains a separate subset.

## Reference

For each anchor:
- enumerate exact posterior over all 2,450 ordered opponent hands;
- select 16 posterior-stratified opponent hands;
- sample 4 independent future boards per hand;
- exact opponent level 1;
- average all 64 hidden deals to form a lower-noise conditional reference.

The reference remains diagnostic, not GTO.

## Production-feasible candidate

For each anchor:
- select 16 independent posterior-stratified opponent hands from a separate stream;
- keep each selected opponent hand fixed;
- sample 8 independent future boards for that same hand;
- use production-like exact opponent level 0;
- form board-only averages at K = 1, 2, 4, 8 boards.

Each estimator therefore conditions on one sampled opponent hand but integrates more future-board chance.

This mirrors the intervention that is much easier to implement inside the current preflop target-generation path.

## Metrics

Per K:
- target MSE to the full conditional reference;
- regret-matching policy TV;
- argmax agreement;
- branch mismatch;
- signed reference-policy minus candidate-policy value gap;
- candidate policy regret to the reference best action;
- actual traversal nodes.

## Decision

Compare this result to the already-completed full-independent-hidden-deal exact0 frontier.

- If board-only K4/K8 captures most of the policy-space gain at much lower implementation complexity, build the first bounded training pilot around board averaging.
- If board-only plateaus far above full-deal averaging, opponent-hand posterior variance must also be attacked before training.
- If board-only lowers MSE but barely changes TV/regret, move next to a policy-aligned Advantage objective rather than brute-force chance averaging.

No long root scaling is authorized by this audit.

## Launcher

```bash
bash tools/run_lt2_hu_preflop_board_only_averaging.sh
```

Expected marker:

`LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_PASS`
