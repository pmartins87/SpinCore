# R7.5 Architecture Reset — V1+ Phase2B5 Preflop Feedback Stabilization Screen

Status: **FROZEN BEFORE PHASE2B5 OUTPUTS**  
Date: 2026-08-22

## 1. Purpose

Phase2B4 localized the dominant same-chance downstream feedback to **preflop continuation behavior**. With the root baseline controlled and postflop policies commonized, pooled target-policy TV remained `0.32010786853721923`; commonizing the preflop continuation as well reduced it to `0.060271017892879135` (81.6% reduction from the Phase2B3 common-root-sigma reference).

Phase2B5 is a read-only screen with two goals before any new training:

1. localize how early in the preflop continuation tree the feedback enters; and
2. test whether a **seed-independent, mild preflop policy smoothing** can materially reduce cross-seed target divergence without requiring the impossible oracle operation of averaging two independent training seeds together.

No optimizer step, model fit, reservoir insertion, checkpoint mutation, architecture selection, production training, or table use is permitted.

## 2. Frozen source identity

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED`.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Exact opponent levels: `2`.
- Source Phase2A execution SHA: `4bfa55d69029cd69536fa6dbfcadd162719cb887`.
- Source behavior seeds: `1342191342`, `1801739323`.
- Phase2B1 result SHA-256: `f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef`.
- Phase2B4 result SHA-256: `6b639b1608a0572c0ae2f6641c038786fa30cb8858bdda62da3fd5e30f49f0aa`.
- Frozen Phase2B4 control TV (`COMMON_FROM_FLOP`): `0.32010786853721923`.
- Frozen Phase2B4 oracle-common-preflop TV (`COMMON_FROM_PREFLOP`): `0.060271017892879135`.

Use exactly the 15 Phase2B1 collision groups and their 16 stored deck seeds. No new collision search is allowed.

## 3. Common controls in every arm

For every stored deal:

- the root traverser baseline policy is the arithmetic mean of the two source behavior policies, exactly as in Phase2B4;
- both source traversals use independent deterministic traversal-RNG namespaces exactly as Phase2B3/B4;
- all policy calls from **flop onward** use the pointwise arithmetic mean of the two source behavior policies;
- only the **preflop continuation policy after the root action** varies by arm.

Thus Phase2B5 isolates preflop continuation feedback.

## 4. Exact preflop depth definition

SPNNIV3 stores complete public history. For each root, record the number of public events with `street == PREFLOP` and `forced == false`. For a later preflop policy call, define:

`delta = current_nonforced_preflop_events - root_nonforced_preflop_events`.

The child reached after the fixed root action therefore has `delta >= 1`. This definition is observable, deterministic, and independent of process scheduling.

The diagnostic parser must verify the SPNNIV3 wire length (`120 + 20 * history_count`) before using the history.

## 5. Arm family A — oracle depth localization

- `PREFLOP_NATIVE_POSTFLOP_COMMON`: source-specific preflop continuation, common postflop. Must reproduce Phase2B4 `COMMON_FROM_FLOP` TV exactly within `1e-12`.
- `DEPTH_COMMON_GE_1`: use common mean policy on preflop calls with `delta >= 1`. Must reproduce Phase2B4 `COMMON_FROM_PREFLOP` TV exactly within `1e-12`.
- `DEPTH_COMMON_GE_2` ... `DEPTH_COMMON_GE_6`: leave progressively more early preflop decisions source-specific, then commonize the remaining preflop continuation.

These are oracle localization arms only. They can never authorize training because cross-seed policy averaging is unavailable inside an independent production learner.

Sequential increments are measured as:

- `DELTA1 = TV(GE2) - TV(GE1)`;
- `DELTA2 = TV(GE3) - TV(GE2)`;
- ...;
- `DELTA5 = TV(GE6) - TV(GE5)`;
- `DEEPER_THAN_6 = TV(CONTROL) - TV(GE6)`.

Because traversal and regret matching are nonlinear, negative increments are retained rather than clipped; the largest positive increment is localization evidence only.

## 6. Arm family B — seed-independent uniform-floor smoothing

On every **preflop continuation** policy call, take the native source policy `p` and replace it with:

`p' = (1 - f) * p + f * U(legal)`

where `U(legal)` is uniform over the exact legal universal actions.

Frozen floors:

- `UNIFORM_FLOOR_010` (`f=0.10`)
- `UNIFORM_FLOOR_025` (`f=0.25`)
- `UNIFORM_FLOOR_050` (`f=0.50`)
- `UNIFORM_FLOOR_075` (`f=0.75`)
- `UNIFORM_FLOOR_100` (`f=1.00`)

The already-existing uncertainty-damped source behavior is not modified internally. This screen adds a deterministic preflop floor *after* the frozen source policy, so the only changed variable is the extra seed-independent smoothing strength.

## 7. Frozen pilot screen

Reference is `PREFLOP_NATIVE_POSTFLOP_COMMON`.

A **mild** smoothing candidate (`f <= 0.25`) may authorize freezing one small causal training pilot only if all of the following hold:

1. pooled mean target-policy TV absolute reduction >= `0.08`;
2. relative reduction >= `25%`;
3. pooled residual TV <= `0.24`;
4. at least `12/15` scenarios improve in mean TV versus control;
5. no scenario degrades by more than `0.05` absolute TV;
6. pooled dominant-action mismatch rate does not increase by more than `0.02` absolute.

Select the **smallest floor** satisfying all six conditions.

If no mild candidate passes but `f=0.50` satisfies the same numerical conditions, classify `HEAVY_DAMPING_REQUIRED_NO_PILOT` and do not train yet; the next route is a better preflop anchor / lagged-target diagnostic because 50% forced uniform mixing is strategically intrusive.

If only `f>=0.75` satisfies them, classify `STRONG_ANCHOR_REQUIRED_NO_PILOT`.

If even `f=1.00` does not satisfy them, classify `UNIFORM_DAMPING_INSUFFICIENT`.

Only `MILD_PREFLOP_DAMPING_CANDIDATE` sets `small_training_pilot_precommit_allowed=true`. Production training remains false in every Phase2B5 outcome.

## 8. Compute contract

There are 12 arms total: one control, six oracle depth arms, and five uniform-floor arms.

Exact work: `15 scenarios x 16 deals x 2 source behaviors x 12 arms = 5760` root action-value reconstructions.

- up to 12 independent worker processes;
- one Torch/OMP/MKL thread per worker;
- deterministic aggregation sorted by scenario;
- no learned-state mutation.

## 9. Interpretation guardrail

A successful uniform-floor screen would show that preflop feedback is regularizable by a seed-independent smoothing mechanism. It would **not** prove strategic strength or production readiness. Any later pilot must be evaluated on independent chance blocks and eventually compared against the stable V1 control for strategic quality.

`READY FOR TABLES = NO`.
