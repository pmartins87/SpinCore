# SpinCore — LT2 Advantage value-sensitivity result

Date: 2026-09-17
Status: **COMPLETE — LARGE SAMPLED-TARGET VALUE DISAGREEMENT; NEAR-TIE HYPOTHESIS REJECTED; TARGET-NOISE/APPROXIMATION DECOMPOSITION NEXT**

## Source

Stage B checkpoint: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

The audit evaluated 100,000 deterministic stored Advantage-memory samples per domain using the production Lean positive-regret plus masked-softmax policy mapping. It performed no optimizer steps and no root training.

Important interpretation limit: stored Advantage targets are Monte-Carlo external-sampling targets. The chip-equivalent quantities below measure disagreement **on those sampled targets**. They are not realized hand EV, exploitability, or a claim that the target-induced policy is a GTO oracle.

## Overall result

### THREE_HANDED

- target span mean: `398.300` chip-equivalent;
- model-policy regret to sampled-target best action: `153.993` chips/decision;
- target-policy regret to sampled-target best action: `52.631`;
- signed target-policy minus model-policy sampled-target value gap: `+101.362`;
- positive model value loss versus target-induced policy: `133.382`;
- absolute target/model-policy value gap: `165.403`;
- policy TV: `0.58873`;
- argmax agreement: `36.66%`;
- branch mismatch: `37.95%`.

### TRUE_HEADS_UP

- target span mean: `494.782` chip-equivalent;
- model-policy regret to sampled-target best action: `180.028` chips/decision;
- target-policy regret to sampled-target best action: `65.860`;
- signed target-policy minus model-policy sampled-target value gap: `+114.168`;
- positive model value loss versus target-induced policy: `156.783`;
- absolute target/model-policy value gap: `199.398`;
- policy TV: `0.59490`;
- argmax agreement: `30.07%`;
- branch mismatch: `40.51%`.

The disagreement is therefore not explained by TV on nearly indifferent sampled targets.

## Target-span concentration

The `<1 chip` span bucket has substantial state weight but essentially zero sampled-target value gap:

- 3H weight `8.39%`, signed gap about `0.0005` chips/decision;
- HU weight `10.35%`, signed gap about `0.0003`.

By contrast, the `100+ chip` target-span bucket carries:

- 3H weight `85.98%`, signed gap `116.223`; weighted contribution about `99.929`, or `98.6%` of the overall signed gap;
- HU weight `86.92%`, signed gap `130.586`; weighted contribution about `113.512`, or `99.4%` of the overall signed gap.

Thus the current sampled-target value disagreement is overwhelmingly a large-span phenomenon, not a near-tie artifact.

## Branch result

The all-nonpositive / positive-regret branch mismatch is conspicuous, but it does **not** explain the net sampled-target value loss by itself.

3H:

- target-has-positive weight `67.07%`, signed gap `+183.307`, weighted contribution `+122.94`;
- target-all-nonpositive weight `32.93%`, signed gap `-65.521`, weighted contribution `-21.58`.

HU:

- target-has-positive weight `62.98%`, signed gap `+237.422`, weighted contribution `+149.53`;
- target-all-nonpositive weight `37.02%`, signed gap `-95.530`, weighted contribution `-35.36`.

On sampled targets, the main positive loss lives in states where the target already has positive regret. A fallback-only calibration is therefore not justified as the next intervention.

## Action-mass signal

The model-induced Advantage behavior is much less fold-heavy and more call/shove-heavy than the target-induced sampled policy.

Overall 3H:

- FOLD target `39.37%`, model `15.39%`;
- CHECK_CALL target `28.23%`, model `45.76%`;
- ALL_IN target `22.02%`, model `26.21%`.

Overall HU:

- FOLD target `36.51%`, model `6.61%`;
- CHECK_CALL target `28.46%`, model `40.09%`;
- ALL_IN target `22.32%`, model `41.17%`.

Within HU target-has-positive states the contrast is stronger:

- FOLD target `47.66%`, model `7.02%`;
- CHECK_CALL target `19.53%`, model `39.30%`;
- ALL_IN target `17.81%`, model `39.09%`.

This is directionally consistent with a strategy that can perform poorly against an all-in-heavy opponent, but it is not yet a causal explanation for the confirmed HU-Jammer loss because the target-induced policy is built from noisy one-sample Advantage targets and the final benchmark agent uses AveragePolicy.

## Street result

Sampled-target signed gaps are broad rather than isolated to one street.

3H per-decision gap:

- preflop `181.64`;
- flop `116.18`;
- turn `73.08`;
- river `67.60`.

HU per-decision gap:

- preflop `200.77`;
- flop `149.58`;
- turn `96.00`;
- river `87.17`.

After multiplying by each street's audit weight, HU contribution is distributed across the hand: approximately `22.2`, `27.9`, `30.5`, and `33.6` chips/decision-equivalent for preflop through river. This is not evidence for a single-street bug.

## Why target variance is the next causal question

The collector constructs each target as `action_value - current_policy_node_value`. At traverser nodes all hero actions are expanded, while with production `exact_opponent_levels=0`, opponent decisions below those branches are Monte-Carlo sampled. Therefore a stored target can have substantial external-sampling variance even at an identical information state.

The current audit cannot distinguish:

1. irreducible / Monte-Carlo target noise;
2. model approximation error against the conditional target expectation;
3. representation/capacity error.

A direct repeated-state decomposition is required before changing architecture, loss, optimizer budget, or root count.

## Decision

Do **not** interpret `+101` or `+114` chip-equivalent sampled-target gaps as realized poker EV loss. Do **not** increase Advantage steps from this result. Do **not** change the fallback alone.

Next gate: `docs/LT2_REPEATED_TARGET_VARIANCE_AUDIT_20260917.md`.
