# SpinCore — LT2 HU B400 bounded online continuation pilot

Date: 2026-09-19  
Status: **PASS — CURRENT BEHAVIOR IMPROVES; AVERAGEPOLICY REFRESH REQUIRED**

## Trigger

The B400 broad generalization gate passed across all three deterministic refits.

The immediate question is now different:

**does the 400-step HU Advantage fit remain beneficial once it is put back inside the actual Deep-CFR training loop, where its behavior generates new roots and new AveragePolicy samples?**

## Minimal intervention

Only the implicated domain changes:

- THREE_HANDED Advantage fit: 100 steps, unchanged;
- TRUE_HEADS_UP Advantage fit: 400 steps;
- K4: off;
- all root-generation semantics unchanged;
- all other training configuration inherited from Stage B.

A domain-specific HU fit override is used so the 3H brain is not modified merely because the old config had one global fit-budget field.

## Pilot size

Source:
- Stage B iteration 7500;
- 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Pilot:
- iterations 7501..7600;
- 100 additional iterations;
- 600 roots/iteration;
- 60,000 additional roots;
- target total 4.56M roots.

This is deliberately small. The aim is not to claim convergence, but to test online feedback before authorizing another large block.

## Why 100 iterations

A fresh Stage-B reservoir fit already responds immediately to the 400-step budget.

The pilot therefore needs enough iterations to test repeated reset/refit plus feedback into the reservoirs, but not enough to risk another expensive blind continuation.

At the end, the current Advantage network has undergone 100 production-loop 400-step HU refits and the policy reservoir has begun receiving corrected HU behavior.

## Post-pilot evaluation

Automatically run the existing full-forensic HU policy-chain audit with:

- source = production Stage B;
- candidate = pilot checkpoint;
- forensic seeds `20260920..20260925`;
- 5000 scenarios/seed;
- UNIFORM_LEGAL, PASSIVE_CALLER and JAMMER;
- both AveragePolicy and current Advantage behavior.

Primary acceptance criterion:

- pilot current behavior improves versus Stage B against JAMMER with resolved positive paired delta;
- no resolved material current-behavior regression against the other two baselines.

AveragePolicy is secondary at this 100-iteration horizon because the 2M policy reservoir still contains predominantly historical samples from the previous regime.

## Decision branches

### Current behavior passes, AveragePolicy has not moved enough

Extend the same intervention in a second bounded block to refresh more policy memory.

### Current behavior and AveragePolicy both improve

Proceed to a larger but still gated continuation block.

### Current behavior loses the B400 repair

Stop. The fresh-refit result does not survive online feedback; inspect target/reservoir dynamics before more roots.

### A different baseline regresses materially

Stop and localize the tradeoff before extension.

Holdout `20261001..20261006` remains sealed.

## Result

See `docs/LT2_HU_B400_ONLINE_PILOT_RESULT_20260920.md`.
