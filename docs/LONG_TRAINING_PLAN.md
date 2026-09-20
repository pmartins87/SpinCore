# SpinCore — Long-Training Plan

Status: **ITERATION 8000 FROZEN — DEPLOYMENT POLICY IMPROVES — TRAINING-BEHAVIOR TRADEOFF UNDER FORENSIC REVIEW**
Date: 2026-09-20

## Immutable checkpoints

Stage B 7500:
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

HU400 pilot 7600:
- SHA256 `c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80`.

HU400 refresh 8000:
- SHA256 `773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886`.

## What has succeeded

The original Jammer defect has been repaired at multiple levels.

At iteration 8000 versus Stage B:

AveragePolicy:
- JAMMER: resolved `+1.902`;
- PASSIVE_CALLER: no resolved regression;
- UNIFORM_LEGAL: no resolved regression.

Current behavior:
- JAMMER: resolved `+5.531`.

## New blocker

Current behavior versus PASSIVE_CALLER is now:

- `-2.921`;
- CI95 `[-5.541,-0.302]`.

This violates the preregistered no-current-behavior-tradeoff condition.

## Decision

No more root training until localized.

Do not discard iteration 8000: its AveragePolicy is a promising deployment candidate.

Do not unseal holdout yet: first resolve whether the current-behavior regression is a localized training-policy artifact or a broader HU400 tradeoff.

## Next gate

Read-only direct 7600 -> 8000 first-divergence forensic.

If one state class dominates the Passive loss:
- inspect that class next.

If diffuse:
- evaluate fit-budget/time specialization versus passive exploitation before any continuation.

## Immediate direction

Run `bash tools/run_lt2_hu_7600_8000_behavior_first_divergence.sh`.

Stop after the report.
