# SpinCore — Long-Training Plan

Status: **HU400 ONLINE PILOT PASS — POLICY-MEMORY REFRESH TO ITERATION 8000 ACTIVE**
Date: 2026-09-20

## Immutable milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

HU400 pilot:
- iteration 7600 / 4.56M roots;
- SHA256 `c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80`.

## Intervention

Frozen:

- 3H Advantage fit = 100;
- HU Advantage fit = 400;
- K4 off.

## Online pilot result

The first 100 online HU400 iterations passed.

Current behavior versus Stage B:
- JAMMER: resolved `+4.183`;
- PASSIVE_CALLER: no resolved regression;
- UNIFORM_LEGAL: resolved `+8.423`.

AveragePolicy did not materially change yet.

## Why another bounded block is justified

The algorithmic intervention itself has now passed:

1. low-noise structural validation;
2. broad fresh-refit validation;
3. online-feedback validation.

The remaining lag is in AveragePolicy deployment memory.

Therefore one additional bounded block is justified specifically to refresh policy memory under the corrected behavior.

## Stage C refresh

- source 7600;
- target 8000;
- +400 iterations;
- +240,000 roots;
- cumulative HU400 duration from Stage B: 500 iterations.

At 8000 run the full forensic HU policy chain against preserved Stage B.

## Stop rule

If AveragePolicy still has no measurable positive Jammer movement after iteration 8000:
- stop training;
- inspect policy-memory replacement/composition;
- inspect iteration weighting;
- inspect final policy fit;
- do not authorize another block merely to wait for drift.

If AveragePolicy begins moving while current behavior remains healthy:
- then design the next larger continuation gate.

Holdout `20261001..20261006` stays sealed.

## Immediate direction

Run `bash tools/run_lt2_hu_b400_refresh_to_8000.sh`.

Do not go past 8000.
