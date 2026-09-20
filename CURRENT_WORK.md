# SpinCore Current Work

Date: 2026-09-20
Status: **LT2 STAGE B PRESERVED — HU400 ONLINE PILOT PASS — CURRENT BEHAVIOR REPAIRED — AVERAGEPOLICY LAGS — REFRESH TO 8000 NEXT**

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

HU400 pilot:
- iteration 7600 / 4.56M roots;
- SHA256 `c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80`.

## Online pilot result — PASS

Current Advantage behavior, pilot minus Stage B:

- JAMMER: `+4.1830`, CI95 `[+0.3044,+8.0616]`;
- PASSIVE_CALLER: `+1.3629`, unresolved;
- UNIFORM_LEGAL: `+8.4233`, CI95 `[+4.3980,+12.4486]`.

The HU400 repair survives 100 real training iterations and online feedback.

## AveragePolicy

Pilot minus Stage B:

- JAMMER: `+0.6792`, unresolved;
- PASSIVE_CALLER: `+0.0045`, unresolved;
- UNIFORM_LEGAL: `+0.7639`, unresolved.

Deployment policy has not yet moved materially.

This is not a reason to change the intervention. It is the expected consequence of a mature 2M strategy reservoir after only 100 corrected iterations.

## Frozen intervention

- THREE_HANDED Advantage fit = 100;
- TRUE_HEADS_UP Advantage fit = 400;
- K4 off.

## Active gate

Continue the pilot checkpoint for exactly 400 more iterations:

- 7601..8000;
- +240,000 roots;
- cumulative HU400 exposure from Stage B = 500 iterations.

Then compare preserved Stage B directly with iteration 8000 on the full forensic HU policy-chain.

If AveragePolicy still does not move materially after 500 cumulative HU400 iterations, stop and audit policy-memory composition/weighting instead of blindly extending again.

Holdout remains sealed.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_hu_b400_refresh_to_8000.sh
```

Wait for `LT2_HU_B400_REFRESH_TO_8000_PASS`.

Then send `SpinCore_LT2_HU_B400_refresh_to_8000_summary.json`.

Do not continue beyond iteration 8000.
