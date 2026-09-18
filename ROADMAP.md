# SpinCore Roadmap — active state 2026-09-18

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**: 1.8M roots.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- 30k weak-baseline gate — **HU JAMMER NEGATIVE; PASSIVE HU REGRESSION ALSO CONFIRMED**.
- AveragePolicy extra-fit — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED**.
- HU-preflop conditional target decomposition — **93.73% HIDDEN/CHANCE VARIANCE**.
- exact0/exact1 estimator sweep — **EXACT0 + MORE DEALS WINS COMPUTE FRONTIER**.
- board-only averaging — **K4 ESTIMATOR ELBOW**.
- K4 mechanics smoke — **PASS**.
- Stage-A -> Stage-B first-divergence forensic — **COMPLETE**.
- Jammer FACING_ALL_IN target overlay V1 — **COMPLETE; DIRECTIONALLY POSITIVE BUT STAGE-SPECIFIC REFERENCE MISMATCHED**.
- Jammer COMMON-REFERENCE V2 — **NEXT**.
- K4 causal training pilot — **NOT AUTHORIZED YET**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Forensic result

Jammer Stage-B-minus-Stage-A:
- total `-1.682`, CI `[-2.767,-0.597]`;
- FACING_ALL_IN first divergence contribution `-1.126`, CI `[-2.049,-0.202]`;
- root contribution `-0.557`, CI `[-1.139,+0.026]`.

The FACING_ALL_IN class alone explains about 67% of the Jammer regression.

PassiveCaller simultaneously shows a resolved FLOP contribution of `-0.672`, so a HU-preflop K4 fix cannot be assumed to solve the whole Stage-B regression.

## Anti-overfitting status

The K4 hypothesis now has three independent supports:
1. independent target-variance diagnosis;
2. independent compute-normalized estimator selection;
3. independent deployed-policy localization to FACING_ALL_IN.

This is stronger than tuning directly to Jammer.

One causal bridge still remains: confirm that the actual Stage-B Advantage and AveragePolicy errors on those forensic states point in the same wrong direction and that K4 improves the target estimator there.

## V1 overlay correction

V1 showed Stage B farther from its own reference in AveragePolicy TV and Advantage TV, and K4 improved estimator TV.

However, the reference was stage-specific self-play posterior.

For the fixed JAMMER benchmark this is not the correct A-vs-B reference: JAMMER's action rule depends only on legal actions and is independent of hole cards. The observed shove therefore leaves the opponent-hand posterior uniform.

Direct A-vs-B causal comparison requires one shared Jammer-conditioned reference.

## Next gate

`tools/run_lt2_jammer_facing_allin_common_reference_v2.sh`

V2 uses:
- common uniform compatible opponent hands;
- uniform future boards;
- one shared target reference for A and B;
- fixed-deal Stage-A/Stage-B target equality assertion;
- K1 vs K4 against the same reference.

Uses only forensic seeds `20260920..20260925`.

The untouched acceptance family `20261001..20261006` remains sealed.

## Branch after overlay

- If Stage-B AveragePolicy and Advantage are both farther from low-noise reference and K4 fixes estimator error: consider one bounded K4 pilot.
- If AveragePolicy worsens but Advantage does not: investigate strategy-memory/AveragePolicy accumulation.
- If K4 does not improve estimator quality on the actual failure states: reject K4 as causal fix.
- Regardless, separately account for the PassiveCaller FLOP regression before reopening long training.

## Immediate action

Run `bash tools/run_lt2_jammer_facing_allin_common_reference_v2.sh`. Stop at PASS or first error. Do not train K4 first.
