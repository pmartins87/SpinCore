# SpinCore Current Work

Date: 2026-09-21
Status: **INDEPENDENT LEARNED-POLICY CROSSPLAY PASS — ENS8 8100 FROZEN AS FINAL HU CANDIDATE — FINAL HOLDOUT AUTHORIZED**

## Independent design-set result

New preregistered seeds 20260926..20260930:

Historical learned ecosystem:
- ENS8 8000 absolute: +6.086;
- ENS8 8100 absolute: **+7.538**, CI [+4.575,+10.502];
- ENS8 8100 − ENS8 8000: +1.453, CI [-0.434,+3.339];
- ENS8 8100 − AVG 8100: **+13.059**, CI [+9.704,+16.414].

Direct:
- ENS8 8100 vs ENS8 8000: +1.227, CI [-0.711,+3.165];
- ENS8 8100 vs AVG 8100: **+12.663**, CI [+9.509,+15.817].

Current ENS8 8100 therefore passes the independent learned-policy design gate.

AveragePolicy remains lagged and is not the HU deployment candidate.

## Frozen candidate

TRUE_HEADS_UP current ENS8 @ iteration 8100.

Required artifact pair:
- ordinary iteration-8100 checkpoint;
- iteration-8100 `hu_ensemble_state.pt`.

No more training or member changes before final holdout.

## Final holdout

Protocol frozen in:
`docs/LT2_HU_ENS8_FINAL_HOLDOUT_PROTOCOL_20260921.md`

Seeds:
`20261001..20261006`

All primary acceptance criteria are committed before outcomes.

One completed holdout run is final; no tuning or rerun from its result.

## Immediate action

```bash
bash tools/run_lt2_hu_ens8_final_holdout.sh
```

Wait for `LT2_HU_ENS8_FINAL_HOLDOUT_COMPLETE`, then send
`SpinCore_LT2_hu_ens8_final_holdout.json`.

Do not train beyond 8100.
