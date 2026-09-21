# SpinCore Current Work

Date: 2026-09-21
Status: **FINAL HU HOLDOUT PASS — ENS8@8100 STRATEGIC CANDIDATE FROZEN — DEPLOYMENT PARITY NEXT**

## Final sealed holdout

All 11 preregistered primary criteria passed.

Highlights:

- learned ecosystem ENS8 8100 absolute: +8.177, CI [+5.504,+10.850];
- ecosystem 8100−8000: +1.223, CI [-0.488,+2.933], non-inferiority PASS;
- ecosystem 8100−AVG8100: +12.608, CI [+9.528,+15.689];
- direct 8100 vs 8000: +0.510, CI [-1.231,+2.252], non-inferiority PASS;
- direct 8100 vs AVG8100: +7.195, CI [+4.342,+10.048];
- Uniform absolute +31.811;
- Passive absolute +10.288;
- Jammer absolute +13.996.

The holdout is closed. Do not rerun it or tune from it.

## Frozen HU candidate

TRUE_HEADS_UP current ENS8 @ iteration 8100.

Exact artifacts:
- checkpoint SHA256 `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`;
- ensemble sidecar SHA256 `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`.

## Deployment engineering

Strategic testing is finished for this candidate.

Next:
- export a compact hybrid deployment bundle;
- THREE_HANDED remains finalized AveragePolicy;
- TRUE_HEADS_UP uses the frozen current ENS8 raw-output ensemble;
- mechanically prove inference parity against source artifacts.

No roots, no EV gate and no holdout reuse.

## Immediate action

```bash
bash tools/run_lt2_hu_ens8_deployment_parity.sh
```

Wait for `LT2_HYBRID_DEPLOYMENT_PARITY_PASS`, then send
`SpinCore_LT2_hybrid_deployment_parity.json`.

Keep `SpinCore_LT2_hybrid_deployment_8100.pt` in Downloads.
