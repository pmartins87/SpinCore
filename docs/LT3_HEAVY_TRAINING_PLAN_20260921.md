# SpinCore — LT3 Heavy Training Plan

Date: 2026-09-21  
Status: **H1 READY — RESEARCH LANE ISOLATED FROM FROZEN LT2 PRODUCTION**

## Purpose

LT3 is a new research lane intended to test whether substantially more online
ENS8 training can improve beyond the frozen LT2 ENS8@8100 production candidate.

LT3 does **not** replace LT2 by construction.

The frozen LT2 production pair remains read-only:

- checkpoint:
  `/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/checkpoint.pt`
- checkpoint SHA256:
  `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`
- ENS8 sidecar:
  `/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/hu_ensemble_state.pt`
- sidecar SHA256:
  `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`

The LT2 final holdout is never reused for LT3 development decisions.

## H1 — heavy continuation 8100 -> 8600

H1 changes only training duration.

It deliberately preserves the exact mechanics that produced the accepted LT2
ENS8@8100 behavior:

- source iteration: 8100;
- target iteration: 8600;
- additional iterations: 500;
- roots per iteration: 600;
- new training roots: 300,000;
- expected cumulative roots: 5,160,000;
- THREE_HANDED: fresh100;
- TRUE_HEADS_UP: ENS8;
- HU member fits: 8 x fresh400 per iteration;
- HU optimizer steps per iteration: 3,200;
- new HU optimizer steps in H1: 1,600,000;
- K4: off / K=1;
- workers: 31;
- parent Torch threads: 8;
- checkpoint cadence: every 50 iterations.

The ordinary checkpoint and ENS8 sidecar are always treated as a mandatory
pair. The ordinary checkpoint alone is not the current HU behavior.

## Why H1 does not change architecture or budgets

The LT2 forensic work already established that single fresh HU models are
high-variance, whereas ENS8 stabilized aggregate behavior and produced the
accepted candidate.

Changing ensemble size, member fit budget, roots per iteration and training
duration simultaneously would make any improvement or regression hard to
attribute.

H1 therefore asks one question only:

**does substantially more of the already-validated ENS8 online-feedback process
continue to improve the research policy beyond iteration 8100?**

## Development-set preregistration

LT3 development seeds are reserved now, before H1 results:

- 20261101
- 20261102
- 20261103
- 20261104
- 20261105
- 20261106

They may be used after H1 for root-policy diagnostics, weak-baseline evaluation,
learned-policy ecosystem crossplay and direct seat-balanced comparison against
LT2 ENS8@8100.

They are not training seeds.

## Sealed LT3 holdout preregistration

The following seed literals are sealed and must not be used for any H1/H2
development decision:

- 20261121
- 20261122
- 20261123
- 20261124
- 20261125
- 20261126

They are reserved only for a future LT3 promotion candidate after all research
choices are frozen.

The historical LT2 final holdout seeds 20261001..20261006 remain retired and
must never be used as LT3 development data.

## H1 stop rule

H1 stops mechanically at iteration 8600.

Do not continue training from 8600 automatically.

After H1 completes:

1. verify source LT2 hashes are unchanged;
2. record H1 checkpoint + sidecar hashes;
3. run LT3 development-set forensic policy drift;
4. run paired weak-baseline evaluation;
5. run learned-ecosystem crossplay against LT2 ENS8@8100;
6. inspect AveragePolicy capture/lag;
7. decide whether H2 is justified.

No sealed LT3 holdout is opened during these steps.

## H2 — conditional only

H2 is not yet authorized.

If H1 is non-regressing on the preregistered development battery and shows
credible improvement, H2 may extend the same research lane further.

If H1 regresses materially, the correct action is to diagnose or terminate the
lane rather than spend more roots automatically.

## Production separation

LT2 remains the only deployment candidate while LT3 is under research.

The OpenHoldem Windows shadow-DLL work continues independently.

No LT3 artifact may be copied into the deployment bundle or DLL until a future
promotion protocol explicitly replaces LT2.
