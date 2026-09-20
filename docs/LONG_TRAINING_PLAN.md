# SpinCore — Long-Training Plan

Status: **8000 FROZEN — ENSEMBLE MECHANISM BROAD-EV PASS — SIZE-8 REPLICATION ACTIVE**
Date: 2026-09-20

## What changed

The size-4 ensemble screen had weak statewise TV stabilization but strong aggregate frequency stabilization.

Broad EV now shows that both disjoint size-4 ensembles are strategically strong:

- both positive against all three baselines;
- both resolved improvements versus Stage B on all three;
- both improve Passive and Jammer versus 7600;
- neither has a resolved Uniform regression versus 7600.

Therefore the residual per-state TV is not, by itself, a failure.

## Remaining issue

Composition still matters:
- LEFT beats RIGHT significantly on Uniform and Jammer.

Selecting the better group after observing those outcomes would introduce selection bias.

## Next gate

Build two independent size-8 ensembles from sixteen deterministic fresh400 models.

Evaluate groups sequentially to keep worker memory bounded, but use identical deterministic scenario/deal/seat/RNG schedules so ENS8_B-minus-A remains paired.

Acceptance direction:
- both ensembles preserve the Stage-B repair;
- neither shows a resolved regression versus 7600 on any weak baseline;
- A/B composition difference should materially contract relative to ENS4 and preferably become unresolved.

Only then design an online ensemble-training pilot.

No new roots and no holdout yet.
