# SpinCore — Long-Training Plan

Status: **8000 SOURCE FROZEN — ENS8 BROAD-EV PASS — ONLINE ENS8 PILOT ACTIVE**
Date: 2026-09-20

## Stabilization decision

The size-8 replication resolved the main composition concern sufficiently for an online pilot.

ENS8_A and ENS8_B are both strong on all three transparent baselines and both preserve the Stage-B/Jammer repair.

The B-minus-A composition gap contracts relative to ENS4:
- Uniform and Jammer become unresolved;
- Passive remains only narrowly resolved at +2.50 chips.

This is not proof of GTO quality, but it is enough to test the mechanism under actual Deep-CFR feedback.

## Anti-selection rule

ENS8_B had the higher observed mean EV, but production experimentation will not select it on that basis.

The pilot uses ENS8_A, the first predeclared group.

## Pilot contract

Source:
- preserved iteration 8000 checkpoint;
- exact SHA256 `773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886`.

Target:
- iteration 8100;
- +100 iterations.

3H:
- unchanged fresh100.

HU:
- eight fresh400 estimators from the same evolving HU reservoir;
- same eight predeclared ENS8_A init/batch seeds reused every iteration;
- mean raw Advantage output before unchanged lean regret matching;
- no K4.

RNG:
- ensemble minibatch RNG is isolated from the authoritative sampled-policy RNG.

Artifacts:
- isolated ordinary checkpoint;
- mandatory HU ensemble-state sidecar;
- pilot JSON report.

The ordinary checkpoint is not a complete representation of HU current behavior without the sidecar.

## After pilot

Do not unseal holdout immediately.

First evaluate the iteration-8100 current ENS8 behavior and AveragePolicy on the forensic baselines and trained-policy/cross-play diagnostics.

Only if the online feedback survives should the intervention be frozen for final holdout validation.
