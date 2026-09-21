# SpinCore Roadmap — active state 2026-09-21

## Active status

- strategic candidate ENS8@8100 — **FROZEN / HOLDOUT PASS**;
- Python deployment parity — **PASS EXACT**;
- native C++ inference parity — **PASS**;
- hidden opponent/future-board filler invariance — **PASS**;
- exact public-transcript canonical rebuild — **NEXT**;
- OpenHoldem snapshot-to-transcript reconciler — **AFTER REBUILD PASS**;
- live user-DLL bridge — **AFTER RECONCILER**.

## Runtime reconstruction architecture

The authoritative solver remains the only source of observation/legal/action semantics.

At a Hero decision, reconstruct from scratch using:
- hand-start scenario;
- Hero hole cards;
- currently visible board;
- deterministic fillers only for hidden cards;
- exact public action transcript.

This avoids stale future-card fillers and avoids maintaining a second poker engine inside the DLL.

## Next gate

Prove exact state parity for transcript replay across old forensic trajectories.

After that, solve the remaining OpenHoldem-specific problem: reliably deriving the exact public transcript from successive scraped table snapshots, failing closed on ambiguity.
