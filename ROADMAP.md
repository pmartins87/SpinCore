# SpinCore Roadmap — active state 2026-09-21

## Active status

- strategic candidate ENS8@8100 — **FROZEN / HOLDOUT PASS**;
- Python deployment parity — **PASS EXACT**;
- native C++ inference parity — **PASS**;
- hidden filler invariance — **PASS**;
- exact public-transcript from-scratch rebuild — **PASS**;
- public snapshot -> canonical exact action reconciler — **NEXT**;
- OpenHoldem heartbeat tracker + fail-closed cache — **AFTER RECONCILER PASS**;
- Windows user-DLL bridge — **AFTER TRACKER**.

## Runtime architecture now established

At Hero decision:
1. use hand-start scenario;
2. use actual Hero cards;
3. use actual visible board;
4. fill only still-hidden cards;
5. replay exact public voluntary transcript;
6. query authoritative solver observation/legal/exact sizing;
7. run frozen native deployment model.

## Remaining event-tracking problem

The DLL sees table snapshots, not authoritative action objects.

The tracker must infer one canonical exact action from each observed public transition and reject:
- skipped transitions;
- impossible stack/pot changes;
- domain drift;
- illegal exact actions;
- ambiguous alias semantics.

Canonical alias normalization follows the validated lean strategy convention.

After this gate passes, wire the reconciler into the actual OpenHoldem heartbeat/lifecycle callbacks.
