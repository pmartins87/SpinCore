# SpinCore Roadmap — active state 2026-09-20

## Active status

- original Jammer FAI underfit — **REPAIRED**.
- mature single-model fit instability — **CONFIRMED**.
- more single-model optimizer steps — **REJECTED**.
- size-2 ensemble — **NOT USEFUL**.
- size-4 broad strategic EV — **PASS**, composition variance remains.
- replicated size-8 broad strategic EV — **PASS**.
- size-8 composition robustness — **MATERIALLY IMPROVED / NOT PERFECT**.
- isolated size-8 online-feedback pilot 8000→8100 — **NEXT**.
- original iteration-8000 source — **FROZEN**.
- holdout — **SEALED**.

## ENS8 evidence

Both independent ENS8 groups:
- are decisively positive against Uniform, Passive and Jammer;
- beat Stage B significantly on all three;
- significantly improve Passive and Jammer versus 7600;
- show no resolved Uniform regression versus 7600.

Composition B-minus-A:
- Uniform +1.94 unresolved;
- Passive +2.50 barely resolved;
- Jammer +2.16 unresolved.

This is materially tighter than the ENS4 split.

## Online intervention

Use ENS8_A by preregistration order, not by observed EV.

For HU each iteration:
- fit eight independent 400-step models on the same reservoir;
- reuse the exact predeclared ENS8_A init/batch seeds;
- average raw Advantage outputs;
- apply unchanged lean regret matching.

3H remains single fresh100.

Pilot only to iteration 8100. No generic continuation and no holdout yet.
