# SpinCore Roadmap — active state 2026-09-21

## Primary objective now

**BUILD A CLEAN LT3 TRAINING LINE FROM ITERATION 0.**

OpenHoldem deployment is paused until the user explicitly returns to it.

## LT2 production baseline

- ENS8@8100 strategic candidate — **FROZEN / HOLDOUT PASS**;
- Python deployment parity — **PASS EXACT**;
- native C++ inference parity — **PASS**;
- hidden filler invariance — **PASS**;
- exact public-transcript rebuild — **PASS**;
- OpenHoldem observable tracker — **PASS**;
- native C++ OpenHoldem tracker — **PASS**;
- native shadow decision engine — **PASS**;
- Windows x64 shadow DLL mock-host gate — **PASS**.

LT2 artifacts remain read-only while LT3 research proceeds.

## LT3 research/training lane — ACTIVE

1. LT3 H1 plan preregistered — **PASS / READY**.
2. Sequential LT3 continuation from 8100 — **ABORTED / HISTORICAL ONLY**.
3. ENS8 exact-parity throughput matrix — **NEXT**.
4. Freeze the clean-from-zero training schedule — **AFTER PERFORMANCE MATRIX**.
5. LT3 clean rebuild from iteration 0 — **AFTER SCHEDULE FREEZE**.
6. Development-set comparisons at preregistered milestones versus preserved LT2 baselines.
7. Continue only while learning evidence justifies more compute.
8. Freeze the final LT3 candidate.
9. LT3 sealed holdout — **ONLY AFTER ALL RESEARCH CHOICES ARE FROZEN**.
10. Deployment promotion — **ONLY AFTER SEALED HOLDOUT PASS**.

### Clean rebuild rationale

The preserved LT2@8100 candidate remains a valid, holdout-passed baseline, but
its training lineage is mixed:

- iterations 1..7500 used HU fresh100;
- iterations 7501..8000 used HU fresh400 with a single current model;
- iterations 8001..8100 introduced online ENS8 fresh400 behavior.

The later causal audit showed that HU fresh100 was insufficient at the mature
Stage-B reservoir.  The Stage-B reservoir itself was not shown to be poisoned,
which is why continuation was a defensible repair path.  However, continuation
does not answer the stronger question: what happens when the corrected schedule
is used from the start?

Therefore LT3 will be a clean rebuild from iteration 0 after the execution
schedule and parallel fitting are frozen.  The abandoned partial continuation
checkpoint (8200) is preserved only as evidence and is not a source for LT3.

## OpenHoldem deployment lane — PAUSED

Preserved state:

- Windows shadow DLL mock-host gate — **PASS**;
- tested x64 DLL SHA256:
  `7566be1b3c73207d437171c2b4e94f6a94477786a2a48599994a647808030062`;
- actual OpenHoldem host architecture inspection — **PAUSED**;
- real OpenHoldem shadow load — **PAUSED**;
- real-table shadow gate — **PAUSED**;
- action-enabled integration — **NOT AUTHORIZED**.

Resume this lane only when the user explicitly asks to return to OpenHoldem
deployment.


## Mandatory performance gate before long training

No future multi-hour training block may start merely because the algorithmic
contract is correct.

Before any run expected to exceed 60 minutes, the training implementation must
pass a dedicated throughput gate on the target Ryzen host:

1. identify the dominant wall-time component;
2. benchmark obvious independent parallelism;
3. require exact or explicitly bounded numerical parity;
4. measure end-to-end speedup including serialization/snapshot overhead;
5. record CPU utilization and memory headroom;
6. only then freeze the execution plan for the long run.

For ENS8 fresh-member fitting specifically, sequential execution is not an
accepted final implementation unless the process-parallel benchmark fails exact
parity or provides no material speedup.

Current H1 status:
- sequential H1 launch: **ABORT / SUPERSEDED FOR PERFORMANCE REVIEW**;
- ENS8 parallel-fit exact-parity benchmark: **NEXT**;
- H1 restart from frozen LT2@8100: **ONLY AFTER PERFORMANCE GATE**.
