# SpinCore Roadmap — active state 2026-09-21

## Primary objective now

**TRAIN LT3.**

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
2. LT3 H1 heavy continuation 8100 -> 8600 — **PAUSED / SUPERSEDED PENDING PERFORMANCE GATE**.
3. ENS8 exact-parity throughput benchmark — **NEXT**.
4. LT3 H1 optimized restart 8100 -> 8600 — **ONLY AFTER BENCHMARK PASS**.
5. LT3 H1 development-set adjudication — **AFTER OPTIMIZED H1 TRAINING PASS**.
6. LT3 H2 — **ONLY IF H1 EVIDENCE JUSTIFIES IT**.
7. Freeze final LT3 research choices.
8. LT3 sealed holdout — **ONLY AFTER FREEZE**.
9. LT3 deployment promotion — **ONLY AFTER SEALED HOLDOUT PASS**.

### H1 training contract

- source: exact LT2 ENS8@8100 checkpoint + sidecar;
- +500 iterations;
- +300,000 roots;
- target iteration 8600;
- 3H fresh100;
- HU ENS8 = 8 x fresh400;
- K4 off;
- 31 workers;
- Torch threads 8;
- no LT2 final-holdout reuse;
- no LT3 sealed-holdout access;
- no mutation of LT2 production artifacts.

Hard stop after H1:

`STOP HERE. Do not extend beyond 8600 before LT3 H1 development-set adjudication.`

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
