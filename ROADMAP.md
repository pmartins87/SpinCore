# SpinCore Roadmap — active state 2026-09-21

## Primary objective now

**CONTINUE THE VALIDATED LT2/LT3 LEARNING STATE, AFTER THE ENS8 PERFORMANCE GATE.**

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
2. Sequential LT3 continuation from 8100 — **INTERRUPTED FOR PERFORMANCE OPTIMIZATION**.
3. Durable continuation checkpoint+sidecar @8200 — **PRESERVED**.
4. ENS8 exact-parity throughput matrix — **PASS; 4x8 SELECTED**.
5. 8200 end-to-end semantic + whole-iteration throughput gate — **NEXT**.
6. Resume from 8200 with 4x8 in a **~24-hour precommitted block** — **ONLY AFTER END-TO-END GATE PASS**.
7. Preserve iteration 8600 as an internal comparison checkpoint; final block target is derived from measured throughput and rounded to a checkpoint boundary before training starts.
8. Development-set adjudication compares the preserved 8600 milestone and the final ~24-hour endpoint.
9. Continue only while learning evidence justifies more compute.
10. Freeze the final LT3 candidate.
11. LT3 sealed holdout — **ONLY AFTER ALL RESEARCH CHOICES ARE FROZEN**.
12. Deployment promotion — **ONLY AFTER SEALED HOLDOUT PASS**.

### Continuation rationale

The Stage-B diagnosis did **not** support reservoir poisoning.  Instead, the
mature HU reservoir retained usable signal and the 100-step HU refit budget was
insufficient to extract it reliably.  HU400 then passed structural, broad and
online-feedback gates, and ENS8 stabilized the mature current HU behavior.

Therefore there is no evidence that the accumulated learning state must be
discarded.  Starting again from iteration 0 would be a separate expensive
research experiment, not a required repair.

The durable LT3 checkpoint+sidecar at iteration 8200 may be resumed **only
after** the parallel ENS8 implementation proves exact parity with the sequential
fit semantics.


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

Current LT3 status:
- sequential continuation reached a durable matched checkpoint+sidecar at **8200**;
- fit-only ENS8 matrix: **PASS; 4x8 exact-parity selected**;
- end-to-end semantic/throughput gate from 8200: **NEXT**;
- ~24-hour continuation from 8200: **NOT AUTHORIZED UNTIL THAT GATE PASSES**.
