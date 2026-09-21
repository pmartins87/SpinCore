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
5. 8200 end-to-end semantic + whole-iteration throughput gate — **PASS EXACT; WHOLE-ITERATION 1.530x**.
6. Resume from 8200 with 4x8 to **9105** (+905 iterations, projected 20.998 h) — **RUNNING**.
7. Preserve iteration 8600 automatically as an internal raw comparison checkpoint while the same precommitted run continues to 9105.
8. After PASS, create a derived finalized evaluation copy of raw 8600, then compare 8100 / 8600 / 9105 on the LT3 development battery without touching the LT3 sealed holdout.
9. External-strength lane: finish DeepCrusher DC0 oracle/source-runtime fidelity against frozen R8 v22 before making any canonical "beats DeepCrusher" claim.
10. Run DC1 mechanical paired smoke (1k–5k sampled states), then DC2 qualification (>=100k paired sampled states, extend only if precision requires it).
11. Continue training only if the development battery and external-strength evidence justify more compute.
12. Freeze the final LT3 research candidate.
13. LT3 sealed holdout — **ONLY AFTER ALL RESEARCH CHOICES ARE FROZEN**.
14. Deployment promotion — **ONLY AFTER SEALED HOLDOUT PASS**.

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
- end-to-end semantic/throughput gate from 8200: **PASS EXACT**;
- whole-iteration speedup: **1.530x** (123.932 s -> 80.992 s on shared 8201);
- parallel median over 8201..8203: **81.108 s**;
- checkpoint-amortized planning time: **83.130 s/iteration**;
- precommitted ~21-hour endpoint: **9105** (+905 iterations), projected **20.998 h**;
- long continuation 8200 -> 9105: **RUNNING (started 2026-09-21 13:11 local)**.


### External competitiveness / DeepCrusher interpretation

Iteration count alone is not a valid strength estimator. Earlier Stage-A vs
Stage-B cross-play did not establish a monotonic strength-vs-iteration curve:
independent runs changed direction and confidence intervals included zero.

The canonical DeepCrusher benchmark therefore uses measured paired chip EV, not
iteration count. The currently frozen external opponent is DeepCrusher R8 v22.
A canonical qualification claim requires the DC0 faithful oracle gate first.

DC2 qualification success requires:
- overall SpinCore-minus-DeepCrusher paired chip EV > 0;
- 95% CI lower bound > 0;
- no result driven solely by one isolated blind/position while a major domain
  collapses;
- no material illegal-action/action-translation rate.

Until those measurements exist, do not state a crossover iteration at which
SpinCore "starts beating DeepCrusher". Once 8100 / finalized-8600 / 9105 are
benchmarked under the same frozen seeds, the observed results may bound a
crossover interval, but they still do not justify assuming monotonic improvement.
