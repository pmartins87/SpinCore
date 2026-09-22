# SpinCore Roadmap — active state 2026-09-22

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
6. Resume from 8200 with 4x8 to **9105** (+905 iterations) — **PASS**. Completed in 20.303 h with source 8200 unchanged, raw 8600 preserved, final 9105 AveragePolicy finalized, and no holdout access.
7. Preserve iteration 8600 automatically as an internal raw comparison checkpoint — **PASS / IMMUTABLE**.
8. Post-9105 development battery — **PASS / INCONCLUSIVE; SEALED HOLDOUT UNTOUCHED**. Corrected execution completed on the frozen seed/protocol. AveragePolicy 8100 -> 9105 ALL = -0.055 chips/hand, 95% CI [-2.636,+2.526]; 3H = -2.303 [-4.841,+0.235]; HU = +2.630 [-2.151,+7.411]. Current HU ENS8 direct 8100 -> 9105 = -0.804 [-7.391,+5.784]. Policy drift is measurable, especially HU, but no resolved strength improvement was demonstrated.
9. External-strength lane: finish DeepCrusher DC0 oracle/source-runtime fidelity against frozen R8 v22 before making any canonical "beats DeepCrusher" claim. **Preparation is active in parallel with LT3 training**: expression semantics, ordered WHEN/SET control flow, canonical list parsing, hand-scoped user variables, persistent me_* memory semantics and full-source compile audit tooling are now implemented; native symbol coverage, exact action sizing and runtime-trace parity remain.
10. Run DC1 mechanical paired smoke (1k–5k sampled states), then DC2 qualification (>=100k paired sampled states, extend only if precision requires it).
11. Canonical promotion training remains **UNPROVEN** by the development battery, but the separate utilization continuation 9105 -> 10105 is **RUNNING HEALTHY** so the Ryzen is not idle while DC0 is engineered. Run directory: `runs/lt3_parallel_9105_10105/20260922_132104`. Preflight PASS; 9106/9107 completed at 80.12/77.10 s. This block is research-only, preserves 9105, saves raw 9600, adds 600,000 roots and touches no sealed holdout. Its existence must not be interpreted as evidence that more roots improve strength.
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
- long continuation 8200 -> 9105: **PASS**; elapsed 20.303 h; endpoint checkpoint SHA256 `21945e27c43c7e6c1cdb77018cd66dc90b3fab72c29a9034bb4a9f97cc0e6c68`; endpoint ENS8 sidecar SHA256 `b9c3ffffc7139eeb77c4b4182136e10ada5cad2f023aa6e560e164e2e0ac9256`.


### DeepCrusher DC0 preparation while LT3 trains

The R8 v22 benchmark lane is intentionally being built without touching the running trainer.

Completed foundation work:
- frozen R8 v22 operational source/hash pin and transitive dependency inventory;
- strict OpenHoldem-compatible expression evaluator;
- ordered OpenPPL WHEN / RETURN / SET control flow;
- physical-line continuation normalization for the frozen source;
- direct BetMax/BetPot/fractional-bet action token parsing;
- all 545 list sections parsed as canonical 169-class hand ranges;
- actor observable-state bridge now exposes the canonical hero hand class;
- OpenHoldem user_* variables persist for the hand and clear on hand reset;
- me_st/me_re/me_inc/me_add/me_sub memory semantics persist until connection reset;
- full frozen-source OpenPPL compile audit and a static-preparation runner added;
- exact R8 v22 operational source vendored in SpinCore with SHA256 pin so CI can test the real frozen artifact;
- full-source compile gate passes on the real 1,270,138-byte R8 source: 1,267 sections, 721 functions, 545 hand-list sections;
- strict primitive/native bridge plus frozen environment profile are active; pinned OpenPPL library overlay resolves 126 direct source dependencies that were previously misclassified as native; OpenHoldem-compatible card/hand provider raises direct provider coverage to 60 source identifiers; source-level direct unresolved dependencies are now 20;
- DC0 CI corrected to execute pytest-style contracts instead of merely importing files and is currently PASS.

Post-9105 development tooling is now frozen in `docs/LT3_POST9105_DEVELOPMENT_BATTERY_PROTOCOL_20260922.md` and `tools/run_lt3_post9105_dev_battery.sh`.

Still blocking canonical DC0:
- complete the **transitive** native/OpenPPL leaf provider. Current static closure: 336 native leaves, 201 resolved, 135 syntactically unresolved before pruning Hold'em-dead Omaha branches and freezing game/table constants;
- port exact action-history / raiser / caller semantics from the preserved OpenHoldem implementation;
- close substantive equity leaves, including `prwin/prtie` and the R8 backup-opponent-all-in-range multiplex symbols;
- exact DeepCrusher sizing/action conversion into ExternalExactAction;
- frozen environment treatment for optional PokerTracker/network/chair/log symbols — **PASS** via `GGPoker_NoPT_NoNotes_V1`;
- broad parity fixtures against real OpenHoldem traces, including preflop/flop/turn/river and sizing.

The historical R8 v22 OpenHoldem smoke proves the frozen artifact itself loaded and made 153 decisions, but the five original smoke logs are not stored in the DeepCrusher repository, so that audit document alone is not sufficient as an oracle parity fixture.

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


## Decision-level external audit — added 2026-09-22

The DeepCrusher benchmark now records enough state for hand-by-hand strategic review rather than relying only on aggregate chip EV. Each traced decision can include actor hole cards, visible board, pot, amount to call, stack/commitment geometry, exact action and exact amount.

The first sanity queue intentionally surfaces:
- pocket-aces preflop folds (critical);
- 72o preflop jams at >=10bb effective (high review; shallow jams are not auto-flagged);
- postflop top-pair folds (review, not automatically an error);
- trips+ folds and full-house-or-better folds;
- deep high-card all-ins (review because draws may justify them).

These flags are diagnostics, not poker-theory verdicts. Repeated patterns plus context are evidence; isolated flagged hands still require inspection.
