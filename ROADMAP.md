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
9. External-strength lane: finish DeepCrusher DC0 oracle/source-runtime fidelity against frozen R8 v22 before making any canonical "beats DeepCrusher" claim. **Native/OpenPPL transitive Hold'em closure is now PASS with 0 unresolved leaves** (290 resolved; 47 syntactically dead Omaha/extra-card leaves). Transcript-derived history/raiser/caller symbols, exact current-board nhandshi enumeration and canonical 169-class R8 preflop multiplex equity are implemented. Remaining DC0 gates: exact action-origin/sizing translation, full oracle callback integration and real OpenHoldem parity fixtures.
10. Run DC1 mechanical paired smoke (1k–5k sampled states), then DC2 qualification (>=100k paired sampled states, extend only if precision requires it).
11. Canonical promotion training remains **UNPROVEN** by the development battery. The separate utilization continuation 9105 -> 10105 is **PASS**: +1000 iterations / +600,000 roots, 22.564 h, source 9105 unchanged, raw 9600 preserved, postvalidation PASS, no sealed holdout touched. Endpoint checkpoint SHA256 `f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0`; HU ENS8 sidecar SHA256 `8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d`. It remains research-only and is not evidence that more roots improve strength.
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


### DC0 executable runtime checkpoint — 2026-09-23

- [x] Frozen R8 source + OpenPPL library compile/closure.
- [x] Native/transitive Hold'em symbols fail-closed with zero unresolved leaves.
- [x] OpenHoldem lifecycle callbacks and persistent memory.
- [x] Exact OpenPPL action/sizing translator.
- [x] Executable `DeepCrusherR8Policy.choose_exact()`.
- [x] Exact solver application and decision provenance, including OpenHoldem `didrais` vs `didbetsize` action-origin preservation.
- [x] Compact benchmark bundle preserving the intended SpinCore hybrid semantics: 3H AveragePolicy + HU current ENS8.
- [x] Real-solver end-to-end runtime smoke over 3H/HU and all four streets.
- [x] Paired DC1 development runner + decision JSONL output.
- [ ] Real OpenHoldem parity fixtures covering preflop/flop/turn/river and representative sizing families.
- [ ] Freeze DC0 canonical PASS only after parity.
- [ ] DC1 1k–5k development smoke against the compact SpinCore policy.
- [ ] DC2 >=100k paired benchmark after DC1 mechanics and parity are clean.

## DC1 10105 first diagnostic smoke — 2026-09-23

The guarded 200-scenario DC1 DEVELOPMENT_ONLY smoke completed at source commit
`c8c3d64684814aa4ccdb1f824f2990ed787dbbd2` with 115 THREE_HANDED and 85
TRUE_HEADS_UP scenario clusters, 860 balanced games and 3,506 traced decisions.

Mechanical result:
- overall paired SpinCore-minus-DeepCrusher: **-19.993 chips/policy-seat-hand**,
  95% CI **[-55.266,+15.279]**;
- THREE_HANDED: **-28.562**, CI **[-56.770,-0.354]**;
- TRUE_HEADS_UP: **-8.400**, CI **[-82.315,+65.515]**.

These numbers remain non-canonical and do not authorize a strength claim: the
sample is intentionally small and the real-OpenHoldem DC0 parity fixture gate is
still pending.

The external sanity queue produced 29 review events: 28 deep postflop high-card
all-ins and one trips+ fold. Hand-level review split the 28 high-card jams into
9 with an immediate straight/flush draw and 19 without an immediate
straight/flush draw. The trips flag is Qs8d on 8s8c4c, facing 60 into a 120-chip
pot at about 7.13bb effective. This does not prove a policy defect by itself,
because SpinCore is stochastic and exact ALL_IN can also be the resolved result
of a pot-size action near commitment.

**Gate decision:** do not scale DC1 yet. First make SpinCore traces expose the
sampled universal action slot, its probability, the full legal probability
vector, the RNG draw and the resolved exact action. This distinguishes a genuine
ALL_IN policy choice from a POT_33/POT_50/POT_75/POT_100 slot that collapses to
all-in under the legacy 60% near-commitment resolver, and distinguishes a
meaningful trips-fold probability from a tiny sampled tail.

Instrumentation is now on main:
- `e867b6679b98cfc35a7255030d4a1b5f2f26938c` — sampled SpinCore
  slot/probability metadata;
- `399b1a635d18dfb172cd71303fdf3e9a2be347b0` — sanity examples now include
  policy metadata and immediate straight/flush draw outs;
- `9974b3c9d39cb6995632209708dd978b1c5a8eb0` and
  `b0ce0704eb156990b899139ab6bd5d45bc0cf11f` — regression coverage.

Next local gate: rerun the exact same guarded 200-scenario smoke on the new main
and inspect the same 29 decisions with their selected probabilities before
authorizing any 1k-5k DC1 scale-up.

## DC1 10105 instrumented reproducibility rerun — 2026-09-24

The instrumented rerun of the exact same 200-scenario DC1 smoke completed at
git head `01e42844ecb877e9a4e845c9049954b2cbe6a006` using the same frozen 10105
checkpoint, matched HU ENS8 sidecar and seed 20260923.

Reproducibility gate: **PASS EXACT at aggregate/sanity level**.
- traces: 3,506;
- SpinCore decisions: 1,803;
- overall paired SpinCore-minus-DeepCrusher: **-19.993** chips/policy-seat-hand,
  CI95 **[-55.266,+15.279]**;
- sanity queue: **28 POSTFLOP_DEEP_HIGH_CARD_JAM + 1 POSTFLOP_TRIPS_PLUS_FOLD**.

These values exactly match the pre-instrumentation smoke, so the added decision
metadata did not alter the benchmark trajectory or RNG behavior.

The runner also successfully printed an Explorer-ready UNC path and copied the
bundle to the Windows Desktop:
`C:\Users\Rz9\Desktop\SpinCore_DC1_10105_smoke_bundle.zip`.

Next gate remains unchanged: inspect the new bundle's per-decision
`policy_detail` for the 29 flagged actions before any 1k-5k DC1 scale-up.

## DC1 instrumented 29-case audit — 2026-09-24

The uploaded instrumented bundle was inspected decision by decision.

### Trips fold

The sole trips fold was not a translation artifact and not the policy's modal
choice.  At Qs8d on 8s-8c-4c, facing 60 into 120 at ~7.13bb effective, the 3H
AveragePolicy distribution was:
- FOLD **7.3276%**;
- CHECK_CALL **60.0245%**;
- ALL_IN **32.6479%**.

The sampled RNG draw was 0.052933, so the 7.33% fold tail happened to be
selected.  Thus the policy continued **92.67%** of the time in that exact state;
the evidence does not support the hypothesis that the model globally regards
trips as a losing hand.

Across all 17 exact-TRIPS SpinCore postflop decision states in this 200-scenario
smoke, the sum of current fold probabilities was 0.4058 expected folds and one
fold was observed.  Under those heterogeneous current probabilities, the chance
of at least one trips fold in the 17 observed states is ~34.46% (exactly one
~28.86%).  The single observed fold is therefore unsurprising conditional on
the current policy, although the existence of a 7.33% fold tail may still be a
coverage/generalization leak.

### High-card all-ins

All **28/28** flagged high-card all-ins came from the literal SpinCore
`ALL_IN` universal slot.  None was a POT_33/POT_50/POT_75/POT_100 sizing that
collapsed to all-in through the 60% near-commitment resolver.

Among all 141 SpinCore postflop HIGH_CARD decision opportunities at >=10bb
effective in this smoke:
- observed ALL_INs: **28**;
- sum of the policy's ALL_IN probabilities: **29.3663** expected sampled all-ins.

For the 99 such states with no immediate straight/flush draw:
- observed ALL_INs: **19**;
- expected from policy probabilities: **21.0772**;
- mean ALL_IN probability: **21.29%**.

Therefore the count of high-card jams is not an unlucky RNG realization.  It is
a real feature of the current policy.  That still does not make the jams
automatically wrong: bluff jams can be strategically correct and require
counterfactual EV/range context.  Some individual states are strong review
targets, including HU J5o on 3c-8c-Qs-Ks facing a turn raise where the current HU
ENS8 assigns ALL_IN probability 1.0.

### Gate decision

Do **not** patch strategy or restart training from these flags.  Also do not
scale directly to 5k DC1 yet.  The user's low-frequency/coverage hypothesis is
now the next falsifiable gate.

A guarded 10105 reservoir audit has been added:
- `tools/audit_lt3_10105_trip_coverage.py`;
- `tools/run_lt3_10105_trip_coverage_audit.sh`.

It reads the frozen 3H Algorithm-R strategy and advantage reservoirs, counts
TRIPS and the specific paired-board/one-hole-card trips morphology, estimates
their prevalence in the complete seen sample streams with Wilson intervals, and
measures historical strategy-target fold mass in those trip states.  This is a
cheap diagnostic and does not train or alter the checkpoint.

## LT3 10105 trips coverage audit — 2026-09-24

The frozen 10105 3H reservoirs were audited directly.  The low-frequency
hypothesis is **not supported at the broad trips-class level**.

Strategy reservoir:
- 2,000,000 retained Algorithm-R samples from 7,482,676 total seen;
- TRIPS: 11,166 retained, implying ~41,776 strategy decision samples in the
  full stream (Wilson95 ~41,010..42,556);
- paired-board + one-hole-card trips: 6,702 retained, implying ~25,074 full-
  stream decision samples (Wilson95 ~24,482..25,681).

Advantage reservoir:
- 2,000,000 retained samples from 102,908,714 total seen;
- TRIPS: 59,004 retained, implying ~3.036 million advantage decision samples;
- paired-board + one-hole-card trips: 34,431 retained, implying ~1.772 million
  advantage decision samples.

These are **decision samples, not unique poker hands**.  They nevertheless rule
out explanations like "the learner only saw roughly ten trips states" for the
class as a whole.

Historical strategy targets also show that fold mass is not unique to the
benchmark hand.  Among 3,422 retained paired-board-trip samples where fold was
legal:
- mean fold target: 8.393%;
- iteration-weighted mean: 8.246%;
- median: 0%;
- 709 (20.72%) had fold target >=5%;
- 693 (20.25%) had fold target >=10%.

The observed Q8o/884 current-policy fold probability of 7.328% is therefore
close to the historical class-level mean, not an obvious isolated RNG artifact.
However the broad class mixes flop/turn/river, prices, stack depths and histories,
so this does **not** yet prove that the 7.33% tail is justified in the exact
flop geometry.

Gate decision: no strategy patch and no retraining yet.  A narrower local-
geometry audit is now required before judging the fold tail.

Added guarded diagnostics:
- `tools/audit_lt3_10105_trip_local_geometry.py`;
- `tools/run_lt3_10105_trip_local_geometry_audit.sh`.

The local audit progressively narrows the 3H Algorithm-R reservoirs around the
actual benchmark state (flop paired-board trips, live_count=2, half-pot price,
5-10bb hero stack, ~4bb pot/~2bb call/current-bet, dealer_rel=1, then Q kicker
and finally Q8/884 rank morphology) and breaks strategy fold targets into
<=8100, 8101-9105 and 9106-10105 cohorts.  This tests whether the fold mass is
an old AveragePolicy residue, a locally persistent target, or a sparse-neighbor
generalization effect.

## Correction to first local-geometry audit — 2026-09-24

The first V1 local-geometry report is **invalid from subset B onward** because
the diagnostic script interpreted SPNNIV1 categorical `live_count` as the
number of players still contesting the current pot.  That is not the frozen V1
semantic: `live_count` is the topology seat count for the hand and remains 3
after a player folds.  Fold state is carried separately in the actor-relative
`statuses` vector.

The uploaded V1 report therefore produced:
- valid subset A (flop paired-board trips facing action);
- artificial zero counts for B..H due to the incorrect `live_count==2` filter.

This is a diagnostic-script bug only.  It is **not evidence of missing training
coverage** and does not affect the frozen checkpoint or benchmark.

The exact benchmark hand had THREE_HANDED topology `live_count=3`, with actor-
relative statuses `(0,1,0)`: hero active, dealer folded preflop, remaining
opponent active.  The local audit has been corrected to V2 at commit
`1e5b918539d6f331fc507da5f32eddae8e2f5bd9`.

The valid V1 subset-A result remains useful: 2,152 retained strategy samples of
flop paired-board trips facing action, with fold-target mean 9.13%.  Recent
9106..10105 samples did **not** show the fold mass disappearing: 185 retained
samples, mean fold target 9.45%, compared with 9.28% for <=8100 and 7.72% for
8101..9105.  This argues against a simple "old AveragePolicy residue only"
explanation at the broad flop-paired-trips level, but the corrected V2 narrow
geometry audit is still required.

## Corrected V2 local trips geometry audit — 2026-09-24

The corrected V2 audit confirms that the broad trips class is well represented
but the actual benchmark neighborhood is extremely sparse.

Strategy Algorithm-R reservoir (2,000,000 retained / 7,482,676 seen):
- A: flop paired-board trips facing action: 2,152 retained, ~8,051 full-stream;
- B: same, 3-seat topology with one opponent already folded: 370 retained,
  ~1,384 full-stream, mean fold target 7.03%;
- C: plus ~half-pot price: 89 retained, ~333 full-stream, mean fold target 9.68%;
- D: plus 5-10bb hero stack: **2 retained**, ~7.48 full-stream
  (Wilson95 ~2.05..27.29);
- E: plus near-target pot/call/current-bet geometry: **2 retained**, ~7.48
  full-stream, both <=8100 and both fold target 0;
- F: plus target dealer-relative position: **1 retained**, ~3.74 full-stream,
  <=8100 and fold target 0;
- G: plus Q kicker: **0 retained** (95% upper full-stream estimate ~14.37);
- H: exact Q8/884 rank pattern: **0 retained** (same upper bound).

Advantage Algorithm-R reservoir (2,000,000 retained / 102,908,714 seen):
- A: 4,224 retained, ~217k full-stream;
- B: 488 retained, ~25.1k;
- C: 79 retained, ~4,065;
- D/E: only **2 retained**, ~103 full-stream each;
- F: **1 retained**, ~51;
- G/H: **0 retained** (95% upper full-stream estimate ~198).

Interpretation:
- the user's original hypothesis is wrong only in its broad form: trips itself
  is not rare in training;
- it is directionally correct for the *specific decision neighborhood*.
  The final 3H AveragePolicy has almost no direct recent strategy-target
  coverage near Q8o/884 at ~7bb facing ~half pot;
- the two near-target strategy samples both predate iteration 8100 and carry
  fold target 0, while broader recent half-pot neighbors can carry substantial
  fold mass.  The observed 7.3276% fold tail is therefore more consistent with
  neural generalization/interpolation under sparse local coverage than with a
  directly learned local fold target;
- simply adding another 1000 iterations under the same sampling distribution is
  unlikely to fix this neighborhood efficiently.  At the observed strategy-
  sample prevalence, it would add on the order of <1 near-target strategy
  sample in expectation; this is an extrapolation, not a training guarantee.

Do not patch trips or restart long training yet.  The next discriminating gate
is whether the *current iteration-10105 3H Advantage network* already assigns
near-zero fold at the exact benchmark state while the finalized AveragePolicy
still assigns 7.33%.  If so, the bottleneck is primarily AveragePolicy
distillation/generalization.  If current Advantage also carries fold mass, the
problem lies deeper in advantage coverage/representation/generalization.

Guarded exact-state replay added:
- `tools/audit_dc1_trip_policy_vs_current_advantage.py`;
- `tools/run_dc1_trip_policy_vs_current_advantage.sh`.

The replay is pinned to DC1 scenario 86 / lineup 2 and fails unless it exactly
reproduces the known AveragePolicy fold probability and sampled FOLD action.
It evaluates current 3H Advantage on the same observation without consuming RNG
or changing the hand trajectory.  Current Advantage remains diagnostic only.

## Exact Q8/884 AveragePolicy vs current Advantage — 2026-09-24

The guarded replay reproduced the original DC1 trips-fold state exactly and
separated the deployed 3H AveragePolicy from the iteration-10105 current
Advantage policy on the very same observation.

Exact state:
- Qs8d on 8s-8c-4c;
- blind 15/30;
- pot 120, to-call 60;
- hero stack 214;
- sampled AveragePolicy RNG draw 0.0529332285 reproduced the original FOLD.

3H AveragePolicy (actual DC1 behavior):
- FOLD **7.3276%**;
- CHECK_CALL **60.0245%**;
- ALL_IN **32.6479%**;
- argmax CHECK_CALL.

Current 3H Advantage @10105 on the identical observation:
- FOLD **0.0000%**;
- CHECK_CALL **41.7595%**;
- ALL_IN **58.2405%**;
- argmax ALL_IN;
- raw outputs: FOLD -0.009634, CHECK_CALL +0.018504, ALL_IN +0.025807.

Total-variation distance between AveragePolicy and current Advantage on this
state is **0.255926**.

Interpretation:
- the latest 3H Advantage signal does **not** consider fold a positive-regret
  action in the exact trips state; regret matching removes FOLD completely;
- therefore the observed 7.33% fold tail is not evidence that the current
  iteration-10105 Advantage learner itself currently prefers or even mixes fold
  there;
- combined with the V2 local-coverage audit (only 2 retained strategy samples
  near the target geometry, both old and target-fold 0; zero retained Q-kicker
  neighbors), the fold tail is now most consistent with the historical
  AveragePolicy / strategy-memory generalization layer under sparse local
  coverage;
- this is **not yet authorization to deploy current Advantage instead of
  AveragePolicy**. Deep CFR intentionally distinguishes current behavior from
  its time-averaged strategy, and the current 3H Advantage is a single fresh
  estimator rather than a validated ensemble.

The next gate is therefore not more roots and not a trips hardcode.  Audit the
finalized 3H AveragePolicy against its own strategy-memory targets to determine
whether the issue is:
1. an underfit/distillation problem in the AveragePolicy network;
2. faithful fitting of a historical average that genuinely contains fold mass;
3. a broader policy-vs-current drift pattern.

Added guarded diagnostics:
- `tools/audit_lt3_10105_average_policy_fit.py`;
- `tools/run_lt3_10105_average_policy_fit_audit.sh`.

The audit reports the actual checkpoint `policy_steps`,
`policy_optimizer_steps`, strategy-reservoir size/seen count, and compares
stored strategy targets vs final AveragePolicy on a fixed 50k uniform reservoir
sample plus the progressively narrowed trips subsets.  It also reports
AveragePolicy-vs-current-Advantage drift on the same states.  It performs no
training and does not modify the checkpoint.

## 10105 AveragePolicy fit audit — 2026-09-24

The finalized THREE_HANDED AveragePolicy was audited directly against a fixed
uniform sample of its own 2,000,000-item strategy reservoir.

Checkpoint/training facts:
- policy_steps per finalization config: 4,000;
- batch size: 1,024;
- cumulative policy_optimizer_steps recorded in checkpoint: 32,000;
- strategy reservoir: 2,000,000 retained / 7,482,676 seen.

Global 50k reservoir sample:
- target fold mean: 23.575%;
- AveragePolicy fold mean: 23.928% (good marginal calibration);
- target-vs-AveragePolicy TV mean: **0.4277**;
- weighted TV mean: **0.4272**;
- median TV: 0.4303;
- p95 TV: 0.7712;
- argmax mismatch: **48.72%**;
- fold MAE: 0.2319.

AveragePolicy vs current 3H Advantage is also far apart globally:
- mean TV 0.4435;
- median 0.4488;
- p95 0.7815.

Trips subsets retain the same pattern: broad marginal fold rates can be close
while statewise target-vs-policy TV remains large (A: TV 0.3733, argmax mismatch
40.20%; B: TV 0.4709, mismatch 56.49%; C: TV 0.4566, mismatch 53.93%).

Near the exact Q8/884 geometry, the only retained E/F sample has historical
target Fold 0, final AveragePolicy Fold 7.287%, and current Advantage Fold 0.
This is directionally consistent with the exact benchmark state where final
AveragePolicy Fold is 7.3276% and current Advantage Fold is 0.

Important interpretation constraint:
these per-sample target-vs-policy distances do **not by themselves prove
undertraining**.  Strategy-memory targets are nonstationary historical behavior
targets; identical or nearby observations can legitimately carry conflicting
targets across iterations.  A deterministic AveragePolicy must approximate the
weighted conditional average, so some per-sample TV is irreducible.

The next discriminating gate is a shadow strategy-only refit of a copy of the
final 10105 AveragePolicy on the current frozen strategy reservoir with a fixed
50k holdout and all trips-local samples excluded from training.  If additional
policy-only optimizer steps materially improve holdout cross-entropy/TV and the
exact Q8/884 fold probability, the final AveragePolicy is underfit.  If not, the
remaining mismatch is more consistent with historical-target conflict, model
capacity or representation/generalization.

Added:
- `tools/audit_lt3_10105_average_policy_shadow_refit.py`;
- `tools/run_lt3_10105_average_policy_shadow_refit.sh`.

The shadow audit trains only a copied model/optimizer at extra-step milestones
0/500/1000/2000/4000.  It never mutates the checkpoint and does not authorize
deployment of the shadow model.

## 10105 AveragePolicy shadow-refit result — 2026-09-25

The fixed-holdout strategy-only shadow refit rejects the simple hypothesis that
the finalized THREE_HANDED AveragePolicy merely needed more of the same fitting.

Starting from the exact frozen 10105 policy/Adam state and excluding both a
fixed 50k holdout and all 2,152 trips-local samples from shadow training:
- baseline weighted holdout cross-entropy: **1.0839426**;
- +500 steps: **1.0847677**;
- +1000: **1.0854933**;
- +2000: **1.0864551**;
- +4000: **1.0884514**.

Thus +4000 extra strategy-only steps worsened the fixed holdout weighted CE by
**+0.0045088**.  Holdout TV changed only 0.426801 -> 0.427156 and argmax
mismatch worsened 48.616% -> 48.954%.

The exact Q8/884 fold probability also did not converge toward the local
historical/current-Advantage target of zero:
- 0 extra: 7.3276%;
- +500: 7.2878%;
- +1000: 10.6315%;
- +2000: 10.5706%;
- +4000: 8.2991%.

Therefore:
- "just increase policy_steps" is **not supported** as a repair;
- the original final policy is at least as good as this continuation under the
  current objective/optimizer on a fixed held-out reservoir sample;
- the large per-sample target-vs-policy distances are now more likely to contain
  substantial irreducible historical target conflict and/or representation/
  capacity/generalization effects rather than simple optimizer underfitting.

This still does not prove that AveragePolicy is strategically correct.  It only
localizes the failure mode.

Next gate: quantify how much target disagreement exists for identical frozen V1
observations, then repeat after canonicalizing only absolute suit labels.  This
separates historical/nonstationary conflict from policy approximation error and
tests whether SPNNIV1 wastes coverage/capacity on physically equivalent suit
labels.

Added:
- `tools/audit_lt3_10105_strategy_target_conflict.py`;
- `tools/run_lt3_10105_strategy_target_conflict.sh`.

The audit scans the full retained 3H strategy reservoir, reports exact-V1 and
suit-canonical duplicate prevalence, samples up to 20k repeated-state groups,
compares each group's iteration-weighted empirical conditional target mean to
the final AveragePolicy, measures within-group historical target disagreement,
and measures V1 policy variation across suit-equivalent representatives.
No training or checkpoint mutation occurs.

## 10105 strategy-target conflict / suit-canonical audit — 2026-09-25

The full retained THREE_HANDED strategy reservoir was grouped by frozen V1
observation identity and then by the same observation after canonicalizing only
physical suit labels.

EXACT_V1:
- 2,000,000 retained items;
- 1,996,682 unique hashed keys;
- only 2,754 duplicate groups / 6,072 repeated items (**0.3036%** of reservoir);
- max multiplicity 6;
- within-group sample-target -> iteration-weighted conditional-mean TV:
  **0.31974**;
- historical sample argmax disagreement within group: **34.08%**;
- conditional-mean target -> final AveragePolicy TV: **0.36430**;
- conditional-mean argmax mismatch vs AveragePolicy: **53.23%**;
- mean iteration span inside repeated groups: ~3,730 iterations.

SUIT_CANONICAL_V1:
- 11,872 duplicate groups / 29,264 repeated items (**1.4632%**), a 4.82x
  increase in repeated-item coverage;
- max multiplicity 28;
- within-group target conflict TV: **0.32804**;
- conditional-mean target -> AveragePolicy TV: **0.35642**;
- AveragePolicy prediction variation across suit-equivalent representatives:
  only **0.00752 TV**.

Interpretation:
- historical strategy-target conflict is real and large.  This is expected to be
  dominated by temporal current-policy drift: sampled strategy targets are
  produced directly from the iteration's current V1 behavior on the same V1
  observation/legal set, so identical V1 observations can acquire different
  targets as the current Advantage policy changes across thousands of
  iterations;
- the audit does **not** show that absolute suit labels are the main source of
  the weird actions. Canonicalizing suits improves repeated-state coverage
  materially, but the learned V1 policy is already almost suit-invariant on
  those groups (mean prediction variation TV ~0.0075);
- the reservoir is overwhelmingly sparse at exact-state level even after suit
  canonicalization, so the final policy necessarily depends heavily on neural
  generalization;
- final AveragePolicy remains materially separated from the empirical
  conditional target mean on the repeated groups, but most exact groups have
  only 2-4 retained samples.  Therefore this metric alone cannot be interpreted
  as a clean capacity failure;
- combined with the shadow-refit result, "more of the same policy optimizer
  steps" remains rejected as the next repair.

Historical repository evidence already documents material SPNNIV1 limitations
(lossy public history, padding-sensitive GRU, absolute suit/order redundancy)
and an existing SPNNIV3/H2/H3 research path.  However richer representations
previously failed to demonstrate a robust production improvement, so this audit
does not by itself authorize reopening a broad representation migration.

The next practical gate returns to the actual weird-action surface: replay the
same 200-scenario DC1 trajectory and compare 3H AveragePolicy to the current
iteration-10105 Advantage on **all** SpinCore 3H decisions, especially the
high-card jams and trips fold.  This determines whether the latest current
learner already removes a broad class of suspicious AveragePolicy actions or
whether the high-card aggression is present in both.

Added:
- `tools/export_3h_current_advantage_probe.py`;
- `tools/evaluate_dc1_3h_average_vs_current_advantage.py`;
- `tools/run_dc1_3h_average_vs_current_advantage.sh`.

The replay keeps the actual hybrid played policy unchanged
(3H AveragePolicy + HU ENS8), consumes the exact same benchmark RNG for played
actions, and records current 3H Advantage only as observational metadata.

## DC1 3H AveragePolicy vs current Advantage — 2026-09-25

The fixed 200-scenario / 860-balanced-game DC1 trajectory was replayed with
unchanged played semantics (3H finalized AveragePolicy + HU validated ENS8).
The iteration-10105 current 3H Advantage model was evaluated only as
counterfactual metadata on every SpinCore 3H decision.

Global 3H disagreement is very large:
- 1,557 SpinCore 3H decisions;
- AveragePolicy vs current-Advantage mean TV: **0.46166**;
- median TV: **0.47057**;
- p95 TV: **0.77767**;
- argmax disagreement: **67.89%**.

Therefore the Q8/884 anomaly is not occurring in an otherwise nearly identical
current-vs-average policy pair.  The two 3H policy objects encode materially
different behavior over the actual DC1 trajectory.

Trips anomaly:
- AveragePolicy Fold probability on Q8 / 8s8c4c: **7.3276%**;
- current 3H Advantage Fold probability: **0%**;
- current policy: **41.76% CHECK_CALL / 58.24% ALL_IN**.
This specific fold is therefore localized to AveragePolicy
time-averaging/distillation/generalization rather than the final current
Advantage signal.

High-card jam anomaly does **not** localize to AveragePolicy:
- 23 3H high-card-jam flags;
- AveragePolicy mean ALL_IN probability: **29.66%**;
- current Advantage mean ALL_IN probability: **39.85%**;
- current median ALL_IN probability: **32.84%**;
- current Advantage assigns a lower ALL_IN probability than AveragePolicy in
  only 6/23 flags;
- current Advantage assigns exactly zero ALL_IN in only 4/23;
- ALL_IN is current-Advantage argmax in 9/23.

A direct parse of the flag rows shows 17/23 have no immediate straight/flush
draw.  Even in that no-immediate-draw subset, current Advantage mean ALL_IN is
~36.64% versus ~30.11% for AveragePolicy, with ALL_IN argmax in 7/17.  Thus the
broad weak/high-card aggression cannot be repaired merely by replacing the
AveragePolicy with the final single 3H Advantage model.

Several extreme current policies are produced by the lean regret-matching map
when only one predicted legal Advantage is positive.  Examples include:
- 73o on 9c2h5d: current ALL_IN = 100% from raw ALL_IN advantage ~+0.004997
  while the other legal outputs are negative;
- 96o on 3cKdQd: current ALL_IN = 100% from raw ~+0.006632;
- J8o on Qc4s9d: current ALL_IN = 100% from raw ~+0.013253.

The raw magnitude alone is not a confidence interval and cannot establish that
these decisions are wrong.  It does, however, make independent-fit uncertainty
the correct next diagnostic: a small sign change around zero can radically
change regret-matched action mass.

Next gate:
- fit eight independent fresh 3H Advantage models from the exact same frozen
  10105 3H Advantage reservoir using the checkpoint's own step/batch/lr
  contract;
- generate zero new CFR roots and mutate no source artifact;
- replay the exact same DC1 trajectory;
- compare AveragePolicy, actual final current single Advantage, and raw-Advantage
  ENS8;
- separately report all high-card jams, the no-immediate-draw subset, and the
  trips fold, including across-member action-probability dispersion.

Added:
- `tools/build_3h_ens8_diagnostic_probe_10105.py`;
- `tools/evaluate_dc1_3h_ens8_uncertainty.py`;
- `tools/run_dc1_3h_ens8_uncertainty.sh`.

This is a diagnostic uncertainty experiment only.  It does not promote a 3H
ensemble or alter DC1/DC2 qualification semantics.

## DC1 3H ENS8 uncertainty audit — 2026-09-25

The fixed 200-scenario / 860-balanced-game DC1 trajectory was replayed again
with unchanged played semantics. Eight independent THREE_HANDED AdvantageNets
were freshly fit from the exact same frozen 10105 Advantage reservoir using the
historical 100-step 3H budget. The original current 10105 Advantage, raw-
Advantage ENS8 and every independent member were observational only.

### Global result

Independent-fit uncertainty is large rather than incidental:
- 1,557 SpinCore 3H decisions;
- current-single vs raw-ENS8 mean TV: **0.53790**;
- median TV: **0.55070**;
- p95 TV: **1.0**;
- current-single vs raw-ENS8 argmax disagreement: **63.39%**.

AveragePolicy is also far from both:
- AveragePolicy vs current: mean TV **0.46166**, argmax disagreement **67.89%**;
- AveragePolicy vs raw-ENS8: mean TV **0.43151**, argmax disagreement **49.84%**.

This is direct evidence that fresh100 does not extract one stable 3H current
policy from the frozen mature reservoir.

### Trips fold

The Q8/884 fold remains specifically an AveragePolicy tail:
- AveragePolicy FOLD: **7.3276%**;
- original current Advantage FOLD: **0%**;
- raw-ENS8 FOLD: **0%**.

The raw-ENS8 policy on that state is ~75.59% CHECK_CALL / 24.41% ALL_IN.
Some individual fresh100 members fall into the repaired all-nonpositive fallback
and therefore show approximately one-third Fold, but their raw average still
makes Fold negative and removes it.  The observed benchmark Fold should not be
used as evidence that the current learner believes trips should fold.

### High-card jams

Raw-Advantage ENS8 does **not** eliminate the broad aggression:
- 23 3H flagged high-card jams;
- AveragePolicy mean ALL_IN: **29.66%**;
- original current Advantage: **39.85%**;
- raw-ENS8: **52.07%**, median **53.89%**;
- raw-ENS8 ALL_IN argmax: 13/23.

For the 17 flags with no immediate straight/flush draw:
- AveragePolicy mean ALL_IN: **30.11%**;
- original current: **36.64%**;
- raw-ENS8: **56.55%**, median **66.40%**;
- raw-ENS8 ALL_IN argmax: 10/17.

However this does **not** represent strong eight-model consensus:
- no flagged state had ALL_IN as argmax for all eight members;
- mean across-state member ALL_IN-probability standard deviation is ~**0.300**;
- direct postprocessing of the stored eight member policies gives only 8/23
  states with a >=5/8 ALL_IN argmax majority (7/17 in the no-immediate-draw
  subset);
- averaging member **policies** instead of raw Advantages gives mean ALL_IN
  ~**39.54%** over all 23 and ~**40.81%** over the 17 no-draw flags, much lower
  than the raw-ensemble 52.07% / 56.55%;
- raw-ensemble vs policy-mixture mean TV over the 23 flagged states is
  ~**0.3546**.

A key nonlinear example is 83o on K-7-4: raw-Advantage averaging produces
100% ALL_IN even though only one of the eight member policies has ALL_IN as its
argmax and the member-policy mixture assigns only ~13.8% ALL_IN.  This occurs
because averaging raw values can leave ALL_IN as the sole slightly-positive
action and the unchanged regret-matching map then converts that sign pattern
into 100% action mass.

Interpretation:
- the original final single 3H Advantage is demonstrably high-variance;
- the shared frozen reservoir still contains a broad aggressive signal, because
  the average of independent member policies retains substantial ALL_IN mass;
- but the extreme raw-ENS8 action probabilities are partly a nonlinear
  sign-threshold amplification and must not be read as eight-model confidence;
- therefore neither "AveragePolicy alone is broken" nor "all eight learners
  agree on the jams" is supported.

This is closely analogous to the previously diagnosed HU fresh100 failure, for
which a larger fit budget materially stabilized the mature-reservoir learner.
The 3H lane has never passed the equivalent budget-stability gate.

### Next gate — 3H Advantage budget stability

Before changing strategy, representation or benchmark scale, test whether the
historical **fresh100** 3H fit budget is itself insufficient.

Added:
- `tools/build_3h_advantage_budget_probe_10105.py`;
- `tools/evaluate_3h_advantage_budget_stability_10105.py`;
- `tools/run_3h_advantage_budget_stability_10105.sh`.

The gate fits the same eight deterministic replicas cumulatively to
**100 -> 200 -> 400 steps** on the exact same frozen 10105 3H Advantage
reservoir and snapshots each budget. It generates zero CFR roots and mutates no
source artifact. The exact same DC1 trajectory is then replayed and each budget
is compared on:
- global member pairwise TV and argmax disagreement;
- unanimous/majority argmax stability;
- raw-Advantage ensemble vs average-of-member-policies divergence;
- all high-card jams and the no-immediate-draw subset;
- the Q8/884 trips fold;
- per-action raw-sign agreement across members.

Decision rule:
- if instability falls materially by 200/400, 3H fresh100 is underfitting the
  mature reservoir and the next research intervention should target fit budget
  / ensemble mechanics;
- if instability remains high at 400, more optimizer steps alone are not the
  main repair and attention returns to representation / target ambiguity /
  training-game coverage.

