# SpinCore Current Work

Date: 2026-09-22
Status: **LT3 9105 -> 10105 UTILIZATION CONTINUATION PASS / POST-9105 BATTERY INCONCLUSIVE — DEEPCRUSHER DC0 EXECUTABLE / REAL-OPENHOLDEM PARITY PENDING**

## Strategic baseline

LT2 ENS8@8100 remains the frozen production baseline.

Its checkpoint, HU ENS8 sidecar, final holdout evidence and deployment artifacts
must remain read-only.

The previous OpenHoldem productionization work is preserved but **paused**.
No OpenHoldem host inspection, DLL installation or table testing is required
while LT3 training is the user's active priority.

## Primary active lane — LT3 continuation from preserved 8200

LT2 ENS8@8100 remains the frozen sealed-holdout-passed baseline.

The interrupted LT3 continuation has a durable matched checkpoint+sidecar at
iteration 8200.  It is preserved and remains eligible for continuation.

Why we are **not** restarting from zero:

- the Stage-B reservoir-poisoning hypothesis was not supported;
- controlled fresh refits showed usable signal remained in the mature reservoir;
- the material defect was insufficient HU fitting (fresh100), repaired by HU400;
- HU400 passed structural, broad and online-feedback validation;
- ENS8 then stabilized the mature HU current behavior and the 8100 candidate
  passed the pre-registered sealed holdout.

Thus there is no evidence that the accumulated 0..8100 learning state is
invalid.  A clean-from-zero run would be a separate research arm, not a required
repair.

The ENS8 fit bottleneck was resolved by the exact-parity 4x8 process-parallel implementation. The frozen 8200 -> 9105 continuation completed successfully. Source 8200 remained unchanged, raw 8600 was preserved, final 9105 was finalized, and neither the LT2 final holdout nor an LT3 sealed holdout was touched.

Operational scheduling constraint: after the performance matrix, do not default
to a short 8200->8600 run that is likely to finish while the user is unavailable.
Use the measured optimized iteration wall time to precommit a block of
approximately 24 hours, rounded to a checkpoint boundary.  Preserve 8600 as an
internal snapshot for comparison, but continue automatically to the predeclared
24-hour endpoint without looking at development outcomes mid-run.


## Parallel work lane — DeepCrusher DC0 preparation

This lane may advance while LT3 trains because it does not consume the running
trainer or modify its checkpoint.

Completed since the 8200 -> 9105 run started:

- portable OpenPPL expression + ordered WHEN/SET evaluator hardened for the full R8 syntax;
- multiline WHEN normalization and DeepCrusher direct bet-action tokens added;
- all frozen R8 list sections parsed as canonical hand-class sets;
- SPNNIV3 state view exposes the exact 169-class hero hand key;
- OpenHoldem user-variable lifetime corrected to **persist for the current hand** and clear only on hand reset;
- OpenHoldem me_* memory commands implemented with connection-scoped persistence;
- full-source structural compile audit added;
- exact frozen R8 operational source vendored into SpinCore under a SHA256 pin for CI;
- the real R8 source now compiles completely through the portable layer: 1,267 sections / 721 functions / 545 hand-list sections — PASS;
- primitive native-symbol provider and fail-closed coverage audit added: 27 of 269 source-level native identifiers currently implemented, 242 unresolved;
- static DC0 preparation runner added;
- CI corrected so pytest-style DeepCrusher contract tests are actually executed; latest DC0 workflow is PASS.

Evidence for the lifetime semantics comes from the preserved OpenHoldem source,
not inference: CSymbolEngineOpenPPLUserVariables clears its map on hand reset and
leaves it unchanged on heartbeat/new-round/my-turn; CSymbolEngineMemorySymbols
clears its map on connection and not on hand reset.

DC0 remains **BLOCKED**, correctly, on the harder semantic gates: native symbol
provider, exact sizing/action translation, explicit environment profile and
runtime parity fixtures.  No DC1/DC2 score is authorized before those gates pass.


## OpenHoldem deployment lane — PAUSED

Completed before pause:

- observable E2E tracker PASS;
- native C++ tracker PASS;
- native tracker + frozen inference shadow engine PASS;
- Windows x64 shadow user-DLL build PASS;
- Windows mock-host LoadLibrary/ABI gate PASS.

Paused next step:

- inspect the actual OpenHoldem host architecture.

No deployment work is needed now.

## Immediate action

The post-9105 evaluation is complete. A separate research-utilization continuation is now authorized while DC0 engineering proceeds.

The post-9105 development protocol is frozen before seeing development outcomes:
`docs/LT3_POST9105_DEVELOPMENT_BATTERY_PROTOCOL_20260922.md`.

The corrected post-9105 development battery completed successfully on seed 20260922 with 3000 pairwise scenarios, 2000 weak-baseline scenarios and 31 workers. No sealed holdout was touched.

Key development result:
- AveragePolicy 8100 -> 9105 ALL: -0.055 chips/hand, 95% CI [-2.636,+2.526] — INCONCLUSIVE;
- 3H: -2.303, CI [-4.841,+0.235] — INCONCLUSIVE;
- HU: +2.630, CI [-2.151,+7.411] — INCONCLUSIVE;
- current HU ENS8 8100 -> 9105 direct: -0.804, CI [-7.391,+5.784] — INCONCLUSIVE;
- policy drift 8100 -> 9105 is real but moderate, stronger in HU (mean TV 0.0614, p95 0.1456, argmax disagreement 15.68%) than 3H (mean TV 0.0321);
- derived 8600 finalization PASS, source unchanged, zero new training roots.

Interpretation: additional 8100 -> 9105 training changed behavior but did not demonstrate a statistically resolved strength gain over 8100. The battery alone did not justify a claim that more roots improve strength. However, keeping the otherwise-idle Ryzen training while DC0 is engineered is now treated as a separate **research-utilization lane**, not as a conclusion that 9105 was insufficient.

The frozen utilization continuation from finalized 9105 -> 10105 is now **PASS**. It completed exactly 1000 additional iterations / 600,000 roots in 22.564 h, source 9105 remained unchanged, raw milestone 9600 was preserved, postvalidation passed, and no sealed holdout was touched. Final checkpoint SHA256: `f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0`; matched HU ENS8 sidecar SHA256: `8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d`. 10105 remains **RESEARCH_ONLY_NOT_PROMOTED**; training completion alone is not a strength result.

In parallel, DC0 now also includes:
- decision-level benchmark traces with hole cards, visible board, pot, to-call, stacks, exact action and sizing;
- a hand-level sanity audit queue for AA preflop folds, >=10bb 72o jams, top-pair folds, trips+ folds, monster folds and deep high-card jams;
- frozen environment profile `GGPoker_NoPT_NoNotes_V1`: GGPoker=true, other networks=false, named chair lookups=-1, log$=true, colour notes=0, PokerTracker unavailable=-1;
- prwin/prtie explicitly excluded from the environment profile because they are substantive equity/card symbols and still require faithful implementation.

First 9105 -> 10105 launch attempt aborted before training because the earliest
checkpoint preflight called `torch.load()` before `PYTHONPATH` exposed the
SpinCore package, producing `ModuleNotFoundError: No module named 'spincore'`.
No training iteration started and the frozen 9105 artifacts were not modified.
The runner now exports the project Python path before any Python preflight.

Canonical local action now: **leave the running 9105 -> 10105 process untouched**. Do not pull/restart or launch another LT3 trainer. Expected final sentinel: `LT3_PARALLEL_9105_10105_TRAINING_PASS`.

Canonical engineering action remains DC0 -> DC1 -> DC2. Training runs concurrently because the external benchmark work is repository-side and does not consume the Ryzen trainer.

DC0 has advanced materially while the utilization run is active:
- pinned OpenPPL library overlay is integrated and CI-covered; 126 direct R8 source symbols formerly classified as native are actually standard OpenPPL library sections;
- OpenHoldem-compatible card/hand provider is integrated, including rank bits, hand/board expressions, pokerval, pcbits/npcbits, suit/straight/rank-count symbols and exact suit enrichment from the solver deal snapshot;
- direct source-level unresolved dependencies are down to 20 after primitive + environment + library + card providers;
- a true transitive closure audit now follows both R8 and OpenPPL-library dependencies: 336 native leaves, 201 resolved and 135 syntactically unresolved before pruning Hold'em-dead Omaha branches and freezing table/game constants;
- DC0 workflow is PASS after the OpenHoldem straight-metric parity correction.

DC0 closure milestone reached after that tranche:
- transcript-derived OpenHoldem history/raiser/caller symbols are wired into the offline oracle;
- exact current-board `nhandshi/nhandslo/nhandsti` enumeration is ported from OpenHoldem;
- R8 dynamic `vs$multiplex$f$backup_opp_allin_range$prwin/prtie` uses a canonical suit-collapsed 169-class exact preflop equity fixture;
- the transitive Hold'em native closure is now **0 unresolved leaves** (290 resolved, 47 statically Hold'em-dead Omaha/extra-card leaves);
- the dedicated **SpinCore DeepCrusher DC0** workflow is PASS at commit `460eeed27b1651266df7eb145e95f5909d4fc194`.

Remaining DC0 work is no longer symbol closure. The next gates are exact action-origin/sizing translation, full oracle callback integration, and real OpenHoldem parity fixtures across streets before DC1.


## ENS8 parallel matrix incident

The first parallel-fit matrix (V1) was terminated after preflight.  Source
inspection found that V1 serialized the complete 2M-sample HU Advantage
reservoir and then deserialized that Python object graph independently in each
fit subprocess.  That design is not acceptable for the Ryzen/WSL memory
envelope and is superseded regardless of the exact OS termination reason.

V2 replaces per-worker reservoir copies with one compact mmap mirror:

- observations, legal masks, targets and weights are stored once;
- workers open the mirror read-only;
- each member reproduces the historical Python-random sample-index stream;
- the canonical sequential path remains the exact parity reference;
- full Python reservoirs are released before worker pools are created;
- worker RSS is recorded;
- steady-state fit speedup excludes one-time pool/mirror initialization but both
  one-time costs are reported separately.

Do not rerun V1.  Run only the V2 matrix.


## ENS8 parallel matrix V2 — PASS

Measured on the Ryzen against the canonical 8-thread sequential ENS8 fit:

- sequential HU ENS8 fit wall: 114.824 s;
- 2x8: 83.143 s, 1.381x, exact state/loss parity;
- 4x8: 73.027 s, 1.572x, exact state/loss parity — **selected**;
- 8x8: 3777.466 s, exact but severe oversubscription collapse;
- lower-thread layouts (4x4, 8x4, 8x2) were faster but failed exact
  state/loss parity and are rejected.

Important: 1.572x is the measured steady-state speedup of the **HU ENS8 fitting
phase**, not yet the whole training iteration.  A short full-iteration
integration benchmark is required before calculating the ~24-hour block target.

Next gate: integrate persistent mmap + 4x8 fitting into the LT3 continuation
path and measure end-to-end iteration wall time without generating a long run.


## 8200 end-to-end integration gate — PASS

The fit-only matrix is no longer sufficient to authorize the ~24-hour run.
The selected 4x8 fitter is now wired into a disposable end-to-end gate from the
preserved 8200 checkpoint.

The gate runs in isolated processes:

- one sequential control iteration: 8201;
- three parallel 4x8 disposable iterations: 8201..8203;
- exact semantic fingerprint comparison after the shared 8201 iteration;
- authoritative reservoir-write indices/sample digests;
- model and optimizer states;
- RNG states and counters;
- sampler state;
- HU ensemble member states;
- whole-iteration wall time.

The source 8200 checkpoint+sidecar remain read-only.  Only after this gate
passes will the ~24-hour target be frozen.


## 21-hour continuation contract — FROZEN

The 8200 end-to-end gate passed with exact semantic parity.

Measured on the same disposable iteration 8201:

- sequential whole iteration: 123.932 s;
- parallel 4x8 whole iteration: 80.992 s;
- whole-iteration speedup: 1.5301658x;
- parallel median over 8201..8203: 81.108 s;
- historical checkpoint cost: 101.099 s every 50 iterations;
- planning time including checkpoint amortization: 83.130 s/iteration.

The next long block is frozen before training starts:

- source: durable matched checkpoint+sidecar @8200;
- target: 9105;
- additional iterations: 905;
- new roots: 543,000;
- projected total wall: 20.998 h;
- checkpoint every 50;
- raw @8600 checkpoint+sidecar preserved automatically;
- HU fit: exact-parity 4x8;
- no LT2 final-holdout access;
- no LT3 sealed-holdout access.

Do not change target based on intermediate results. The earlier 9250/24.35 h plan is superseded.


## After 9105 PASS

1. Preserve the finalized 9105 checkpoint+ENS8 sidecar and hashes — **PASS**.
2. Keep the automatically preserved 8600 milestone raw and immutable — **PASS**.
3. Freeze post-9105 development protocol and tooling before evaluation — **PASS / READY**.
4. Run the development battery; its runner finalizes AveragePolicy only on a derived 8600 copy and compares 8100 / 8600 / 9105 without sealed-holdout access — **NEXT**.
5. Finish the DeepCrusher DC0 faithful-oracle gate against frozen R8 v22 if it is still incomplete.
6. Run DC1 mechanical paired smoke, then DC2 qualification.
7. Decide whether more roots are justified only from those results.

There is currently no defensible iteration-number forecast for when SpinCore
will beat DeepCrusher.  Earlier training evidence did not establish monotonic
strength growth with iteration count.


## DeepCrusher executable-oracle milestone — 2026-09-23

The DC0 offline oracle is now mechanically executable end-to-end on the real
SpinCore solver. Dedicated CI at commit
`ffb329dd6863be9197ec0d47efdc457fdbc5eebd` reached:

- `DEEPC_RUSHER_ORACLE_RUNTIME_SMOKE_PASS`;
- 22 balanced games / 201 total decisions;
- 77 DeepCrusher decisions across both THREE_HANDED and TRUE_HEADS_UP;
- all four streets exercised (31 preflop, 16 flop, 15 turn, 15 river);
- all exact action families observed in the smoke: FOLD, CHECK, CALL, BET_TO,
  RAISE_TO and ALL_IN;
- every DeepCrusher decision carried raw OpenPPL provenance;
- every terminal row remained zero-sum.

Runtime smoke work also closed several semantics that static closure alone could
not expose: `currentbet_bigblindchair`, OpenHoldem's zero-at-end function
terminal, verbose betround constants, nested action-returning sizing helpers and
Hold'em's absent third/fourth technical hole-card slots (`$$pr2/$$ps2/$$pr3/$$ps3 = -1`).

This is a **mechanical runtime PASS, not yet the canonical DC0 parity PASS**.
Real OpenHoldem parity fixtures remain required for action/sizing equality before
DC1 results may support any strength claim. The paired DC1 development runner is
already implemented and deliberately labels its output DEVELOPMENT_ONLY while
that gate is pending.


## 10105 handoff — 2026-09-23

The Ryzen utilization block has ended cleanly and the machine is free for external-strength work.

Before the first DC1 run, the benchmark SpinCore side is now being corrected to use the actual hybrid research behavior: **THREE_HANDED finalized AveragePolicy + TRUE_HEADS_UP matched current ENS8 sidecar**. A compact generic hybrid inference exporter was added so multiprocess DC1 workers do not fan out the ~2.8 GB training checkpoint.

DeepCrusher action-history fidelity was also tightened: the oracle now preserves whether an executed aggression came from OpenHoldem's minimum Raise button (`didrais/prevaction=2`) or from f$betsize / technical pot-size action (`didbetsize/prevaction=3`). This provenance is carried per DeepCrusher seat across the synthetic hand and fails closed on transcript mismatch.

Next local gate after CI is green: export the compact 10105 hybrid bundle, then run a small **DC1 DEVELOPMENT_ONLY** mechanical smoke with decision traces. Do not interpret that smoke as a canonical strength result until real OpenHoldem parity fixtures pass.


### Ready local gate

`tools/run_deepcrusher_dc1_10105_smoke.sh` is the guarded first local DC1 handoff. It verifies the exact 10105 checkpoint + ENS8 hashes, rebuilds the exact-action solver, exports the compact hybrid bundle, runs 200 balanced DEVELOPMENT_ONLY scenarios with 8 workers, audits SpinCore decision traces for obvious strategic red flags, and packages report + sanity audit + traces + terminal log into one ZIP. This is intentionally a short mechanical/diagnostic smoke before the 1k–5k DC1 scale-up.

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

## 3H Advantage budget stability 100 -> 200 -> 400 — 2026-09-25

The matched eight-replica diagnostic was completed on the frozen 10105
THREE_HANDED Advantage reservoir.  Each replica used one fixed initialization
and one fixed minibatch stream, with cumulative snapshots at 100, 200 and 400
optimizer steps.

Global independent-fit stability improves only modestly:
- pairwise member TV mean: **0.57691 -> 0.55927 -> 0.53238**;
- pairwise argmax disagreement: **64.94% -> 61.72% -> 58.88%**;
- mean max-argmax vote share: **52.40% -> 55.76% -> 58.42%**;
- unanimous argmax rate: **4.24% -> 4.24% -> 5.46%**.

Thus 400 steps are somewhat more stable than fresh100, but the mature 3H
Advantage fit is still highly seed/minibatch-sensitive.  The 100->400 relative
reduction is only ~7.7% in pairwise TV and ~9.3% in argmax disagreement.

The suspicious high-card aggression does not disappear with more fitting.
Across the same 23 flagged 3H high-card jams:
- mean ALL_IN probability of the **member-policy mixture** rises
  **39.54% -> 48.39% -> 51.02%**;
- raw-Advantage ensemble ALL_IN rises
  **52.07% -> 79.78% -> 75.28%**;
- >=5/8 member argmax majority for ALL_IN rises 8 -> 10 -> 11 of 23;
- mean number of members with positive raw ALL_IN advantage remains high:
  5.57 -> 5.48 -> 6.04 of 8.

For the 17 no-immediate-draw flags:
- member-policy-mixture ALL_IN: **40.81% -> 49.83% -> 50.84%**;
- raw-ensemble ALL_IN: **56.55% -> 78.08% -> 74.30%**;
- >=5/8 ALL_IN argmax majority: 7 -> 8 -> 8 of 17.

Therefore:
- fresh100 is indeed noisy/under-resolved, because more steps modestly improve
  independent-fit agreement;
- but **fit budget alone is not a repair for the weird high-card aggression**:
  the shared signal becomes at least as aggressive, not less, by 400 steps;
- 400 is also not demonstrably converged, because independent fits still
  disagree strongly.

The trips result remains distinct.  Raw-ensemble Fold stays 0% at all three
budgets.  The member-policy-mixture Fold moves 16.59% -> 27.38% -> 4.71%, with
no majority Fold argmax at any budget.  The exact benchmark Fold therefore
remains an AveragePolicy/local-generalization issue rather than a stable
current-Advantage recommendation.

### Next discriminating gate

Before spending more compute on 800/1600-step fits or blaming representation,
measure whether the existing 100/200/400 snapshots are still improving the
**actual Advantage regression objective on completely unseen reservoir items**.

A clean holdout audit has been added:
- `tools/audit_3h_advantage_budget_clean_holdout_10105.py`;
- `tools/run_3h_advantage_clean_holdout_10105.sh`.

It reuses the already-created 100/200/400 probe artifact.  Using each member's
saved minibatch seed, it reconstructs the exact Python-random sample-index stream
through 400 steps, marks the union of every reservoir item touched by any of the
eight replicas, and draws a fixed 50k holdout only from indices untouched by
**all** replicas.  It then reports the exact weighted legal-action MSE used by
Advantage training for each member and for the raw-output ensemble.

Decision rule:
- materially falling clean-holdout MSE through 400 => larger fit budgets still
  improve unseen target regression; then test 800/1600 before changing
  architecture;
- flat/worsening clean-holdout MSE => additional optimizer steps are not the
  main bottleneck; move to target/local-neighborhood/representation diagnosis.

No training or checkpoint mutation occurs in this gate.

## 3H Advantage clean-holdout result — 2026-09-26

A fixed 50k holdout was drawn only from 3H Advantage-reservoir indices never
sampled by any of the eight diagnostic replicas through 400 steps.  The union
of all training indices touched through 400 covered 1,611,491 / 2,000,000
retained items (80.57%), leaving 388,509 completely untouched candidates.

Clean-holdout regression improves monotonically with budget, but only modestly.

Mean per-member weighted legal-action MSE:
- 100 steps: **0.0320905**;
- 200 steps: **0.0319440**;
- 400 steps: **0.0316713**.

The 100 -> 400 improvement is ~**1.31%**.  Six of eight matched replicas improve
their own holdout MSE from 100 to 400; two worsen.  Between-member MSE dispersion
falls slightly at 200 then rises at 400
(0.0002369 -> 0.0002120 -> 0.0004351).

Raw-output ensemble weighted MSE also improves monotonically:
- 100: **0.0314557**;
- 200: **0.0313765**;
- 400: **0.0311292**,
an improvement of ~**1.04%** from 100 to 400.

Interpretation:
- fresh100 is not at the regression optimum of the mature frozen 3H reservoir;
- larger budgets still extract some generalizable signal on genuinely unseen
  stored Advantage targets;
- the gain through 400 is small, while policy-level independent-fit
  disagreement remains very large and suspicious high-card aggression becomes
  at least as strong;
- therefore 400 is neither clearly converged nor evidence that simply
  increasing optimizer steps will repair strategy behavior.

The previous common-untouched-holdout construction cannot be extended cleanly
to 1600 steps because the union of eight historical-style minibatch streams
would cover almost the entire 2M reservoir.  The next gate therefore uses a
**controlled fixed split**: remove one common 50k validation set before fitting
any model, then train all eight replicas only on the remaining 1.95M items.

Added:
- `tools/build_3h_advantage_controlled_split_probe_10105.py`;
- `tools/evaluate_3h_advantage_controlled_budget_curve_10105.py`;
- `tools/run_3h_advantage_controlled_budget_curve_10105.sh`.

The controlled curve snapshots each matched replica at
**100 / 200 / 400 / 800 / 1600** steps, evaluates the same untouched 50k
validation set at every budget, and then evaluates all budgets on the unchanged
200-scenario DC1 trajectory.  It reports:
- validation weighted Advantage MSE;
- member pairwise policy TV / argmax disagreement;
- raw-ensemble vs member-policy-mixture divergence;
- high-card jams (all and no-immediate-draw);
- Q8/884 trips Fold.

A runtime preflight is built into the first replica: after 400 fit steps it
projects the full 8x1600 fit wall.  If projected fit time exceeds 55 minutes,
the run aborts before becoming a long compute block so the fitter can be
parallelized/optimized first.

Decision rule:
- validation MSE continues materially down at 800/1600 and policy stability also
  improves => fit budget remains a meaningful bottleneck;
- validation MSE improves but policy instability / weird actions remain =>
  regression fit alone is not the strategic repair; inspect target geometry and
  regret-matching sensitivity;
- validation MSE plateaus/worsens => stop increasing fit budget and move to
  representation/target/coverage diagnosis.

## 3H controlled Advantage budget curve 100 -> 1600 — 2026-09-26

A fixed 50k validation split was removed before any diagnostic fitting. Eight
matched fresh THREE_HANDED Advantage replicas were trained only on the remaining
1.95M retained reservoir items and snapshotted at 100/200/400/800/1600 steps.
The same frozen DC1 200-scenario trajectory was then evaluated at every budget.

### Validation regression

Mean per-member weighted holdout MSE decreases monotonically:
- 100: **0.0317269**;
- 200: **0.0315792**;
- 400: **0.0313879**;
- 800: **0.0311076**;
- 1600: **0.0308359**.

100 -> 1600 improves the held-out regression objective by ~**2.81%**.
Raw-output ensemble weighted MSE also improves monotonically:
**0.0312551 -> 0.0306193** (~**2.03%**).

Thus fresh100 is definitively under-resolved on the mature 3H reservoir and
larger fit budgets continue to extract generalizable target signal through 1600.

### Policy stability

Independent-fit policy stability improves substantially by 1600:
- pairwise member TV mean: **0.59018 -> 0.40470** (~31.4% relative reduction);
- pairwise argmax disagreement: **64.35% -> 43.33%** (~32.7% reduction);
- mean max-argmax vote share: **52.80% -> 70.45%**;
- unanimous argmax rate: **5.78% -> 18.88%**;
- raw-ensemble vs member-policy-mixture TV mean:
  **0.37531 -> 0.19986**.

The curve is therefore not plateaued at 400. Fit budget is a real 3H stability
bottleneck.

### Weird-action surface

Greater target fit does **not** remove the high-card aggression.

For all 23 flagged 3H high-card jams:
- member-policy-mixture ALL_IN: **35.14% @100 -> 50.63% @1600**;
- >=5/8 member ALL_IN-argmax majority: **6 -> 12 of 23**;
- raw ensemble ALL_IN: **55.52% -> 57.12%**.

For the 17 flags with no immediate straight/flush draw:
- member-policy-mixture ALL_IN: **33.43% -> 54.37%**;
- >=5/8 member ALL_IN-argmax majority: **4 -> 10 of 17**;
- raw ensemble ALL_IN: **53.97% -> 58.78%**.

This is the key result: as the eight independent learners fit the frozen
Advantage target problem better and become more stable, the suspicious
high-card ALL_IN signal becomes **more**, not less, shared across members.
Therefore underfitting/noisy fresh100 explains much of the policy instability
but does not explain away the broad high-card aggression.

The Q8/884 trips Fold remains separate:
- raw-ensemble Fold = 0 at 100, 200, 800 and 1600;
- member-policy mixture Fold = 0 at 200 and 1600;
- no budget has majority-member Fold argmax.
The benchmark Fold remains an AveragePolicy/local-generalization tail.

### Gate decision

Do not extend immediately to 3200/6400 merely because holdout MSE still trends
down. The primary unresolved strategic question is now upstream of neural fit:
**do the stored 3H Advantage targets themselves favor these high-card jams in
local neighborhoods, or is the 1600 learner extrapolating from sparse/noisy
targets?**

Added:
- `tools/audit_3h_high_card_advantage_targets_10105.py`;
- `tools/run_3h_high_card_advantage_targets_10105.sh`.

The audit replays the same fixed DC1 trajectory to recover all 23 3H high-card
jam states and focuses on the 17 with no immediate straight/flush draw. It scans
the complete frozen 2M 3H Advantage reservoir once and reports progressively
narrower target neighborhoods:
A) all deep no-draw high-card states with ALL_IN legal;
B) same street/facing class;
C) same dealer-relative position, statuses and exact universal legal mask;
D) similar effective stack/pot/call/current-bet geometry;
E) same last two frozen V1 history tokens;
F) same hole-card ranks ignoring suits.

For every subset it reports stored ALL_IN target mean, iteration-weighted mean,
positive fraction, argmax fraction, sole-positive-action fraction and
<=8100 / 8101-9105 / 9106-10105 cohorts.

Positive ALL_IN Advantage target mass in recent tight neighborhoods would show
that the aggressive signal is already in the training targets. Sparse or
contradictory tight neighborhoods would instead implicate generalization and/or
external-sampling variance. The audit does not by itself declare any jam
strategically correct.

## High-card Advantage-target audit V1 correction — 2026-09-26

The first high-card target audit completed mechanically but its narrow subsets
C-F are **invalid** because of a diagnostic representation bug:
- each retained `ActionAdvantageSample.legal` is the 10-slot 0/1 legal mask;
- each replay target stored `legal` as the list of legal slot indices;
- V1 subset C compared the mask tuple directly to the slot-index tuple.

That comparison can never match ordinary states.  Accordingly all 17 no-draw
targets reported C=0 and therefore D/E/F=0.  Those zeros are artificial and
must not be interpreted as evidence of zero local reservoir coverage.

V1 subsets A and B do not use that comparison and remain valid.

Valid V1 broad A result:
- 133,373 retained 3H postflop high-card/no-immediate-draw samples with ALL_IN
  legal under the V1 candidate filter;
- ALL_IN target mean **-0.0128170**;
- iteration-weighted mean **-0.0127007**;
- ALL_IN target positive in **19.736%**;
- ALL_IN target argmax in **12.356%**;
- ALL_IN was the sole positive legal target in **4.979%**.
The recent 9106-10105 cohort is similar: mean **-0.0126578**, positive
**19.896%**.  Broadly, therefore, the stored targets do not support a generic
"jam high card" rule.

Valid V1 subset B also shows heterogeneous street/facing regimes:
- flop facing action: mean ALL_IN target ~**-0.00198**, positive ~19.90%;
- flop checked-to: mean ~**+0.00122**, positive ~29.16%;
- river facing action: mean ~**-0.04470**, positive ~15.58%;
- river checked-to: mean ~**-0.03508**, positive ~16.10%.
These are still broad buckets and cannot decide any flagged jam.

A second diagnostic inconsistency was also exposed.  The original
`decision_sanity.effective_stack_bb` derives its heuristic from every non-DEAD
lineup seat because `DecisionTrace` does not carry folded/all-in status.
For four of the 17 no-draw flagged states, actor-relative SPNNIV1 statuses imply
a smaller stack against the opponent still contesting the pot (<10bb) even
though the original sanity heuristic classified the decision as >=10bb.  This
does not change the played action; it means the "deep" flag is a review heuristic
rather than an authoritative effective-stack calculation.

V2 correction (commit `38707d3b9a7f58d3f40c35ff636846bdb094a049`):
- convert each reservoir legal mask to its legal-slot index tuple before subset
  C comparison;
- record both the original sanity effective-stack provenance and a
  contesting-opponent effective stack derived from actor-relative folded status;
- remove the inconsistent secondary >=10bb cutoff from the reservoir base
  filter so the fixed flagged states are not silently reclassified;
- use contesting-opponent effective stack only for local geometry D;
- schema becomes `SPINCORE_3H_HIGH_CARD_ADVANTAGE_TARGET_AUDIT_V2`.

The V2 rerun is required before any C-F/local-target conclusion or strategy
change.

## Corrected V2 high-card Advantage-target audit — 2026-09-26

The V2 rerun fixes the V1 legal-mask/slot comparison error and removes the
inconsistent secondary >=10bb reservoir cutoff.  All 17 no-immediate-draw
flagged states now have nonzero C-E local coverage.

### Broad target surface

Across 267,649 retained 3H postflop HIGH_CARD / no-immediate-draw samples with
ALL_IN legal:
- ALL_IN target mean: **-0.0088908**;
- iteration-weighted mean: **-0.0088748**;
- ALL_IN target positive fraction: **15.409%**;
- ALL_IN target argmax fraction: **10.147%**;
- ALL_IN sole-positive-legal fraction: **4.840%**.

The recent 9106-10105 cohort is nearly unchanged:
- mean **-0.0090655**;
- positive fraction **15.536%**;
- 27,028 retained samples.

Thus the stored target population does **not** contain a generic high-card jam
rule.

### Local neighborhoods around the 17 DC1 no-draw jams

At subset E (same street/facing class, dealer-relative structure, exact legal
slot set, near stack/pot/price geometry, and same last two V1 history tokens):
- coverage range: **23 .. 1,048** retained samples;
- median coverage: **342**;
- unweighted ALL_IN target mean is negative in **13/17** states;
- iteration-weighted ALL_IN target mean is negative in **12/17**;
- the recent 9106-10105 cohort mean is negative in **14/17**.

Nine E neighborhoods have both >=100 total retained samples and >=30 recent
samples.  Of those nine, **8/9** have a negative recent ALL_IN target mean; the
only clearly positive recent neighborhood is scenario 140.

Therefore the broad 1600-step learner's high-card aggression cannot be
explained simply by saying that the local stored targets generally teach
ALL_IN.  For most flagged states, reasonably local target aggregates point the
other way.

However subset F (E + exact two hole-card ranks ignoring suits) is very sparse:
- median retained count: **3**;
- range: **0 .. 22**;
- 4/17 states have zero F samples;
- 12/17 have zero recent F samples;
- maximum recent F count is only 4.

That prevents treating the E mean as the exact conditional target for the
specific hand.  The evidence instead points to a coverage/generalization
problem boundary: public/geometry neighborhoods are often well populated, while
specific hole-rank conditioning is extremely sparse.

Representative contrasts:
- scenario 30 (83o checked to on K-7-4): E has 1,048 samples and recent mean
  ~-0.00791; exact-hole-rank F has 22 samples with mean ~-0.03664;
- scenario 140 (A6o facing action): E is genuinely positive, including recent
  mean ~+0.02139 over 61 samples, but exact-hole-rank F has only 6 samples and a
  negative mean ~-0.01206;
- several river/facing-action neighborhoods remain strongly negative even
  before exact-hole filtering.

### Interpretation

This result rejects two overly simple explanations:
1. **"fresh100 noise alone"** — already rejected by the controlled 1600 fit;
2. **"the CFR target reservoir broadly teaches these jams"** — V2 shows that
   most local E target means are negative, including most well-covered recent
   neighborhoods.

What remains unresolved is the exact model-target gap at the flagged states.
Because F coverage is sparse, the network must generalize across hole-card
combinations.  Separately, lean regret matching can convert a small positive raw
ALL_IN prediction into a very large action probability when competing outputs
are nonpositive.

Next discriminating gate:
- reuse the already-built controlled 1600-step eight-model probe;
- replay the exact same 17 no-draw flagged states;
- record the exact raw ALL_IN Advantage from the 1600 raw ensemble, the
  member-policy mixture and the post-regret-matching raw-ensemble policy;
- join each exact prediction to its V2 E/F target-neighborhood statistics;
- count model-positive / local-weighted-target-negative sign mismatches;
- measure whether >=50% ALL_IN policies are mostly produced by small positive
  raw margins against locally negative targets.

Added:
- `tools/audit_3h_high_card_model_target_gap_10105.py`;
- `tools/run_3h_high_card_model_target_gap_10105.sh`.

This gate performs no training and generates no CFR roots.  It is specifically
intended to distinguish:
- model/generalization sign error;
- nonlinear regret-matching amplification near zero;
- or genuinely positive hand-specific target evidence where enough F coverage
  exists.

No strategy patch, dead-zone, representation migration, or additional long
training is authorized before this result.

## 3H exact high-card model-target gap @1600 — 2026-09-26

The exact 17 no-immediate-draw DC1 high-card jam states were joined to the
corrected V2 local Advantage-target neighborhoods and evaluated with the
controlled-split eight-member 1600-step 3H Advantage probe.

### Aggregate result

Local target surface (subset E):
- E coverage: **23 .. 1,048** retained samples; median **342**;
- unweighted ALL_IN target mean negative in **13/17** states;
- iteration-weighted ALL_IN target mean negative in **12/17**;
- recent 9106-10105 mean negative in **14/17**;
- among the 9 states with >=30 recent E samples, **8/9** have a negative recent
  ALL_IN target mean.

Exact 1600-step model:
- raw-ensemble ALL_IN Advantage is positive in **14/17** states;
- negative in only **3/17**;
- mean post-regret-matching raw-ensemble ALL_IN probability is **58.78%**;
- mean member-policy-mixture ALL_IN is **54.37%**.

Model/local-target sign mismatch:
- **9/17** exact states have positive raw model ALL_IN Advantage while the
  iteration-weighted E target mean is negative;
- **7/17** simultaneously have negative weighted E targets and >=50%
  raw-ensemble ALL_IN policy;
- across these selected flagged states, raw model ALL_IN Advantage is weakly
  *negatively* correlated with E target means (Pearson ~**-0.22**). This
  correlation is descriptive only because the 17 states were selected by the
  weird-action flag.

The positive raw ALL_IN values are numerically small: among the 14 positive
exact states they range roughly **0.00268 .. 0.02053**, median **0.00776**.
Nevertheless lean regret matching can turn them into very large action mass
when competing legal predictions are <=0 or much smaller.

Clear examples:
- scenario 51 river, J8 on Q-4-9-5-7: E weighted target **-0.03042** with
  613 samples and recent mean **-0.03292** over 66 samples; raw ensemble predicts
  ALL_IN **+0.00565** while CHECK_CALL/POT_33 are negative, therefore regret
  matching returns **100% ALL_IN**;
- scenario 125 flop, 96 on 3-K-Q checked to: E weighted target **-0.01281**;
  exact raw ALL_IN is only **+0.00334**, the other two legal raw outputs are
  negative, therefore regret matching again returns **100% ALL_IN**;
- scenario 39 flop, AT on 5-8-K facing action: E weighted target **-0.02040**;
  exact raw ALL_IN **+0.02053** vs CHECK_CALL +0.00648, producing **76.0%**
  ALL_IN;
- scenario 107 flop, 73 on 9-2-5 facing action: E weighted target **-0.01273**;
  exact raw ALL_IN **+0.01125** vs CHECK_CALL +0.00350, producing **76.3%**
  ALL_IN.

This establishes that **both** mechanisms are present on the flagged surface:
1. a model/generalization sign mismatch relative to reasonably local stored
   target aggregates in a material subset of states;
2. nonlinear regret-matching amplification of small positive raw outputs.

It still does not prove the exact poker-optimal action is non-jam because subset
E is an approximate neighborhood and same-hole-rank subset F remains sparse.
The next gate must determine whether small positive ALL_IN predictions are
actually calibrated on a genuinely untouched holdout, especially within
postflop high-card/no-draw states.

Next diagnostic:
- reconstruct the same fixed 50k holdout excluded from every controlled-split
  replica;
- evaluate the 1600-step raw ensemble on every held-out sample;
- report ALL_IN-head prediction/target calibration globally and specifically
  for postflop high-card/no-immediate-draw states;
- use fixed raw-prediction bins around the exact anomaly range
  (0, .0025, .005, .01, .02);
- report weighted target mean, target-positive rate, residual bias and MSE in
  each bin and by street/facing class;
- attach each of the 17 flagged states to its corresponding holdout calibration
  bin.

Decision rule:
- if positive bins around +0.003..+0.020 have positive held-out conditional
  target means, the regression head is broadly calibrated and the remaining
  problem is state/hand-specific representation + local coverage plus
  regret-matching sensitivity;
- if those bins have zero/negative held-out conditional target means, the
  ALL_IN head itself is sign-miscalibrated and model/loss calibration becomes
  the primary repair target;
- in either case, no production dead-zone or strategy patch is authorized by
  the diagnostic alone.

Added:
- `tools/audit_3h_allin_holdout_calibration_10105.py`;
- `tools/run_3h_allin_holdout_calibration_10105.sh`.

The calibration reuses the existing 1600-step probe and therefore performs no
new fitting.

## 3H ALL_IN untouched-holdout calibration — 2026-09-26

The controlled-split 1600-step raw-Advantage ensemble was evaluated on the
original untouched 50k validation split.

### Global ALL_IN head

Across 38,246 holdout samples with ALL_IN legal, the raw ALL_IN head is broadly
calibrated in aggregate:
- weighted prediction mean: **+0.006295**;
- weighted target mean: **+0.005406**;
- weighted residual bias: **+0.000889**.

The positive raw-prediction bins are also mostly directionally correct globally:
- +0.0025..0.005 -> weighted target **+0.00517**;
- +0.005..0.010 -> **+0.00736**;
- +0.010..0.020 -> **+0.01147**;
- >=0.020 -> **+0.03421**.
Only the tiny +0..0.0025 bin has a negative weighted target mean
(**-0.00189**).

Therefore a global positive-Advantage dead-zone or blanket regret-matching
threshold would suppress many genuinely positive held-out ALL_IN signals.

### Postflop HIGH_CARD / no-immediate-draw subgroup

The exact same model is strongly upward-biased inside the diagnostic hand class:
- 6,639 untouched holdout samples;
- weighted prediction mean: **+0.006184**;
- weighted target mean: **-0.007817**;
- weighted residual bias: **+0.014001**.

Every positive prediction bin below +0.020 has a **negative** weighted target:
- +0..0.0025: target **-0.01558**, bias +0.01696;
- +0.0025..0.005: **-0.00560**, bias +0.00946;
- +0.005..0.010: **-0.00703**, bias +0.01455;
- +0.010..0.020: **-0.00342**, bias +0.01596.
The >=+0.020 bin turns positive (+0.00909) but contains only 14 samples.

Consequently **16/17** exact flagged DC1 states fall into a high-card holdout
prediction bin whose weighted target mean is negative.  Conditioning further by
street/facing class still leaves **10/17** in negative bins.

The street split is heterogeneous:
- river checked-to and river facing-action are strongly negative across the
  small-positive ranges;
- flop facing-action is near zero at +0.005..0.010 and positive at
  +0.010..0.020;
- flop checked-to has some positive small-prediction ranges but is not
  monotonic;
- turn buckets are mixed.

### Gate decision

This result materially narrows the repair target.

It rejects a **global regret-matching dead-zone** as the primary repair because
the ALL_IN head is broadly calibrated outside the problematic subgroup.
Instead, the failure is **state/hand-conditional**: the V1 learner systematically
overpredicts ALL_IN Advantage for postflop high-card/no-immediate-draw states.

This is consistent with a representation/generalization limitation.  SPNNIV1
carries exact card ids but no explicit made-hand, draw or board semantic fields.
The historical SPNNIV2 lane contains explicit made category, pair relation,
overcards, flush/straight-draw and board-texture semantics, but the old full-V2
paired ablation selected C0/V1 because all V2 candidates were globally worse
on the then-frozen corpus.  Therefore reopening full V2 directly is not yet
authorized.

### Next gate — lightweight semantic attribution

Added:
- `tools/audit_3h_semantic_sidecar_attribution_10105.py`;
- `tools/run_3h_semantic_sidecar_attribution_10105.sh`.

The gate performs **no CFR training and no base-network fitting**.  It reuses the
existing 1600-step raw ensemble, draws a deterministic 250k calibration subset
from the existing 1.95M training side, and fits three tiny weighted-ridge
diagnostic calibrators:

A. RAW_AFFINE — raw ALL_IN prediction only;
B. CONTEXT — raw + street/facing/pot geometry;
C. SEMANTIC_SIDECAR — CONTEXT plus semantics derived only from the existing V1
card tokens (made category, pair relation, overcards, draws, board texture and
high-card/no-draw identity).

All three are evaluated on the untouched 50k holdout.

Precommitted attribution PASS requires:
- >=50% reduction in absolute high-card/no-draw residual bias;
- no >2% degradation in global ALL_IN MSE;
- no degradation in high-card/no-draw MSE.

PASS would justify a proper matched V1+semantic shadow-network experiment.
FAIL would move the diagnosis away from simple semantic conditioning and toward
richer history/sequence representation or target/objective structure.

The ridge outputs are diagnostic only and are not deployable poker policy.

## Semantic sidecar attribution V1 correction — 2026-09-26

The first semantic-sidecar run reported a formal PASS, but the high-card/no-draw
population did **not** match the immediately preceding untouched-holdout
calibration population:
- calibration gate: **6,639** high-card/no-draw ALL_IN-legal holdout samples;
- semantic sidecar V1: **4,802** samples.

The discrepancy was traced to river handling.  The prior calibration correctly
treated river HIGH_CARD as "no immediate draw" after excluding already-made
straight/flush/pair categories, because no future community card remains.
Semantic-sidecar V1 instead also excluded river states with four cards to a
flush or four ranks in a straight window.  That silently removed 1,837 states,
including part of the river regime where the strongest positive ALL_IN bias had
been observed.

Therefore:
- V1 global ALL_IN metrics remain mechanically valid for all 38,246
  ALL_IN-legal holdout samples;
- V1 high-card/no-draw attribution metrics and its precommitted PASS are
  **provisional / not decision-valid**, because they were evaluated on the wrong
  subgroup.

The provisional V1 numbers were directionally encouraging:
- global MSE: **0.0115629 -> 0.0113720** with the semantic sidecar (~1.65%
  improvement);
- on the narrower 4,802-sample subgroup, absolute residual bias fell from
  **0.006939 -> 0.002029** (~70.8% reduction) and subgroup MSE also improved
  slightly.

But these numbers cannot authorize a representation experiment until population
parity is restored.

Correction:
- `tools/audit_3h_semantic_sidecar_attribution_10105.py` now emits
  `SPINCORE_3H_SEMANTIC_SIDECAR_ATTRIBUTION_V2`;
- flop/turn retain the one-card draw exclusion;
- river keeps HIGH_CARD states regardless of four-suit / four-rank texture,
  matching `audit_3h_allin_holdout_calibration_10105.py`;
- a hard guard requires exactly **6,639** frozen holdout high-card/no-draw rows,
  so this population drift cannot silently recur.

The same precommitted attribution criteria remain unchanged:
- >=50% reduction in absolute high-card/no-draw residual bias;
- global ALL_IN MSE not worse than BASE_RAW by >2%;
- high-card/no-draw MSE not worse than BASE_RAW.

A V2 rerun is required before moving to any V1+semantic shadow-network
experiment.

## Semantic sidecar attribution V2 PASS — 2026-09-26

The corrected V2 rerun restores exact population parity with the prior untouched
holdout calibration:
- all ALL_IN-legal holdout rows: **38,246**;
- postflop HIGH_CARD / no-immediate-draw rows: **6,639**;
- fixed holdout size: **50,000**.

The precommitted attribution gate passes all three criteria.

Global ALL_IN regression:
- BASE_RAW weighted MSE: **0.01156286**;
- SEMANTIC_SIDECAR weighted MSE: **0.01136353**;
- relative improvement: ~**1.72%**;
- global weighted target mean remains **+0.005406**.

HIGH_CARD / no-immediate-draw subgroup:
- BASE_RAW prediction mean: **+0.006184** vs target **-0.007817**;
- BASE_RAW residual bias: **+0.014001**;
- SEMANTIC_SIDECAR prediction mean: **-0.008073** vs the same target
  **-0.007817**;
- SEMANTIC_SIDECAR residual bias: **-0.000257**;
- absolute subgroup bias reduction: ~**98.17%**;
- subgroup weighted MSE: **0.00867195 -> 0.00834577**
  (~**3.76%** improvement).

Generic context alone does not solve the failure:
- CONTEXT subgroup bias remains **+0.013841**;
- RAW_AFFINE remains **+0.013683**.
The correction appears only when hand/board semantic features are available.

Interpretation:
- the previously measured high-card ALL_IN error is not explained by a global
  ALL_IN calibration defect;
- street/facing/pot context alone is insufficient;
- semantics derivable from information already contained in SPNNIV1 almost
  eliminate the subgroup bias while also improving global MSE;
- this is strong attribution evidence for a **representation/generalization**
  bottleneck in the frozen V1 learner.

This does not authorize deploying the ridge sidecar.  It authorizes the next
matched shadow experiment: keep the entire V1 observation/model path and add
only general poker-semantic features to the neural representation.

Added:
- `tools/run_3h_v1_semantic_shadow_10105.py`;
- `tools/run_3h_v1_semantic_shadow_10105.sh`.

### V1 + semantic shadow contract

The shadow candidate deliberately does **not** receive an explicit
`high_card_no_draw` flag.  It receives general semantics only:
- made-hand category;
- pair relation;
- overcard count;
- pocket-pair / flush-draw / straight-draw / backdoor flags;
- board pairedness, suit density, straight-window occupancy and broadway density.

Those features are derived deterministically from the same card tokens already
present in SPNNIV1.  No new game-state information is introduced.

The candidate is fit against the frozen 10105 3H Advantage reservoir with:
- the exact existing fixed 50k holdout;
- the exact same train split as the controlled V1 1600 probe;
- the same eight init seeds;
- the same eight minibatch RNG streams;
- the same 1600-step budget, batch size and learning rate.

Baseline is the already-built eight-member V1 1600 probe.

Precommitted shadow PASS requires:
1. global legal-action weighted MSE not >2% worse than V1;
2. global ALL_IN weighted MSE not >2% worse;
3. >=50% reduction in absolute high-card/no-draw ALL_IN bias;
4. no worsening of high-card/no-draw ALL_IN MSE;
5. DC1 member-pairwise policy TV not >10% worse.

The fixed DC1 trajectory additionally reports, but does not optimize against:
- all 23 high-card jam flags;
- the 17 no-immediate-draw flags;
- trips-plus-fold;
- raw-ensemble vs member-policy-mixture action mass and majority argmax counts.

PASS means the semantic attribution survives end-to-end neural refitting under a
matched budget.  It still does not promote the candidate to production.

## 3H V1 + general-semantic Advantage shadow PASS — 2026-09-26

The matched eight-member 1600-step neural shadow completed on the exact frozen
10105 THREE_HANDED Advantage reservoir.  The candidate kept the complete V1
observation/model path and added only 30 general poker-semantic features derived
from card information already present in SPNNIV1.  It did **not** receive an
explicit `high_card_no_draw` flag.

The precommitted shadow gate passes every criterion.

Untouched 50k Advantage holdout:
- global legal-action weighted MSE:
  **0.03061927 V1 -> 0.02721929 semantic** (~**11.10%** improvement);
- global ALL_IN weighted MSE:
  **0.01156286 -> 0.01127429** (~**2.50%** improvement);
- HIGH_CARD/no-immediate-draw ALL_IN bias:
  **+0.01400084 -> -0.00167877**, an ~**88.0%** reduction in absolute bias;
- subgroup ALL_IN weighted MSE:
  **0.00867195 -> 0.00825819** (~**4.77%** improvement).

Fixed DC1 development trajectory remains identical:
- 200 scenarios;
- 860 balanced games;
- 3,506 decision traces;
- 1,557 SpinCore 3H decisions.

Independent-member stability improves rather than regresses:
- pairwise policy TV:
  **0.40470 -> 0.39778** (~1.7% improvement);
- pairwise argmax disagreement:
  **43.33% -> 40.75%** (~5.95% relative improvement).

The suspicious high-card surface changes materially:
- all 23 high-card jam flags, member-policy ALL_IN mean:
  **50.63% -> 27.84%**;
- all 23 raw-ensemble ALL_IN:
  **57.12% -> 33.28%**;
- majority-member ALL_IN argmax:
  **12/23 -> 6/23**;
- 17 no-immediate-draw flags, member-policy ALL_IN mean:
  **54.37% -> 22.64%**;
- 17 no-draw raw-ensemble ALL_IN:
  **58.78% -> 24.05%**;
- majority-member ALL_IN argmax:
  **10/17 -> 3/17**.

The Q8/884 trips Fold remains 0% in both matched Advantage ensembles, so the
semantic candidate does not reintroduce that failure.

This is stronger than the linear sidecar attribution:
- semantics improve the actual neural Advantage fit, not only posthoc
  calibration;
- the gain is global, not purchased by degrading the overall heldout objective;
- independent-fit stability improves;
- the previously suspicious DC1 high-card action mass drops sharply.

The result is still **development evidence only**.  The frozen 10105 reservoir
was generated historically under the V1 learner and the actual 3H deployment
continues to use the finalized AveragePolicy.  Therefore this PASS does not
authorize replacing the production representation or resuming DC2.

### Next gate — AveragePolicy semantic bridge

Before any online/CFR semantic continuation, test whether the same general
semantics also improve the second Deep-CFR learner: the THREE_HANDED
AveragePolicy.

Added:
- `tools/audit_3h_average_policy_semantic_continuation_10105.py`;
- `tools/run_3h_average_policy_semantic_continuation_10105.sh`.

The gate creates two paired strategy-only continuation arms from the exact
finalized 10105 AveragePolicy:
1. V1 control;
2. V1 + the same 30 general semantic features.

The semantic model is **functionally identical at step 0**:
- every existing V1 weight is copied;
- added semantic input weights are initialized to zero;
- existing Adam moments are copied exactly;
- new semantic-column moments start at zero.

Both arms then receive the exact same strategy-reservoir train split and exact
same minibatch stream for 4000 extra optimizer steps.  The untouched original
production AveragePolicy is preserved separately as a DC1 reference.

Precommitted gate criteria:
- semantic weighted strategy cross-entropy <= paired V1 control;
- semantic weighted target-policy TV <= paired V1 control;
- >=50% reduction in absolute HIGH_CARD/no-draw ALL_IN target bias on the fixed
  strategy holdout.

DC1 high-card jams and the Q8/884 Fold are descriptive diagnostics only and do
not move the gate.

PASS would show that the representation benefit reaches both Deep-CFR neural
learners and would justify designing one controlled online semantic-CFR
continuation.  FAIL would mean the Advantage improvement alone is insufficient
to bridge the current AveragePolicy deployment path.

## AveragePolicy semantic continuation runtime fix — 2026-09-26

The first run stopped before any optimizer step at the step-0 identity gate:
max absolute logit difference was **1.1920928955078125e-06** against a
hard-coded 1e-6 tolerance.

Inspection showed this is a float32 GEMM-shape effect, not a model-state
mismatch: the semantic model has a wider first linear layer, so BLAS may use a
slightly different accumulation order even though all appended semantic columns
are exactly zero.

The gate was strengthened rather than simply relaxed:
- every pre-existing V1 parameter must match bit-for-bit;
- the copied prefix of the widened first-layer weight must match bit-for-bit;
- every newly-added semantic column must be exactly zero;
- numerical step-0 logits must agree within 2e-6;
- masked action probabilities must agree within 5e-7.

No training occurred in the failed run, so there is no partial state to reuse.
The run should be restarted from the frozen 10105 checkpoint.

## 3H AveragePolicy semantic continuation FAIL — 2026-09-26

The paired 4000-step strategy-only continuation completed on the frozen 10105
THREE_HANDED strategy reservoir.

The gate **FAILS** its precommitted criteria:
- semantic high-card/no-draw ALL_IN absolute-bias reduction >=50%: **FAIL**;
- semantic weighted strategy cross-entropy <= V1 control: **FAIL**;
- semantic weighted target-policy TV <= V1 control: **PASS**.

At 4000 extra steps:
- V1 weighted CE: **1.08594394**;
- semantic weighted CE: **1.08594871** (numerically almost identical, but
  slightly worse);
- V1 weighted TV: **0.42812876**;
- semantic weighted TV: **0.42811827** (slightly better);
- high-card/no-draw target ALL_IN mean: **0.26614217**;
- V1 prediction: **0.26893334**, abs bias **0.00279118**;
- semantic prediction: **0.27013153**, abs bias **0.00398936**.

No milestone satisfies the full gate.  At 1000 steps semantics temporarily
reduce high-card bias versus the matched V1 continuation
(0.002824 vs 0.004865), but CE and TV are both slightly worse and the reduction
does not reach the precommitted 50% rule.  By 4000 the bias ordering reverses.

Both continuation arms also degrade the fixed strategy holdout relative to
step 0:
- step-0 weighted CE ~**1.08157735**;
- 4000-step V1 ~**1.08594394**;
- 4000-step semantic ~**1.08594871**.

This reproduces the earlier conclusion that simply continuing to optimize the
finalized AveragePolicy on the same historical strategy reservoir does not
improve its generalization.  The reservoir contains time-varying historical
strategy targets.

The fixed DC1 weird-action diagnostics do not show a semantic repair in
AveragePolicy:
- 17 no-draw high-card flags: production reference **30.11%** ALL_IN, 4000-step
  V1 control **29.67%**, semantic **31.50%**;
- Q8/884 Fold: production **7.33%**, V1 control **5.70%**, semantic **6.67%**.

### Interpretation boundary

This FAIL does **not** invalidate the semantic Advantage result.  In canonical
Deep-CFR collection, each strategy-memory target is the contemporaneous
behavior policy `sigma` produced by the current Advantage learner.  The frozen
10105 strategy reservoir was generated historically under the V1 behavior
sequence.  Adding semantic inputs to an AveragePolicy while keeping those old
targets fixed cannot retroactively turn them into the corrected semantic
behavior policy.

Therefore the next question is not "can semantics refit the old strategy
reservoir?" — the answer is no.  The next causal bridge must let the already
validated semantic Advantage candidate generate **fresh strategy targets**, and
then test whether those targets can be distilled into an AveragePolicy.

No production promotion, DC2 scale-up, or full online semantic-CFR continuation
is authorized yet.

### Next causal bridge — fresh semantic strategy targets

Because the historical-strategy continuation gate failed while the semantic
Advantage shadow passed strongly, the next test must remove the stale-target
confound before any full online CFR continuation.

Added:
- `tools/audit_3h_fresh_semantic_strategy_distill_10105.py`;
- `tools/run_3h_fresh_semantic_strategy_distill_10105.sh`.

The gate freezes the already-passed eight-member 1600-step semantic Advantage
shadow and generates **fresh sampled 3H trajectories under that policy**.
Every visited state receives the contemporaneous semantic-Advantage
regret-matched `sigma` as its strategy target.  Whole episodes are split
80/20 into train/holdout to avoid same-hand leakage.

Two AveragePolicy arms start from the exact finalized 10105 policy/Adam state:
- V1 control;
- V1 + the same 30 general semantic features, zero-initialized at the added
  input columns.

Both arms receive identical fresh-target minibatches for 4000 steps.

Precommitted PASS:
1. semantic weighted holdout CE < V1;
2. semantic weighted holdout TV < V1;
3. semantic high-card/no-draw ALL_IN absolute bias at least 25% lower than V1;
4. semantic holdout CE improves versus its own step-0 value.

PASS would show that the AveragePolicy can in fact distill the corrected
semantic behavior when the strategy targets are generated by the semantic
Advantage learner.  Only then is one bounded paired online semantic-CFR
continuation justified.

## Fresh semantic strategy bridge runtime fix — 2026-09-26

The first fresh-strategy distillation run stopped during trajectory collection
before any AveragePolicy optimizer step.

Cause:
- the collector wrapped each solver state in `LeanSolverState`;
- however it accidentally queried `state.inner.universal_legal_actions(...)`,
  which uses the generic universal-action legal set;
- the sampled action was then sent to `apply_lean(...)`, whose authoritative
  lean resolver correctly rejected one state-local alias/inactive action.

Fix (commit `103fab5e94735fecba88d880197e6f2e81cb634b`):
- query `state.universal_legal_actions(active)`, i.e. the same lean
  legacy-action contract used by `apply_lean`;
- keep action sampling and application on one consistent legal-action ABI;
- also remove the harmless PyTorch warning by materializing the one-row semantic
  numpy batch before tensor conversion.

The failed run generated no policy-training state and mutated no frozen
checkpoint. Restart the gate from the 10105 source artifacts.

## Fresh semantic strategy bridge holdout sizing fix — 2026-09-26

The corrected lean-action run completed all 4,000 fresh 3H episodes and collected
14,888 strategy decisions, but the whole-episode 80/20 split produced only
3,078 holdout decisions.  The gate had intentionally required at least 5,000
holdout decisions before any AveragePolicy training, so it stopped before the
first optimizer step.

This is a sample-size/preflight issue, not a strategy failure.  The 5,000-row
minimum is preserved.  Rather than weakening the holdout requirement after
seeing the sample count, the collection horizon is increased to **8,000 whole
episodes**, still with the same deterministic 80/20 episode split and the same
frozen semantic-Advantage generator.

The prior 4,000-episode run performed no AveragePolicy fitting and mutated no
source checkpoint.  Restart from the frozen 10105 source artifacts.

## Fresh semantic strategy distillation PASS — 2026-09-26

The fresh-target bridge completed after the holdout-sizing correction:
- **8,000** whole 3H episodes;
- **29,957** fresh strategy decisions generated by the frozen passed semantic
  Advantage ensemble;
- train: **6,400 episodes / 23,862 samples**;
- holdout: **1,600 episodes / 6,095 samples**.

All precommitted criteria pass at the frozen 4000-step reporting point:
- semantic weighted CE < V1 control;
- semantic weighted TV < V1 control;
- semantic high-card/no-draw ALL_IN absolute bias at least 25% lower;
- semantic CE improves from its identical step-0 initialization.

At 4000 steps:
- weighted CE: **0.88639 V1 -> 0.59083 semantic** (~33.3% lower);
- weighted TV: **0.17636 -> 0.11968** (~32.1% lower);
- high-card/no-draw ALL_IN target mean: **0.09182**;
- V1 prediction: **0.24014**, abs bias **0.14832**;
- semantic prediction: **0.10018**, abs bias **0.00837** (~94.4% lower).

The strongest holdout point is earlier, at **500 steps**:
- semantic CE **0.48285** vs V1 **0.62864**;
- semantic TV **0.11566** vs V1 **0.17995**;
- semantic high-card/no-draw ALL_IN bias **0.000335** vs V1 **0.15716**.

After 500 steps the semantic model remains materially better than V1 but begins
to overfit the small fresh-target training set.  Therefore the 4000-step PASS
establishes the causal bridge, while the apparent 500-step optimum must not be
promoted directly because that milestone was selected after inspecting the
first holdout.

DC1 development diagnostics at 4000 also move in the expected direction:
- 17 no-draw high-card states: ALL_IN mean **54.70% V1 -> 35.33% semantic**;
- ALL_IN argmax count **10/17 -> 6/17**;
- Q8/884 Fold **0.982% V1 -> effectively 0% semantic**.

### Interpretation

The stale-target confound is now isolated:
- semantics strongly improve the Advantage learner on the frozen Advantage
  reservoir;
- semantics do not help an AveragePolicy fit the old V1 historical strategy
  reservoir;
- when the passed semantic Advantage ensemble generates fresh contemporaneous
  strategy targets, a semantic AveragePolicy distills them far better than the
  matched V1 control.

This completes the representation-capacity bridge, but the 500-step milestone
was chosen post-hoc.  Before any online semantic-CFR continuation, freeze
**500 steps as a precommitted candidate** and validate it on a completely new
fresh-target episode stream that was not used for milestone selection.

No production promotion or DC2 scale-up is authorized by the first holdout.

### Independent confirmation of the 500-step fresh semantic distill

Because the first fresh-target holdout revealed 500 steps as the best observed
milestone, that exact budget is now frozen **before** a new evaluation stream is
generated.

Added:
- `tools/audit_3h_fresh_semantic_strategy_independent500_10105.py`;
- `tools/run_3h_fresh_semantic_strategy_independent500_10105.sh`.

Contract:
- regenerate the exact original 8,000-episode fresh-target stream;
- use only its original 6,400-episode / 23,862-sample train side;
- ignore the previously inspected 1,600-episode selection holdout;
- train exactly 500 paired V1 and V1+semantic AveragePolicy steps;
- generate a completely new 2,000-episode fresh-target evaluation stream from
  the same frozen semantic Advantage ensemble under a different deterministic
  chance/episode seed;
- perform no model selection on that new stream.

Precommitted independent PASS:
1. semantic weighted CE at least 10% below paired V1;
2. semantic weighted TV at least 15% below paired V1;
3. semantic high-card/no-draw ALL_IN absolute bias at least 50% below paired
   V1;
4. semantic weighted CE at least 10% below its own step-0 value.

A PASS would remove the post-hoc milestone-selection concern and authorize the
design of one bounded online semantic-CFR pilot.  It would still not authorize
production.

## Independent 500-step fresh semantic distillation PASS — 2026-09-26

The post-hoc milestone concern is now closed.  The 500-step budget was frozen
before a completely new fresh-target evaluation stream was generated.

Independent evaluation:
- **2,000** new 3H episodes;
- **7,453** fresh strategy decisions;
- independent seed **1876407691**;
- the previously inspected 6,095-sample selection holdout was discarded and
  did not participate in this confirmation.

At step 0, both AveragePolicy arms remain functionally identical:
- weighted CE ~**1.10881428**;
- weighted TV ~**0.48558553**;
- high-card/no-draw ALL_IN prediction **0.162329** vs target **0.073662**.

At the precommitted 500 steps:
- weighted CE: **0.619981 V1 -> 0.466885 semantic** (~**24.7%** lower);
- weighted TV: **0.177349 -> 0.112830** (~**36.4%** lower);
- high-card/no-draw ALL_IN target: **0.073662**;
- V1 prediction: **0.243481**, abs bias **0.169818**;
- semantic prediction: **0.079953**, abs bias **0.006291**
  (~**96.3%** lower).

All four independent precommitted criteria pass:
- semantic CE >=10% better than V1;
- semantic TV >=15% better than V1;
- semantic high-card/no-draw ALL_IN bias >=50% better than V1;
- semantic CE >=10% better than its own step-0 value.

Fixed DC1 remains directionally favorable versus the paired V1 fresh-target
distill:
- 17 no-draw high-card states: ALL_IN mean
  **51.45% V1 -> 33.95% semantic**;
- all 23 high-card states: **54.29% -> 33.81%**;
- Q8/884 Fold: **48.42% V1 -> 0.031% semantic**.

This independently confirms the complete representation-capacity bridge:
semantic features improve Advantage fitting, semantic Advantage creates a
different/fresher strategy target distribution, and a semantic AveragePolicy
distills that distribution substantially better than V1 on unseen episodes.

The representation-diagnosis phase is therefore closed.  The next gate is
online feedback, not another post-hoc representation test.

### Bounded online semantic-CFR pilot

Added:
- `tools/run_3h_semantic_online_pilot_10105.py`;
- `tools/run_3h_semantic_online_pilot_10105.sh`.

Frozen pilot contract:
- source: read-only 10105 checkpoint;
- domain: THREE_HANDED only; HU is not trained;
- initial behavior: the passed eight-member semantic Advantage shadow;
- **2 online iterations**, **64 roots/iteration**;
- exact canonical lean root recursion/action resolver;
- after each root block: eight fresh **1600-step** semantic Advantage fits,
  reusing the exact validated member init/batch seed contract;
- the fixed controlled-split 50k positions remain excluded from Advantage
  minibatch sampling, matching the validated shadow-fit contract;
- no historical AveragePolicy reservoir is reused for the tail-policy bridge;
- after online feedback, the final ensemble generates a new 8,000-episode
  fresh strategy stream;
- AveragePolicy budget is frozen at the independently confirmed **500 steps**;
- final policy is evaluated on a separate 2,000-episode fresh-target stream.

Precommitted Advantage/DC1 non-regression rules:
- member-pairwise TV no more than 10% above initial semantic ensemble;
- member argmax disagreement no more than 10% above initial;
- no-draw raw-ensemble ALL_IN mass may not rise by >10 percentage points;
- no-draw member-mixture ALL_IN mass may not rise by >10 percentage points.

Precommitted post-online AveragePolicy rules reuse the independent confirmation:
- semantic CE >=10% better than paired V1;
- semantic TV >=15% better than paired V1;
- high-card/no-draw ALL_IN absolute bias >=50% better;
- semantic CE >=10% better than its own step-0 value.

PASS authorizes a longer **research-only** semantic continuation.  It does not
authorize production, sealed-strength claims, or DC2.

## Bounded 3H semantic online CFR pilot PASS — 2026-09-27

The first real online-feedback gate passed all precommitted criteria.

Pilot contract/result:
- source checkpoint: read-only iteration **10105**;
- completed iteration: **10107**;
- 3H only; HU training was not performed;
- 2 online iterations;
- 64 roots/iteration;
- 8 semantic Advantage members x 1600 steps after each root block;
- fixed 50k controlled-split positions remained excluded from Advantage
  minibatch fitting;
- source checkpoint hash remained unchanged.

Online root additions:
- 10106: 985 Advantage samples, 6,354 nodes;
- 10107: 1,131 Advantage samples, 7,125 nodes.

### Advantage/DC1 after online feedback

The semantic ensemble remains stable enough under real feedback:
- pairwise policy TV: **0.39778 -> 0.41561** (+4.48%, inside +10% gate);
- pairwise argmax disagreement: **0.40751 -> 0.42254** (+3.69%, inside
  +10% gate).

The previously suspicious high-card surface improves further:
- all 23 high-card states, member-policy ALL_IN:
  **27.84% -> 24.54%**;
- all 23 raw-ensemble ALL_IN:
  **33.28% -> 28.64%**;
- 17 no-immediate-draw states, member-policy ALL_IN:
  **22.64% -> 18.71%**;
- 17 no-draw raw-ensemble ALL_IN:
  **24.05% -> 18.70%**;
- no-draw majority-member ALL_IN argmax:
  **3/17 -> 2/17**;
- Q8/884 trips Fold remains **0%**.

Thus the semantic repair did not wash out when its own policy entered the CFR
feedback loop.

### Post-online fresh strategy distillation

The final online ensemble generated a new fresh strategy stream:
- 8,000 episodes / 29,062 decisions;
- train: 6,400 episodes / 23,174 samples;
- discarded selection holdout: 5,888 samples.

The already independently confirmed 500-step AveragePolicy budget was reused,
then evaluated on another independent post-online stream:
- 2,000 episodes / 7,293 decisions;
- high-card/no-draw subgroup: 305 samples.

Post-online independent evaluation:
- weighted CE: **0.65699 V1 -> 0.51286 semantic** (~21.94% lower);
- weighted TV: **0.18136 -> 0.11620** (~35.92% lower);
- high-card/no-draw ALL_IN target: **0.06540**;
- V1 prediction: **0.18723**, abs bias **0.12182**;
- semantic prediction: **0.07860**, abs bias **0.01320** (~89.17% lower).

All eight precommitted pilot criteria passed.

### Decision

The representation-diagnosis and bounded-online-admission phases are now
complete.  The V1+general-semantic lane is admitted to a **longer
research-only continuation**, but still not to production and not to DC2.

The next continuation must be resumable and preserve intermediate artifacts.
Because the first online pilot did not persist the mutated 3H reservoir itself,
the longer lane restarts deterministically from the frozen 10105 source and
replays 10106/10107 before advancing further.  Those first two iterations are
expected to reproduce the admitted pilot contract.

The research lane is frozen at **10 online 3H iterations total (10106..10115)**,
64 roots/iteration, eight semantic Advantage members x1600 steps/iteration,
with a rolling resume state saved after every completed iteration and compact
semantic-ensemble milestone snapshots at 10107, 10110 and 10115.  The source
10105 checkpoint and HU sidecar remain read-only.  This is long enough to expose feedback drift
without committing to another 1000-iteration-scale run.

### Resumable 10105→10115 semantic research runner

Added:
- `tools/run_3h_semantic_research_10105_10115.py`;
- `tools/run_3h_semantic_research_10105_10115.sh`.

The runner:
- restarts deterministically from frozen 10105 because the two-iteration pilot
  intentionally did not persist its mutated reservoir;
- requires the replayed 10107 DC1 summary to reproduce the admitted bounded
  pilot within 1e-6 before continuing;
- saves a rolling 3H Advantage-reservoir + sampler + semantic-ensemble resume
  artifact after every iteration;
- saves compact ensemble snapshots at 10107, 10110 and 10115;
- evaluates the final 10115 ensemble on the same fixed DC1 development states;
- generates a brand-new final semantic strategy stream and applies the already
  frozen 500-step AveragePolicy distillation;
- evaluates that policy on another independent 2,000-episode stream.

Precommitted final non-regression gates:
- pairwise policy TV and argmax disagreement no more than 15% above the initial
  admitted semantic ensemble;
- no-draw raw-ensemble and member-mixture ALL_IN mass no more than +10
  percentage points above initial;
- final tail semantic AveragePolicy CE >=10% better than V1;
- final tail semantic TV >=15% better than V1;
- final high-card/no-draw ALL_IN absolute bias >=50% better than V1;
- final semantic CE >=10% better than its own step-0 value.

A PASS advances to a larger development benchmark.  It still does not authorize
production or DC2.

## 3H semantic research continuation 10105→10115 PASS — 2026-09-27

The resumable 10-iteration research lane completed successfully:
- source 10105 checkpoint remained byte-identical;
- completed iteration: **10115**;
- 3H only; HU training not performed;
- 64 roots/iteration;
- eight semantic Advantage members x1600 steps/iteration;
- the fixed controlled-split 50k positions remained excluded from Advantage
  minibatches;
- all eight precommitted final criteria passed.

The deterministic replay guard also reproduced the admitted 10107 bounded pilot
**exactly** (all four checked summary values had absolute difference 0).

### Online Advantage/DC1 trajectory

Initial admitted semantic ensemble:
- pairwise TV **0.39778**;
- argmax disagreement **0.40751**;
- 17 no-draw high-card raw-ensemble ALL_IN **0.24049**;
- 17 no-draw member-mixture ALL_IN **0.22639**.

Final 10115:
- pairwise TV **0.41851** (+5.21%);
- argmax disagreement **0.42401** (+4.05%);
- no-draw raw-ensemble ALL_IN **0.20992** (-3.06 p.p.);
- no-draw member-mixture ALL_IN **0.23217** (+0.58 p.p.);
- no-draw majority-member ALL_IN argmax remains **3/17**;
- Q8/884 trips Fold remains **0%**.

The lane is noisy iteration-to-iteration rather than monotonic. 10110 is the
cleanest milestone on several diagnostics (TV ~0.40265, no-draw raw ALL_IN
~0.15856), while 10115 drifts back upward, but the final state remains inside
all frozen non-regression bounds.  Therefore no post-hoc milestone promotion is
made; 10115 is retained as the precommitted final research endpoint.

### Final fresh-target AveragePolicy bridge

Final 10115 semantic behavior generated:
- 8,000 fresh 3H episodes / **28,418** strategy decisions;
- train: 6,400 episodes / **22,746** samples;
- discarded selection holdout: 5,672 samples.

A new independent evaluation used:
- 2,000 episodes / **7,081** decisions;
- high-card/no-draw subgroup: **233** samples.

At the frozen 500-step policy budget:
- weighted CE: **0.61136 V1 -> 0.47476 semantic** (~22.34% lower);
- weighted TV: **0.16719 -> 0.10942** (~34.55% lower);
- high-card/no-draw target ALL_IN: **0.07251**;
- V1 prediction **0.20543**, abs bias **0.13292**;
- semantic prediction **0.07887**, abs bias **0.00636**
  (~95.21% lower);
- semantic CE improves ~56.48% versus its own step-0 value.

DC1 descriptive tail-policy diagnostics also remain directionally favorable:
- 17 no-draw high-card states: ALL_IN **49.88% V1 -> 29.52% semantic**;
- all 23 high-card states: **48.06% -> 31.72%**;
- Q8/884 Fold **47.24% V1 -> 0.55% semantic**.

### Decision and next gate

The representation diagnosis, independent fresh-target confirmation, bounded
online pilot and 10-iteration online research continuation have all passed.
The next step is therefore a **larger DC1 development benchmark**, not more
representation diagnostics.

DC0 real-OpenHoldem parity is still pending, so this benchmark remains
mechanical/development-only and cannot create a canonical strength claim.

Added:
- `tools/evaluate_deepcrusher_dc1_semantic_candidate.py`;
- `tools/run_deepcrusher_dc1_semantic_1k.sh`.

The next gate runs **1,000 identical empirical scenarios** twice:
1. unchanged 10105 baseline (3H finalized V1 AveragePolicy + HU ENS8);
2. 10115 semantic candidate (3H semantic tail AveragePolicy + the exact same HU
   ENS8).

Both arms use the same scenario/deal seed sequence and the same DeepCrusher
translation.  The comparison reports:
- baseline and semantic absolute development EV vs DeepCrusher;
- scenario-cluster paired **semantic-minus-baseline** delta, especially 3H;
- exact HU margin parity as an implementation guard;
- decision-sanity flag counts/rates for both arms.

Interpretation rule:
- positive 3H delta with lower CI bound >0 is strong development evidence and
  supports scaling DC1 further;
- positive mean with CI crossing 0 supports a larger 5k development sample;
- negative mean or new anomaly regression triggers diagnosis before scale-up.

No outcome of this 1k gate authorizes production or DC2 while DC0 canonical
parity remains incomplete.

## DC1 semantic 1k pre-benchmark blocker: native nouts/NOuts collision — 2026-09-27

The first 1,000-scenario paired DC1 attempt stopped in the **unchanged 10105
baseline arm**, before the semantic-candidate arm started.  The DeepCrusher
offline OpenPPL oracle reached stock library helper `NOuts`; inside
`NOutsFlop`, the stock library references lowercase native OpenHoldem symbol
`nouts`.  The offline provider had not implemented native `nouts`, so the
case-insensitive library fallback resolved lowercase `nouts` back to library
`NOuts`, producing a recursive-library cycle.

This is a DeepCrusher-oracle coverage/parity bug exposed by the larger sample,
not evidence about semantic-10115 strength.

OpenHoldem reference:
`CSymbolEngineCards::CalculateNumberOfOuts` enumerates every unseen card and
increments native `nouts` only when the added card strictly improves the
hero hand type, improves pokerval, and leaves the resulting hero hand type
strictly above the board-only hand type.  River returns zero through the
BETROUND < river guard.

Fixes:
- `DeepCrusherCardSymbols.native_nouts()` ports that OpenHoldem algorithm;
- `DeepCrusherPrimitiveSymbols` preserves the source-level distinction:
  exact lowercase `nouts` resolves natively while `NOuts` remains available
  to the pinned OpenPPL library;
- added `tools/test_deepcrusher_nouts_native_collision.py` with an
  AhKh/QhJh2c parity case (native nouts=18) and an end-to-end NOuts->NOutsFlop
  collision regression;
- the 1k runner now executes this guard before launching either benchmark arm.

The failed run produced no valid 1k comparison and must be rerun from scratch
after pulling the fix.

## Paired DC1 semantic-vs-baseline 1k: positive mean, inconclusive CI; strong-hand fold regression blocks scale-up — 2026-09-27

The paired 1,000-scenario development benchmark completed after the native
nouts/NOuts oracle fix.

Population:
- 1,000 identical scenario/deal seeds in both arms;
- 538 THREE_HANDED scenarios;
- 462 TRUE_HEADS_UP scenarios;
- HU scenario margins are exactly identical between baseline and semantic arms.

Absolute development EV vs frozen DeepCrusher R8:
- baseline 10105 3H: **-21.276 chips/policy-seat-hand**,
  95% CI **[-33.520, -9.032]**;
- semantic 10115 3H: **-13.460**,
  95% CI **[-22.920, -4.000]**.

Paired semantic-minus-baseline delta:
- THREE_HANDED: **+7.816**, 95% CI **[-2.410, +18.043]**, n=538;
- ALL: **+4.205**, 95% CI **[-1.300, +9.710]**, n=1000.

Thus the mean moves in the desired direction, but the precommitted lower-CI>0
rule does not pass.  This is not yet evidence of a reliable strength gain.
The paired 3H median delta is **-8.889** chips, with 221 positive, 298 negative
and 19 zero scenario deltas; a Wilcoxon signed-rank diagnostic is also neutral.
The positive mean is therefore not a broad scenario-by-scenario uplift and must
not be overinterpreted.

Most sanity surfaces improve substantially:
- deep high-card jam: **149 -> 63** total flags;
- top-pair fold: **3 -> 0**;
- preflop AA fold: **6 -> 0**;
- deep 72o jam: **8 -> 1**.

However POSTFLOP_TRIPS_PLUS_FOLD regresses **1 -> 6** sampled folds
(5 unique scenario/state patterns because one state repeats across paired
lineups).  The semantic examples are not merely tiny-probability tails:
- turn trips 73o on 7-8-J-7: Fold ~**75.3%**;
- river 65o making an 8-high straight: Fold ~**99.75%**;
- turn Q9 on 2-T-8-J making a Q-high straight: Fold ~**60.9%**;
- river 83o trips on 8-9-K-5-8: Fold ~**93.45%**;
- river 65o on four-spade board making a low-spade flush: Fold ~**96.21%**
  (same state sampled twice in paired lineups).

The Q9 turn-straight state is especially suspicious strategically and shows
that this is not adequately explained as one stochastic low-probability Fold.

### Decision

Do **not** scale directly to DC1 5k yet.  The original rule would otherwise
send a positive-mean / CI-crossing-zero result to 5k, but the new strong-hand
fold surface is a material anomaly and must be attributed first.

Added:
- `tools/audit_dc1_semantic_strong_hand_surface.py`;
- `tools/run_dc1_semantic_strong_hand_attribution.sh`.

The audit replays only the semantic 1k arm and, on every 3H postflop
trips-or-better state where Fold is legal, compares on the exact same state:
1. finalized 10105 V1 AveragePolicy Fold probability;
2. semantic 10115 tail AveragePolicy Fold probability;
3. semantic 10115 Advantage ENS8 regret-matched Fold probability.

It hard-checks that the replay reproduces the six sampled strong-hand folds.
Interpretation:
- Advantage low + tail high => AveragePolicy distillation/generalization bug;
- Advantage high + tail high => online CFR/Advantage target problem;
- both low => sampled-fold count mostly stochastic.

Only after this attribution should DC1 scale to 5k or the policy learner be
repaired.

## Strong-hand Fold attribution: AveragePolicy tail, not semantic Advantage — 2026-09-27

The exact 1,000-scenario semantic replay reproduced the six sampled
POSTFLOP_TRIPS_PLUS_FOLD events (five unique states) and decisively localizes
the regression.

Across **all 12** 3H postflop trips-or-better states where Fold was legal:
- semantic Advantage ENS8@10115 Fold probability: **0.0 on all 12**;
- finalized 10105 V1 AveragePolicy Fold mean: **8.13%**, max **21.00%**;
- semantic 10115 tail AveragePolicy Fold mean: **49.03%**, max **99.75%**;
- semantic tail expected folds across the 12 states: **5.88**;
- six folds were actually sampled.

By category:
- FLUSH (3 states): Advantage Fold 0%; semantic tail mean **74.34%**;
- STRAIGHT (2): Advantage Fold 0%; semantic tail mean **80.33%**;
- TRIPS (7): Advantage Fold 0%; semantic tail mean **29.24%**.

By street:
- flop (5): tail Fold mean **13.28%**;
- turn (3): **45.44%**;
- river (4): **96.41%**;
while Advantage Fold remains 0 throughout.

The five unique problem states include tail Fold ~75.3%, 99.75%, 60.9%,
93.45% and 96.21%; Advantage Fold is exactly 0 in every one.

### Conclusion

The online semantic CFR/Advantage lane is **not** the source of this regression.
The failure is introduced downstream by the fresh-target semantic AveragePolicy
distillation/generalization step.  Do not modify regret matching or the
semantic Advantage learner in response to these folds.

Before designing a repair, quantify the exact final 10115 strategy-training
coverage.  The next audit regenerates:
- the exact 8,000-episode / 22,746-sample final fresh train stream used for the
  500-step tail policy;
- a new 10,000-episode independent target stream.

It measures, specifically for postflop trips-or-better states where Fold is
legal:
- target Fold mass from semantic Advantage;
- V1 and semantic-tail Fold predictions;
- category/street breakdown;
- exact number of times each strong-hand train sample was selected by the
  frozen 500-step uniform minibatch stream.

Decision logic is frozen:
- material target Fold => revisit Advantage targets;
- low targets + high tail prediction on the train distribution =>
  class-imbalance/objective-allocation underfit;
- train fit good but new-stream fit bad => target-data diversity/generalization;
- unseen strong samples in the 500-step minibatch stream => exposure gap;
- broad fresh streams clean but DC1 bad => narrow distribution shift.

No 5k DC1 scale-up until this coverage diagnosis selects the repair mechanism.

## 10115 strong-hand coverage audit: independent sample-size guard — 2026-09-27

The first coverage audit completed the exact 8,000-episode training-stream
reconstruction and a new 10,000-episode independent stream, but found only
**44** postflop trips-or-better / Fold-legal independent states.

The audit had precommitted to at least **100** independent strong-hand states
before interpreting train-vs-generalization behavior, so it correctly stopped
without producing a diagnosis.

Do not lower the 100-state requirement after observing 44.  The strong-hand
surface is sparse: 44 qualifying states appeared in 10,000 independent
episodes (35,648 decisions).  The independent horizon is therefore increased
to **30,000 episodes**, which should yield roughly 132 qualifying states if the
observed rate remains similar, while preserving the same >=100 gate.

The script now also prints train/eval strong-hand counts before the guard so a
future stop is directly diagnosable.  No model, checkpoint, or strategy was
modified by the failed audit.

## 10115 strong-hand coverage audit: diversity/generalization failure confirmed — 2026-09-27

The 30,000-episode independent coverage audit completed and confirms the exact
failure mode of the semantic 10115 tail AveragePolicy.

Training stream:
- 22,746 ordinary fresh strategy samples;
- only **32** postflop trips-or-better / Fold-legal samples;
- every one of those 32 samples was actually seen by the frozen 500-step
  minibatch stream: min **14**, median **24**, mean **23.625**, max **31** draws;
- therefore this is not a minibatch-exposure gap.

On those 32 training strong-hand samples:
- teacher target Fold weighted mean **2.68%**;
- semantic tail Fold weighted mean **5.30%**;
- V1 Fold weighted mean **11.62%**.

Thus the semantic tail fits its sparse strong-hand training support reasonably
well.

Independent stream:
- 30,000 episodes / 107,228 strategy decisions;
- **148** trips-or-better / Fold-legal states.

On those 148 unseen strong-hand states:
- teacher target Fold weighted mean **2.16%**;
- semantic tail Fold weighted mean **37.36%**;
- V1 Fold weighted mean **15.37%**;
- semantic-tail strong-hand Fold absolute bias **35.21 p.p.**;
- expected semantic-tail folds **55.30** vs teacher **3.20**.

The failure is especially severe for:
- straights: teacher 0%, semantic tail **55.24%**;
- flushes: teacher 0%, semantic tail **56.69%**;
- river strong hands: teacher **6.64%**, semantic tail **83.60%**.

The audit's frozen diagnosis is therefore:
`TAIL_FITS_TRAIN_BUT_GENERALIZES_POORLY_TO_NEW_STRONG_HAND_STATES`.

The likely repair direction is **fresh-target diversity for rare strong-made
hands**, not more replay of the same 32 examples and not an Advantage/CFR
change.

### Semantic fallback contract issue discovered before repair

While preparing that repair, the shared diagnostic helper
`semantic_sigma()` was re-audited against the canonical functional deployment
contract.  A mismatch was found:

- the historical semantic research helper used a **uniform** distribution when
  all legal averaged raw Advantages were non-positive;
- canonical `lean_regret_matching_policy()` uses a **softmax over legal raw
  values** in that case.

This helper was used not only for fresh-target generation but also as the
behavior policy during the semantic online feedback lane.  Therefore no further
tail-policy repair or DC1 scale-up should proceed until the materiality of this
mapping mismatch is quantified.

The helper is now corrected to call the canonical
`lean_regret_matching_policy()` exactly.

Added:
- `tools/audit_3h_semantic_fallback_contract.py`;
- `tools/run_3h_semantic_fallback_contract.sh`.

The audit replays the historical (old-uniform) trajectory semantics and compares
them state-by-state with the canonical lean softmax fallback for:
- the initial semantic shadow;
- semantic milestones 10107 and 10110;
- the exact historical 10115 8,000-episode final tail-target stream, which must
  reproduce 28,418 decisions.

Frozen materiality rule:
- if every audited all-nonpositive fallback fraction <=0.1% and same-random-u
  sampled-action difference <=0.05%, treat the mismatch as immaterial to this
  lane;
- otherwise replay the semantic online lane from frozen 10105 under canonical
  lean semantics before making any repair/promotion decision.

This contract audit now blocks the strong-hand diversity repair and DC1 5k.

## Semantic fallback-contract audit: material lane mismatch, canonical replay required — 2026-09-27

The historical semantic helper mismatch is **material under the precommitted
contract**.

Across audited trajectories, the historical all-nonpositive fallback appears in
roughly 6–7% of decisions:
- initial semantic shadow: 7.223%;
- semantic 10107: 6.207%;
- semantic 10110: 6.567%;
- exact historical 10115 tail-target stream: 6.886% (1,957 / 28,418).

The resulting policy-distance per fallback state is numerically small
(mean TV ~0.16–0.19% on fallback-only states), and even on the exact 10115
8,000-episode stream only 9 / 28,418 sampled actions differ when the same
uniform random draw is reused (~0.0317%).  However the frozen materiality rule
was OR-based:
- fallback fraction must be <=0.1%; and
- same-u action-difference fraction must be <=0.05%.

The first condition fails by a very large margin, therefore the canonical replay
is mandatory before any promotion or AveragePolicy repair.

Important localization:
- **strong-hand states are not affected by this fallback mismatch** in the
  audited trajectories;
- strong-hand fallback count is 0 at initial, 10107, 10110 and 10115;
- therefore the previously diagnosed strong-hand AveragePolicy
  diversity/generalization defect remains independently supported and is not
  explained away by the fallback issue.

The shared semantic helper has already been corrected to call the canonical
`lean_regret_matching_policy()`, including its softmax fallback.

### Canonical 10105→10115 replay

Added:
- `tools/run_3h_semantic_research_canonical_10105_10115.sh`.

The existing research driver now accepts `--canonical-replay`, which:
- starts from the same frozen 10105 checkpoint and the same initial semantic
  Advantage shadow;
- uses the corrected canonical lean behavior contract throughout all ten online
  iterations;
- writes to an isolated run directory
  `runs/3h_semantic_research_canonical_10105_10115`;
- does **not** reuse the historical 10115 resume artifact;
- treats the old 10107 exact-reproduction numbers as descriptive only, because
  changing the behavior mapping is the intervention being tested;
- preserves the same 64 roots/iteration, eight members x1600 steps, protected
  50k split, milestone schedule and final 500-step tail-policy budget;
- preserves the same final non-regression gates and independent fresh-target
  evaluation.

Only after the canonical replay completes should the 10115 strong-hand coverage
audit be repeated against the canonical 10115 ensemble/tail policy.  DC1 5k
remains blocked.

## Canonical semantic 10105→10115 replay PASS — 2026-09-27

The isolated replay under the corrected functional behavior contract completed
successfully.

Contract/integrity:
- `behavior_contract = CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK`;
- `canonical_replay = true`;
- completed iteration **10115**;
- source 10105 checkpoint remained byte-identical;
- HU was not trained;
- all eight previously frozen continuation/tail-policy gates passed.

The canonical path diverges from the old admitted historical-uniform path as
expected.  At 10107 the old bounded-pilot summary is no longer reproduced
exactly, and is therefore retained only as descriptive historical comparison.

Final canonical Advantage/DC1:
- pairwise TV: **0.39778 initial -> 0.40108 final**;
- argmax disagreement: **0.40751 -> 0.40545**;
- all 23 high-card raw-ensemble ALL_IN:
  **0.33280 -> 0.26605**;
- 17 no-draw raw-ensemble ALL_IN:
  **0.24049 -> 0.17550**;
- 17 no-draw member-mixture ALL_IN:
  **0.22639 -> 0.20060**;
- strong-hand Fold at the tracked trips state remains **0** in both raw
  ensemble and member mixture.

Final canonical fresh-target tail-policy bridge:
- 8,000 episodes / **28,427** decisions;
- train: **22,726** samples;
- discarded holdout: **5,701** samples;
- independent eval: 2,000 episodes / **7,087** decisions;
- high-card/no-draw subgroup: 272.

At frozen 500 steps:
- CE: **0.59538 V1 -> 0.44687 semantic** (~24.94% lower);
- TV: **0.17205 -> 0.10672** (~37.97% lower);
- high-card/no-draw target ALL_IN: **0.04002**;
- V1 prediction: **0.17244**, abs bias **0.13242**;
- semantic prediction: **0.06129**, abs bias **0.02127**
  (~83.94% lower than V1).

Thus the semantic representation/online-feedback lane remains healthy under the
correct canonical lean fallback.

### Next gate: canonical strong-hand coverage

The historical strong-hand generalization diagnosis must now be repeated on the
canonical 10115 artifacts before designing any tail-policy repair.

Added:
- `tools/run_3h_semantic_tail_strong_hand_coverage_canonical_10115.sh`.

The canonical final train stream has a different deterministic size
(22,726/5,701 rather than 22,746/5,672), so the existing coverage audit was
parameterized with explicit expected train/holdout counts while preserving the
historical defaults.

The canonical audit preserves the same:
- exact 8,000-episode train-stream reconstruction;
- 30,000-episode independent evaluation;
- >=100 independent strong-hand-state gate;
- minibatch exposure reconstruction;
- frozen diagnosis logic.

DC1 5k and any AveragePolicy repair remain blocked until this canonical
coverage result is known.

## Canonical 10115 strong-hand coverage: diversity/generalization failure reconfirmed — 2026-09-27

The strong-hand coverage audit was repeated on the fully canonical
softmax-fallback 10115 lane and reaches the same diagnosis.

Canonical ordinary final train stream:
- 22,726 strategy samples;
- only **44** postflop trips-or-better / Fold-legal states;
- every one was actually seen repeatedly by the frozen 500-step optimizer:
  min 15, median 23, mean 23.34, max 32 draws;
- zero strong samples were missed by minibatch sampling.

On those 44 train states:
- teacher Fold weighted mean **7.35%**;
- semantic tail Fold weighted mean **10.00%**;
- V1 Fold weighted mean **19.90%**.

Thus the tail policy fits its sparse strong-hand training support reasonably
well.

A completely independent 30,000-episode stream contained 162 strong Fold-legal
states:
- teacher Fold weighted mean **3.04%**;
- semantic tail Fold weighted mean **24.73%**;
- V1 Fold weighted mean **14.71%**;
- semantic-tail absolute Fold bias **21.69 p.p.**.

Category/street failures remain broad:
- flushes: teacher **0%**, semantic tail **42.57%**;
- straights: teacher **0%**, semantic tail **25.91%**;
- river strong hands: teacher **6.90%**, semantic tail **56.98%**.

Therefore the frozen diagnosis again is:
`TAIL_FITS_TRAIN_BUT_GENERALIZES_POORLY_TO_NEW_STRONG_HAND_STATES`.

The canonical fallback repair improved the magnitude of the failure versus the
historical lane, but did not remove it.  The AveragePolicy tail remains the
blocker; the semantic Advantage lane remains the teacher.

### Controlled diversity-repair gate

Added:
- `tools/audit_3h_semantic_strong_diversity_repair_10115.py`;
- `tools/run_3h_semantic_strong_diversity_repair_10115.sh`.

The experiment is designed to separate unique-state diversity from simple
strong-hand reweighting.

All arms:
- start from the exact finalized 10105 AveragePolicy + optimizer state;
- use the same semantic architecture;
- train for exactly 500 steps;
- preserve the canonical 10115 Advantage ENS8 teacher.

Arms:
1. BASELINE: exact canonical 22,726-sample ordinary train stream;
2. REPEAT_CONTROL: BASELINE + 256 extra slots made by repeating the existing
   44 strong-hand train states;
3. DIVERSE_STRONG: BASELINE + 256 genuinely new strong-hand states drawn from
   an independent 60,000-episode teacher stream.

REPEAT_CONTROL and DIVERSE_STRONG have equal dataset length, equal strong-class
weight, equal optimizer budget and the same minibatch-index sequence.  Their
only intended difference is whether the 256 extra strong slots contain repeated
known states or new strong-state diversity.

The candidate is evaluated on a **new** 30,000-episode seed that was not used
for the prior diagnosis or augmentation collection.

Frozen PASS requirements:
- >=150 new independent strong states;
- strong-hand Fold absolute bias >=30% better than canonical baseline;
- strong-hand Fold bias >=15% better than equal-weight repeat control;
- straight/flush strong bias >=30% better than canonical baseline;
- river strong bias >=30% better than canonical baseline;
- global CE and TV no more than 2% worse than canonical baseline;
- high-card/no-draw ALL_IN absolute bias no more than +1 percentage point worse
  than canonical baseline.

The exact canonical 500-step tail model must also be reproduced to <=2e-6
max parameter drift before the intervention arms are trained.

PASS authorizes an exact DC1 1k replay of the diversity candidate.  It still
does not authorize production, DC2 or a 5k benchmark.

## Strong-hand diversity repair PASS — 2026-09-27

The controlled diversity-vs-repeat experiment passed all frozen criteria.

Canonical baseline on the new untouched 30,000-episode evaluation:
- 160 strong Fold-legal states;
- teacher Fold weighted mean **6.55%**;
- canonical semantic tail Fold weighted mean **28.87%**;
- strong-hand absolute Fold bias **22.32 p.p.**;
- straight/flush bias **33.61 p.p.**;
- river strong bias **26.98 p.p.**.

Equal-weight repeat control:
- same +256 strong slots as the candidate, but created only by repeating the
  original 44 train states;
- strong-hand absolute Fold bias remains **21.30 p.p.**.

Diverse candidate:
- +256 genuinely new teacher-labelled strong states;
- selected from **366** novel candidates found in an independent 60,000-episode
  teacher stream;
- strong-hand absolute Fold bias falls to **6.91 p.p.**
  (~69.0% better than baseline, ~67.6% better than repeat control);
- straight/flush bias falls to **12.94 p.p.** (~61.5% better);
- river strong bias falls to **10.91 p.p.** (~59.6% better).

Global cost remains inside the frozen tolerance:
- CE ratio diverse/baseline **1.0084** (+0.84%);
- TV ratio **1.0191** (+1.91%);
- high-card/no-draw ALL_IN abs bias worsens by only **0.248 p.p.**.

The exact canonical 500-step tail was reproduced with max parameter difference
**0.0** before intervention, confirming experiment identity.

All eight precommitted criteria passed.

### Interpretation

The failure mechanism is now causally isolated at the experiment level:
**new strong-state diversity**, not merely additional class weight, is what
repairs the unseen strong-hand Fold surface.

The repeat control received exactly the same number of extra strong-hand slots,
same total dataset size, same 500-step optimizer budget and same minibatch index
sequence, yet barely improved the strong-hand bias.  The diverse arm produces a
large reduction while preserving global fit.

This candidate is still diagnostic-only and not promoted.

### Next gate: exact three-arm DC1 1k replay

Added:
- updated `tools/evaluate_deepcrusher_dc1_semantic_candidate.py` to accept the
  strong-diversity tail-candidate schema;
- `tools/run_deepcrusher_dc1_semantic_strong_diversity_1k.sh`.

The benchmark runs the exact same 1,000 scenario/deal seeds for:
1. unchanged 10105 baseline;
2. canonical 10115 semantic tail;
3. 10115 strong-diversity candidate.

HU ENS8 is identical in all three arms.

The comparison reports:
- absolute 3H development EV vs DeepCrusher for all three;
- paired candidate-minus-baseline, canonical-minus-baseline and
  candidate-minus-canonical deltas;
- exact HU parity;
- sanity-flag counts/rates for all three.

The key repair checks are whether the candidate:
- brings POSTFLOP_TRIPS_PLUS_FOLD back to no worse than baseline and no higher
  than canonical;
- avoids reintroducing high-card jams, top-pair folds or preflop AA folds;
- does not lose paired 3H value versus the canonical tail.

A clean 1k replay is required before any 5k scale-up.

## DC1 1k strong-diversity replay: partial repair, not clean enough for 5k — 2026-09-28

The three-arm exact DC1 development replay completed on identical 1,000
scenario/deal seeds:
- 538 THREE_HANDED;
- 462 TRUE_HEADS_UP;
- HU scenario margins are exactly identical across baseline, canonical and
  diversity arms.

Absolute 3H development EV vs frozen DeepCrusher R8:
- baseline 10105: **-21.276** chips/policy-seat-hand,
  95% CI **[-33.520, -9.032]**;
- canonical 10115: **-14.823**,
  95% CI **[-24.425, -5.221]**;
- +256 strong-diversity candidate: **-13.359**,
  95% CI **[-23.239, -3.479]**.

Paired 3H deltas:
- canonical - baseline: **+6.453**,
  95% CI **[-3.700, +16.606]**;
- diversity - baseline: **+7.918**,
  95% CI **[-2.120, +17.955]**;
- diversity - canonical: **+1.465**,
  95% CI **[-1.121, +4.051]**.

Thus the diversity candidate improves the mean again, but none of these paired
3H intervals has a positive lower bound.

Sanity flags:
- deep high-card jam: baseline **149**, canonical **60**, diversity **64**;
- top-pair fold: **3 / 1 / 1**;
- preflop AA fold: **6 / 0 / 0**;
- deep 72o jam: **8 / 0 / 0**;
- trips-or-better fold: **1 / 6 / 4**.

The diversity repair therefore moves the intended strong-hand surface in the
right direction (6 -> 4 sampled folds) but does not restore it to baseline
(1), and it slightly regresses high-card-jam count versus canonical (60 -> 64).
The frozen repair indicators for those two surfaces fail, so the candidate is
not clean enough for a 5k scale-up.

The remaining sampled strong-hand folds are still high-probability policy
surfaces, not merely sampling tails:
- Q9 straight on turn 2-T-8-J: Fold **53.9%**;
- 83 trips on river 8-9-K-5-8: Fold **93.6%**;
- 65 low flush on river T-A-4-Q-J four-spade board: Fold **45.5%**
  (sampled twice in paired lineups).

Compared with the canonical tail, the +256 diversity intervention materially
reduces two of the three unique residual states but leaves a severe river-trips
outlier.

### Decision

Do not scale DC1 to 5k yet.

The broad independent repair audit already established that unique strong-state
diversity is the correct mechanism. The 1k replay shows that +256 random novel
states is not enough for the tail of the distribution. The next gate increases
novel support and stratifies it toward the broad pre-identified weak surfaces,
without using DC1 states as training data.

Added:
- `tools/audit_3h_semantic_stratified_strong_diversity_10115.py`;
- `tools/run_3h_semantic_stratified_strong_diversity_10115.sh`.

Frozen intervention:
- canonical ordinary 22,726-sample strategy train stream;
- +512 novel strong teacher states from a new 180,000-episode pool;
- quotas:
  - 128 river strong;
  - 160 non-river straight/flush;
  - 160 non-river trips;
  - 64 remaining novel strong;
- same 500 optimizer steps;
- no DC1 state is used in training/selection.

A new 40,000-episode evaluation compares:
- canonical tail;
- prior +256 diversity candidate;
- new +512 stratified candidate.

Frozen PASS requirements:
- >=150 strong states in the new eval;
- strong-hand Fold bias >=20% better than the +256 candidate;
- straight/flush bias >=20% better than +256;
- river strong bias >=20% better than +256;
- total strong bias >=60% better than canonical;
- global CE and TV no more than 2% worse than canonical;
- high-card/no-draw bias no more than +1 p.p. worse than canonical.

Only a PASS returns to exact DC1 1k. 5k remains blocked.

## Stratified +512 strong-diversity repair PASS — 2026-09-28

The second-stage diversity intervention passed all eight frozen criteria on a
new 40,000-episode evaluation stream.

Data/identity:
- canonical ordinary train stream reproduced exactly: 22,726 samples;
- canonical 500-step tail reproduced with max parameter drift **0.0**;
- independent augmentation pool: 180,000 episodes / 640,542 decisions;
- **1,039** novel strong states available;
- fixed +512 stratified selection:
  - 128 river strong;
  - 160 non-river straight/flush;
  - 160 non-river trips;
  - 64 remaining;
- new evaluation: 40,000 episodes / 142,178 decisions;
- **194** strong Fold-legal states.

Strong-hand Fold absolute bias on the new evaluation:
- canonical tail: **25.45 p.p.**;
- prior +256 diversity candidate: **11.08 p.p.**;
- +512 stratified candidate: **7.56 p.p.**.

Therefore the +512 stratified candidate improves:
- ~70.3% vs canonical;
- ~31.7% vs the already-passed +256 diversity candidate.

Targeted strata:
- straight/flush bias:
  - canonical **37.62 p.p.**;
  - +256 **17.52 p.p.**;
  - +512 stratified **10.78 p.p.**
  (~38.5% better than +256);
- river strong bias:
  - canonical **37.58 p.p.**;
  - +256 **14.04 p.p.**;
  - +512 stratified **9.96 p.p.**
  (~29.1% better than +256).

Global cost remains comfortably inside the frozen tolerance:
- CE ratio stratified/canonical **1.00317** (+0.32%);
- TV ratio **1.00967** (+0.97%);
- high-card/no-draw absolute-bias increase vs canonical only **0.588 p.p.**.

All precommitted criteria passed.

### Next gate: exact DC1 1k stratified replay

Added:
- evaluator support for schema
  `SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_DIVERSITY_TAIL_CANDIDATE_V1`;
- `tools/run_deepcrusher_dc1_semantic_stratified_1k.sh`.

To avoid needless recomputation, this runner reuses the previously completed
exact 1k baseline/canonical/+256 reports and sanity audits, validates their seed
and scenario identity, and executes only the new stratified arm on the same
1,000 scenario/deal seeds.  HU remains unchanged.

Frozen scale-up indicators:
- exact HU parity across all four arms;
- stratified POSTFLOP_TRIPS_PLUS_FOLD count no higher than baseline;
- high-card-jam rate no more than 10% above the +256 diversity arm;
- top-pair fold no higher than +256;
- preflop AA fold no higher than +256;
- deep 72o jam no higher than +256;
- stratified 3H paired mean vs baseline remains positive.

PASS authorizes **DC1 5k development scale-up only**.  It does not authorize
production, DC2 or a canonical strength claim while DC0 real-OpenHoldem parity
remains pending.

## DC1 1k stratified +512 replay: better again, but frozen scale-up gate still fails — 2026-09-28

The exact fixed-seed DC1 replay of the +512 stratified strong-diversity
candidate completed on the same 1,000 scenarios:
- 538 THREE_HANDED;
- 462 TRUE_HEADS_UP;
- exact HU scenario-margin parity preserved.

Absolute 3H development EV vs DeepCrusher:
- baseline 10105: **-21.276**;
- canonical 10115: **-14.823**;
- +256 diversity: **-13.359**;
- +512 stratified: **-12.729** chips/policy-seat-hand.

Paired 3H deltas:
- +512 stratified - baseline: **+8.548**,
  95% CI **[-1.745, +18.841]**;
- +512 stratified - canonical: **+2.095**,
  95% CI **[-0.517, +4.706]**;
- +512 stratified - +256 diversity: **+0.630**,
  95% CI **[-2.337, +3.597]**.

So the mean continues to move in the desired direction, but the paired lower
bound is still not positive.

Sanity flags:
- deep high-card jam: **149 / 60 / 64 / 68**
  (baseline / canonical / +256 / +512);
- top-pair fold: **3 / 1 / 1 / 1**;
- AA fold: **6 / 0 / 0 / 0**;
- deep 72o jam: **8 / 0 / 0 / 0**;
- trips-or-better fold: **1 / 6 / 4 / 3**.

The +512 candidate therefore improves the target anomaly again (6 -> 4 -> 3)
but still fails the frozen requirement of no more strong-hand sampled folds than
the baseline (3 > 1).  All other frozen 1k scale-up indicators pass.

The remaining three sampled folds are not broad regressions; they are three
specific states whose Fold probabilities have already been reduced sharply:
- turn Q9 straight on 2-T-8-J: canonical **89.28%**, +256 **53.89%**,
  +512 **24.84%**;
- river 83 trips on 8-9-K-5-8: **96.13% -> 93.58% -> 35.60%**;
- river 65 flush on T-A-4-Q-J four-spade board:
  **88.38% -> 45.48% -> 11.90%**.

Thus the intervention direction remains correct, but the rare-state tail is not
yet clean enough to satisfy the frozen exact-DC1 gate.  5k remains blocked.

### Next gate: full novel strong-state pool

The validated 180,000-episode augmentation run already contained **1,039**
unique novel strong Fold-legal states.  The +512 experiment intentionally used
only a stratified subset.  The next intervention uses **all 1,039** novel
teacher-labelled strong states, still without using any DC1 state for training.

Added:
- `tools/audit_3h_semantic_fullpool_strong_diversity_10115.py`;
- `tools/run_3h_semantic_fullpool_strong_diversity_and_dc1_10115.sh`;
- DC1 evaluator support for the full-pool candidate schema.

The combined runner is intentionally autonomous:
1. reconstruct exact canonical base train;
2. regenerate the deterministic 180k pool and require exactly 1,039 novel
   strong states;
3. train the +1,039 full-pool tail for the same 500 steps;
4. evaluate on a brand-new 50,000-episode stream;
5. only if all frozen offline criteria pass, automatically execute the exact
   fixed-seed DC1 1k candidate replay.

Offline frozen criteria:
- >=220 strong states in new eval;
- overall strong Fold bias >=15% better than +512;
- straight/flush bias >=15% better than +512;
- river bias >=15% better than +512;
- overall strong bias >=70% better than canonical;
- global CE and TV no more than 2% worse than canonical;
- high-card/no-draw bias no more than +1 p.p. worse than canonical.

Conditional DC1 scale-up gate remains:
- exact HU parity;
- trips-or-better sampled Fold count <= baseline;
- high-card-jam rate <=10% above +512;
- top-pair Fold, AA Fold and deep 72o jam no worse than +512;
- paired 3H mean vs baseline positive.

Only a full PASS authorizes DC1 5k development scale-up.

