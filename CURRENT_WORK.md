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

