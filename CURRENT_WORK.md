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

