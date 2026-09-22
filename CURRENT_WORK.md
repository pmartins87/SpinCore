# SpinCore Current Work

Date: 2026-09-22
Status: **LT3 9105 -> 10105 UTILIZATION CONTINUATION READY / POST-9105 BATTERY INCONCLUSIVE — DEEPCRUSHER DC0 ACTIVE / OPENHOLDEM PAUSED**

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

A frozen continuation from finalized 9105 -> 10105 (+1000 iterations / +600,000 roots, projected ~23.2 h) is ready. It preserves source 9105 read-only, preserves raw milestone 9600, checkpoints every 50, touches no sealed holdout and cannot supersede 8100/8600/9105 from training evidence alone.

In parallel, DC0 now also includes:
- decision-level benchmark traces with hole cards, visible board, pot, to-call, stacks, exact action and sizing;
- a hand-level sanity audit queue for AA preflop folds, >=10bb 72o jams, top-pair folds, trips+ folds, monster folds and deep high-card jams;
- frozen environment profile `GGPoker_NoPT_NoNotes_V1`: GGPoker=true, other networks=false, named chair lookups=-1, log$=true, colour notes=0, PokerTracker unavailable=-1;
- prwin/prtie explicitly excluded from the environment profile because they are substantive equity/card symbols and still require faithful implementation.

Canonical local action now:

`bash tools/run_lt3_parallel_9105_10105.sh`

Expected early sentinel: `LT3_PARALLEL_9105_10105_PREFLIGHT_PASS`.
Expected final sentinel: `LT3_PARALLEL_9105_10105_TRAINING_PASS`.

Canonical engineering action remains DC0 -> DC1 -> DC2. Training may run concurrently because the external benchmark work is repository-side and does not require consuming the Ryzen trainer.


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
