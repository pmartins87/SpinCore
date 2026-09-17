# SpinCore — Long-Training Plan

Status: **CANONICAL TRAINING DIRECTION — LT2 STAGE B PASS / LEARNING REVIEW PENDING**
Date: 2026-09-17

## Current state

The active learning line is continuous:

- LT0: 120k roots — calibration only;
- LT1: 1.2M roots — production-shaped milestone;
- LT2 Stage A: 1.8M roots — first policy-reservoir saturation gate PASS;
- production concurrent-fit path — exact semantic parity PASS;
- LT2 Stage B: 4.5M roots / iteration 7500 — **PASS**;
- all four 2M reservoirs — now in saturation/replacement regime;
- next gate — Stage A -> Stage B paired learning review.

Read `LT2_STAGE_B_RESOURCE_REVIEW_20260917.md` for the completed Stage B resource evidence.

## Core training contract

The functional line uses:

- empirical SpinGo 3H/HU/blind/stack sampling;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation;
- mature legacy action vocabulary;
- external-sampling Deep CFR with repaired all-nonpositive regret fallback;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per domain at milestone finalization.

Current admitted Ryzen execution profile:

- 31 root workers;
- one numerical-library thread per root worker;
- 8 parent Torch threads;
- vectorized neural batch construction;
- production `concurrent_fit` iteration mode.

The canonical sequential iteration path remains preserved for regression/portability.

## LT0 — calibration — DONE

120k roots proved the repaired pipeline trains, saves, resumes and plays complete 3H/HU hands. It is not a competitive-strength checkpoint.

## LT1 — production-shaped milestone — DONE

LT1 completed 2000 iterations / 1.2M roots and established the 2M-reservoir line and the first production-shaped Ryzen profile.

Preserve:

`/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt`

SHA256:

`beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`

## LT2 Stage A — first policy-reservoir saturation gate — PASS

Stage A continued the exact LT1 state through iteration 3000 / 1.8M roots.

Preserve:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`

SHA256:

`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`

At Stage A:

- both Advantage reservoirs were saturated;
- 3H AveragePolicy had crossed 2M;
- HU AveragePolicy was only 820,667;
- no swap was used;
- weak-baseline learning remained positive overall/3H but HU was flat/noisy.

That HU warning was deliberately deferred until HU policy saturation rather than treated as a ceiling.

## Concurrent-fit optimization — CLOSED / ADMITTED

The repeated fit screen showed a 1.312188x median fit speedup for 8-thread concurrent 3H/HU Advantage fitting versus sequential 8-thread fitting, with exact same-thread model/loss/RNG parity.

A full concept iteration and then the production function both passed exact semantic parity. Production `concurrent_fit` is therefore admitted and was used for Stage B.

## LT2 Stage B — all-reservoir saturation gate — PASS

Stage B continued from the exact iteration-3000 checkpoint to iteration 7500:

- +4500 iterations;
- +2.7M roots;
- 4.5M roots total;
- 31 root workers;
- 8 parent Torch threads;
- vectorized batches;
- production `concurrent_fit` path;
- checkpoint every 250 iterations.

Current continuation checkpoint:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`

SHA256:

`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`

Final state:

- 3H roots: 2,452,500;
- HU roots: 2,047,500;
- 3H Advantage seen: 76,144,669;
- HU Advantage seen: 62,622,782;
- 3H AveragePolicy seen: 5,549,800;
- HU AveragePolicy seen: 2,072,704.

Therefore all four memories have crossed their 2M capacity and are in replacement regime.

Resource/postvalidation result:

- `LT2_STAGE_B_POSTVALIDATION_PASS`;
- final checkpoint 2.623474 GiB;
- finalized save 72.938 s;
- minimum observed WSL MemAvailable 7.473 GiB;
- maximum observed swap 0 GiB;
- preserved Stage A source unchanged.

The original launcher's `KeyError: 'roots'` occurred only in its post-run validator after successful training/finalization. The validator is corrected and the completed run was independently audited; do not rerun Stage B.

### Timing note

Keep the two observed wall-time sources separate:

- trainer report `wall_seconds`: 41,375.819 s;
- `/usr/bin/time` elapsed: 10:50:23.

Do not average these or use either as a precise projection until the discrepancy is understood.

## Immediate strategic gate — paired Stage A -> Stage B learning review

Before any further training, run:

`tools/run_lt2_stage_b_learning_review.sh`

This is read-only policy evaluation. It compares Stage A (1.8M roots) with Stage B (4.5M roots) using:

- the same 1000 fixed-seed empirical scenarios;
- identical solver deal seeds;
- the same opponent families: uniform legal, passive caller and jammer;
- hero seat rotation;
- row-level preservation and a direct paired checkpoint-delta 95% CI clustered by scenario.

This improves on the earlier learning-curve comparison, whose checkpoint-to-checkpoint deltas were descriptive point-estimate differences without a dedicated paired CI.

Primary decision questions:

1. Does overall learning continue from 1.8M -> 4.5M roots?
2. Does 3H continue improving after its policy reservoir has long been saturated?
3. Does HU improve after finally crossing its policy-reservoir capacity?
4. Are any apparent gains/regressions distinguishable from fixed-scenario evaluation noise?

Weak baselines are regression/learning sentinels, not an exploitability or GTO proof.

## Later LT2 / LT3

Do not select the next training target yet.

If Stage B learning evidence is healthy, continue from the exact Stage B checkpoint through another bounded multi-million-root block. If HU materially regresses or the learning curve stalls, investigate training dynamics/architecture before spending more compute.

The eventual line may extend to tens/hundreds of millions of roots, but only while meaningful improvement remains measurable and resources/semantics remain healthy.

## Strength tracking

Primary future product-strength evidence:

- SpinCore vs faithful DeepCrusher R8 v22 paired chip-EV;
- HU and 3H separately;
- blind/stack/position breakdowns;
- later full Spin & Go tournament win rate after continuous tournament progression is frozen.

DeepCrusher oracle construction proceeds in parallel. Do not use a simplified imitation for canonical head-to-head claims.

## Reservoir policy

Do not shrink or enlarge the 2M reservoirs merely on intuition. Stage B has now established the first all-four-reservoir saturated operating point. The next decision should be driven by learning quality and resource behavior.

## Operational files

- `tools/run_long_training_lt1.sh` — historical LT1 launcher;
- `tools/run_long_training_lt2_stage_a.sh` — completed Stage A launcher;
- `tools/run_long_training_lt2_stage_b.sh` — completed Stage B launcher, post-run validator corrected;
- `tools/audit_completed_lt2_stage_b.sh` — completed Stage B audit, PASS;
- `tools/run_lt2_stage_b_learning_review.sh` — immediate read-only paired learning gate;
- `tools/evaluate_lean_strategy_quality_with_rows.py` — weak-baseline evaluator preserving row-level evidence;
- `tools/compare_lean_checkpoint_delta.py` — direct paired checkpoint-delta CI;
- `tools/run_lean_functional_training.py` — authoritative trainer.

## Immediate direction

1. Preserve both Stage A and Stage B checkpoints.
2. Do not train beyond iteration 7500 yet.
3. Pull current `main`.
4. Run `bash tools/run_lt2_stage_b_learning_review.sh`.
5. Review the Stage A report, Stage B report and direct paired checkpoint delta.
6. Only then decide the next training block or an architecture/training investigation.
7. Continue faithful DeepCrusher oracle construction in parallel.
