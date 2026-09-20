# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PRESERVED — HU B400 BROAD PASS — HU400/3H100 INTERVENTION FROZEN — 100-ITERATION ONLINE PILOT ACTIVE**
Date: 2026-09-19

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Resolved diagnosis

The Stage-B HU Jammer regression was traced through:

1. current Advantage behavior;
2. preflop facing-all-in;
3. wrong Stage-B overfold;
4. low-noise infoset action-gap degradation;
5. insufficient fresh fitting rather than poisoned Stage-B reservoir.

Controlled refit:
- 100 steps fails;
- 400 and 1600 recover.

Broad natural-HU generalization:
- all three B400 candidates significantly improve versus Stage B against JAMMER;
- all three significantly improve against PASSIVE_CALLER;
- none significantly regresses against UNIFORM_LEGAL.

## Frozen intervention

Use the smallest evidenced change:

- THREE_HANDED Advantage fit = 100;
- TRUE_HEADS_UP Advantage fit = 400;
- K4 = 1/off;
- no other poker semantics changed.

The trainer exposes a backward-compatible `hu_advantage_steps` field so old checkpoints load with the historical global budget unless the HU override is explicitly set on continuation.

## Why not resume long training yet

Fresh refits prove the signal is present in the reservoir, but they do not prove that the repair survives the online feedback loop.

A corrected HU behavior changes:

- traversal behavior;
- newly collected Advantage samples;
- sampled AveragePolicy targets;
- future reservoir composition.

That feedback must be tested before another large block.

## Active pilot

From Stage B:

- +100 iterations;
- +60,000 roots;
- target iteration 7600;
- 31 workers;
- concurrent fit;
- vectorized batches;
- HU 400 / 3H 100.

Stage B itself remains read-only.

## Automatic post-pilot gate

Full forensic HU policy-chain:

- seeds 20260920..20260925;
- 5000 scenarios/seed;
- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER;
- production Stage B versus pilot;
- both AveragePolicy and current Advantage behavior.

Primary:
- current behavior retains resolved Jammer improvement;
- no resolved material regression on other baselines.

Secondary:
- measure whether AveragePolicy has begun moving in the same direction.

## Next branch

If current behavior passes:
- decide whether to extend 400–500 more iterations based on AveragePolicy movement and policy-memory refresh.

If current behavior fails:
- no extension; inspect online target/reservoir dynamics.

Holdout `20261001..20261006` remains sealed.

## Immediate direction

1. Keep Stage A/B immutable.
2. Pull `main`.
3. Run `bash tools/run_lt2_hu_b400_online_pilot.sh`.
4. Send `SpinCore_LT2_HU_B400_online_pilot_summary.json`.
5. Do not train beyond iteration 7600 yet.
