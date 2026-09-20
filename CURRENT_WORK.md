# SpinCore Current Work

Date: 2026-09-19
Status: **LT2 STAGE B PASS — HU B400 BROAD GENERALIZATION PASS — HU-ONLY 400-STEP INTERVENTION FROZEN — 100-ITERATION ONLINE PILOT NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_JAMMER_FAI_CONTROLLED_REFIT_RESULT_20260919.md`
- `docs/LT2_HU_B400_BROAD_GENERALIZATION_RESULT_20260919.md`
- `docs/LT2_HU_B400_ONLINE_PILOT_20260919.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## B400 broad gate — PASS

Against JAMMER, all three deterministic Stage-B 400-step HU refits improve significantly versus production Stage B:

- R0: `+16.8266`, CI95 `[+12.8183,+20.8348]`;
- R1: `+10.8855`, CI95 `[+7.1343,+14.6368]`;
- R2: `+10.5116`, CI95 `[+6.4837,+14.5395]`.

All three are positive in absolute Jammer EV.

Against PASSIVE_CALLER all three also improve significantly.

Against UNIFORM_LEGAL none regresses significantly; R1 improves significantly.

Therefore B400 generalizes beyond the selected FAI cohort.

## Frozen intervention

Change only the implicated domain:

- THREE_HANDED Advantage fit: **100 steps unchanged**;
- TRUE_HEADS_UP Advantage fit: **400 steps**;
- K4: off.

Do not raise 3H to 400 without evidence.

The trainer now supports an explicit `hu_advantage_steps` override while preserving old-checkpoint compatibility.

## Active gate

Bounded online continuation from Stage B:

- iterations 7501..7600;
- +100 iterations;
- +60,000 roots;
- source Stage B remains read-only;
- post-pilot: full forensic HU policy-chain comparison of Stage B vs pilot, for both AveragePolicy and current Advantage behavior.

Primary requirement:
- current HU behavior must retain a resolved Jammer improvement after online feedback;
- no resolved material regression against the other weak baselines.

Holdout stays sealed.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_hu_b400_online_pilot.sh
```

Wait for `LT2_HU_B400_ONLINE_PILOT_PASS` or the first error.

Then send `SpinCore_LT2_HU_B400_online_pilot_summary.json`.

Do not extend beyond iteration 7600 yet.
