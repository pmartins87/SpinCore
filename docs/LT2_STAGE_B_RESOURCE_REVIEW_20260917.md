# SpinCore — LT2 Stage B resource/completion review

Date: 2026-09-17
Status: **PASS — COMPUTE COMPLETE / POSTVALIDATION PASS**

## Completed milestone

The LT2 Stage B continuation completed the exact LT2 learning line from iteration 3000 to iteration 7500.

Run directory:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911`

Final checkpoint:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`

Final checkpoint SHA256:

`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`

Preserved Stage A source remained byte-identical:

`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`

## Postvalidation result

`LT2_STAGE_B_POSTVALIDATION_PASS`

Validated:

- target iteration 7500;
- total roots 4,500,000;
- 3H roots 2,452,500;
- HU roots 2,047,500;
- 31 root workers;
- 8 parent Torch threads;
- vectorized batching;
- production `concurrent_fit` iteration path;
- finalized checkpoint metric present at iteration 7500;
- both policy reservoirs crossed the 2M capacity;
- preserved Stage A source unchanged.

## Final reservoir state

3H:

- Advantage seen: 76,144,669;
- AveragePolicy seen: 5,549,800;
- both memories are in replacement regime.

HU:

- Advantage seen: 62,622,782;
- AveragePolicy seen: 2,072,704;
- HU AveragePolicy crossed 2M during Stage B, which was the principal milestone target.

All four 2M memories are therefore now in, or have entered, saturation/replacement regime.

## Resource result

Audit summary:

- final checkpoint size: 2.623474 GiB;
- finalized save: 72.938 s;
- minimum observed WSL `MemAvailable`: 7.473 GiB;
- maximum observed swap used: 0 GiB;
- `/usr/bin/time` maximum RSS reported by the completed process: 20,638,976 KiB;
- kernel-reported swaps: 0.

The memory gate remains healthy. Stage B consumed more resident memory than Stage A, as expected after HU policy growth, but retained more than 7 GiB of observed available WSL memory and never entered swap.

## Timing note

Two wall-time sources disagree materially and must not be silently averaged:

- trainer report `wall_seconds`: 41,375.819 s;
- `/usr/bin/time` elapsed shown by the completed process: 10:50:23 (39,023 s).

The discrepancy is telemetry-only and does not affect completion semantics, checkpoint state or postvalidation. For future throughput claims, preserve both sources separately until the source of the clock discrepancy is understood.

## Wrapper bug closed

The original Stage B launcher completed training successfully but then raised `KeyError: 'roots'` in its post-run validator because it read `report["final"]` instead of `report["final"]["domains"]`.

The bug occurred after finalization/report/checkpoint creation. The launcher was corrected for future use, and `tools/audit_completed_lt2_stage_b.sh` independently validated the already-completed run. Stage B must not be rerun because of that historical wrapper error.

## Next gate

Resource/completion evidence is sufficient to move to the strategic-learning review, not to further training.

The next bounded gate is a fixed-seed Stage A versus Stage B paired checkpoint evaluation using the same empirical scenarios/deals/opponent families, with a direct paired CI for the checkpoint delta. The principal question is whether the earlier HU flat/noisy signal improved after HU AveragePolicy saturation.

Do not extend beyond iteration 7500 until that learning review is complete.
