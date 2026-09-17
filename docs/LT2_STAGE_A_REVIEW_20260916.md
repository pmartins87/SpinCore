# SpinCore LT2 Stage A review — 2026-09-16

Status: **PASS — resource gate healthy; learning curve positive overall/3H; HU still noisy/flat; one bounded fit-concurrency screen before Stage B**

## Stage A execution

Source line: finalized LT1 checkpoint at iteration 2000, copied into an isolated LT2 run directory.

Stage A completed iterations 2001–3000, adding 600,000 roots and reaching 1.8M roots total. Production execution remained 31 root workers, 8 parent Torch threads and vectorized batch construction. The trainer exited 0 and emitted `LT2_STAGE_A_PASS`.

Final Stage A checkpoint:

- path: `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`
- SHA256: `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`
- iteration: 3000
- size: 2.235843 GiB
- finalized save: 54.468 s
- reported trainer wall scope: 10,547.402 s = 2h55m47s

## Resource result

Stage A memory telemetry contains 179 one-minute/boundary samples. WSL exposed about 30.14 GiB RAM plus 8 GiB swap.

Observed:

- minimum `MemAvailable`: about **10.67 GiB**;
- maximum swap used: **0 GiB**;
- process maximum RSS from `/usr/bin/time -v`: about **16.3 GiB**;
- no OOM, no swap pressure, exit status 0.

This is a clear resource PASS for the current 2M-per-memory reservoir contract.

## Reservoir/checkpoint transition

Final counters:

- 3H roots: 981,000;
- HU roots: 819,000;
- 3H Advantage samples seen: 30,645,342;
- HU Advantage samples seen: 24,655,892;
- 3H AveragePolicy samples seen: 2,216,393;
- HU AveragePolicy samples seen: 820,667.

Both Advantage reservoirs were already saturated at the 2M capacity. During Stage A the 3H policy reservoir crossed 2M and therefore entered replacement regime. HU policy remains below capacity.

Checkpoint growth confirms that transition. From iteration 2100 through 2700 the file grew roughly 33–35 MiB per 100 iterations. After the 3H policy reservoir saturated, growth fell to roughly 9–11 MiB per 100 iterations (2700→3000). There is no evidence of runaway checkpoint growth.

## Learning-curve diagnostic

A fixed-seed 1,000-scenario comparison used exactly the same weak-opponent methodology for LT0 120k, LT1 1.2M and LT2-A 1.8M. Checkpoint-to-checkpoint deltas are descriptive because the current report does not compute a dedicated paired CI for checkpoint-vs-checkpoint differences.

### Uniform-legal opponents

Overall SpinCore cEV:

- LT0 120k: +10.836 chips/hand;
- LT1 1.2M: +15.873;
- LT2-A 1.8M: +17.015.

Overall paired gain versus the uniform-control Hero:

- +19.582 -> +24.618 -> +25.761 chips/hand.

3H cEV improved monotonically:

- +2.407 -> +8.898 -> +12.769.

3H paired gain likewise improved:

- +19.811 -> +26.302 -> +30.174.

HU remains positive but did not improve from LT1 to LT2-A:

- cEV +20.534 -> +23.898 -> +21.900;
- paired gain +19.317 -> +22.681 -> +20.683.

The LT2-A HU paired 95% CI remains barely above zero at its lower bound, so this is not a collapse, but it is not evidence of continued HU improvement either.

### Passive caller

Overall cEV moved toward break-even:

- -2.402 -> -0.553 -> -0.064.

3H cEV improved:

- +3.743 -> +6.013 -> +7.670.

HU stayed weak/noisy:

- -9.473 -> -8.108 -> -8.961.

No passive-caller paired result is statistically established as positive at 95% in the aggregate report.

### Jammer

Overall cEV improved monotonically:

- -5.498 -> -2.719 -> -1.298.

Overall paired gain improved:

- +15.040 -> +17.819 -> +19.240, with 95% CI above zero at all three checkpoints.

3H cEV and paired gain improved strongly:

- cEV +2.295 -> +4.045 -> +7.422;
- paired gain +17.895 -> +19.645 -> +23.022.

HU improved from LT0 to LT1 but was essentially flat/noisy afterward:

- cEV -14.465 -> -10.501 -> -11.331;
- paired gain +11.754 -> +15.717 -> +14.887.

## Decision

Stage A passes its intended infrastructure gate. The training line is healthy enough to continue and the weak-baseline learning curve gives positive evidence that additional training still helps, especially in 3H. There is no basis to restart or shrink reservoirs.

The main unresolved strategic signal is HU: additional 600k roots did not improve the fixed-seed HU point estimates versus any of the three weak families. That is not yet a failure because the HU intervals remain wide and the HU policy reservoir is still far below its 2M capacity.

At the observed Stage-A strategy-sample rate, HU needs roughly another 1.18M policy samples to reach 2M, corresponding to about 4.3k more iterations. Therefore the next natural training milestone is around iteration 7500, where the HU AveragePolicy reservoir should have crossed saturation.

Before spending roughly another half day of Ryzen time, run exactly one bounded read-only fit-concurrency benchmark. Stage A confirmed average CPU utilization remains low because the two domain fits are serial and fit remains the dominant phase. The new screen tests whether overlapping the independent 3H/HU optimizer loops yields >=5% fit-wall gain while preserving exact same-thread model hashes, losses and batch RNG states. If not, close this optimization branch and continue with the existing 31/8/vectorized path. If yes, implement and validate one full-iteration semantics-preserving candidate before Stage B.

Canonical benchmark wrapper: `tools/benchmark_lean_lt2_concurrent_fit.sh`.
