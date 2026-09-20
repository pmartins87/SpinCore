# SpinCore Roadmap — active state 2026-09-20

## Active status

- original Jammer FAI underfit — **REPAIRED BY MORE COMPLETE HU FITTING**.
- AveragePolicy at 8000 vs Stage B Jammer — **RESOLVED IMPROVEMENT**.
- root open-jam inflation — **CONFIRMED**.
- 7600 -> 8000 reservoir effect — **REAL, ABOUT +8 pp ALL_IN UNDER MATCHED FRESH400 REFITS**.
- production +56 pp ALL_IN jump — **NOT EXPLAINED BY RESERVOIR EVOLUTION ALONE**.
- fresh 400-step current-policy fit instability — **CONFIRMED**.
- same-memory budget stability sweep 400/800/1600/3200 — **NEXT**.
- further root training — **PAUSED**.
- holdout — **SEALED**.

## Decision path

If a larger fit budget sharply stabilizes the mature 8000 reservoir:
- select the smallest stable budget;
- run broad EV/generalization before any online continuation.

If even 3200 steps remain materially seed-sensitive:
- do not keep buying optimizer steps;
- evaluate a small Advantage ensemble / other fit-stability mechanism.

No production semantic change is authorized yet.
