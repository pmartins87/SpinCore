# R7.5.4 eligible-final inventory and lean strategic screen

Date: 2026-09-03

## Scope

This inventory covers the five postflop candidates that are actually eligible to win R7.5.4A: PF0, PF1, PF2, PF3 and PF4. It excludes `PF_DENSE_REFERENCE`, whose historical role is referee only (`eligible_to_win: false`).

All 30 eligible final cells already exist from historical Actions run `31804178848` at source SHA `457996944f76e9f1fa0475691df978f450259641`: 5 candidates × HU/3H × 3 training seeds. Each final has 160 roots and a finalized AveragePolicy.

## Cheap evidence already extracted

All 30 cells pass both the frozen advantage-fit and policy-fit gates. Thus there is no learning-fit failure that removes a candidate before strategic comparison.

Three-seed means by domain:

| candidate | domain | advantage NRMSE | policy TV | nodes/root | tree s/root | full s/root | unique aggressive branches/decision |
|---|---|---:|---:|---:|---:|---:|---:|
| PF0 | 3H | 0.5104 | 0.0333 | 2503.9 | 42.74 | 60.65 | 1.750 |
| PF1 | 3H | 0.5130 | 0.0336 | 3672.3 | 109.07 | 126.70 | 1.959 |
| PF2 | 3H | 0.5094 | 0.0326 | 4159.5 | 138.07 | 155.13 | 2.052 |
| PF3 | 3H | 0.5297 | 0.0340 | 3106.2 | 70.49 | 89.01 | 1.881 |
| PF4 | 3H | 0.5226 | 0.0333 | 2853.5 | 53.06 | 70.17 | 1.864 |
| PF0 | HU | 0.4816 | 0.0121 | 2762.9 | 3.43 | 19.99 | 1.899 |
| PF1 | HU | 0.4816 | 0.0162 | 5870.9 | 7.17 | 24.75 | 2.092 |
| PF2 | HU | 0.4953 | 0.0171 | 8915.4 | 10.96 | 29.03 | 2.172 |
| PF3 | HU | 0.5014 | 0.0142 | 5099.0 | 6.11 | 22.81 | 2.023 |
| PF4 | HU | 0.4883 | 0.0136 | 4517.9 | 5.47 | 22.64 | 2.011 |

These are eligibility/cost diagnostics, not EV rankings. They do not by themselves select a winner.

## Immediate engineering interpretation

PF0 is the cheapest tree in both domains. PF4 is the cheapest richer abstraction in 3H and cheaper than PF1/PF2/PF3 in both domains. PF2 is by far the most expensive, especially HU, without an obvious learning-fit advantage. Nevertheless none of PF1–PF4 may be discarded solely from fit/cost because an extra sizing can still create material EV.

The correct next experiment is therefore a cheap direct strategic screen, not more dense-referee training.

## Lean strategic screen

1. Preserve all 30 eligible checkpoints/reports before Actions expiry.
2. Use exact action identity and common deals/common random numbers.
3. Run direct candidate-vs-candidate or common-baseline crossplay in HU and 3H at a small first tranche.
4. Eliminate a candidate early only when its EV deficit is large enough that plausible sampling uncertainty cannot make it preferred, while also considering compute cost.
5. Escalate hands only among close survivors.
6. Run exact-action omission diagnostics only where crossplay leaves a plausible strategic mechanism unresolved.
7. Resume dense 3H referee recovery only if it is decision-relevant to an unresolved close contest.

## Quality safeguard

Fit statistics and runtime cost are not substitutes for strategic EV. The lean policy removes proof-for-proof's-sake, not the direct evidence required to avoid selecting a weaker poker strategy.
