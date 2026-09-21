# SpinCore — LT2 HU ENS8 Final Holdout Protocol

Date frozen: 2026-09-21  
Candidate: **TRUE_HEADS_UP current ENS8 @ iteration 8100**

## Candidate identity

The candidate is not AveragePolicy.

It is the current HU behavior represented by the pair:

- iteration-8100 ordinary checkpoint;
- iteration-8100 `hu_ensemble_state.pt` sidecar.

Semantics:

- 8 fresh-400 Advantage members;
- predeclared ENS8_A member seed contract;
- raw Advantage mean;
- unchanged lean regret matching;
- K4 off.

No further training is permitted before holdout.

## Holdout seeds

`20261001, 20261002, 20261003, 20261004, 20261005, 20261006`

These are unsealed only after this protocol and its decision criteria are committed.

## Evaluation families

### Learned historical ecosystem

Opponent is deterministically assigned from:

- AVG_7600;
- AVG_8000;
- BEH_7600;
- ENS8_8000.

Hero policies compared with common random numbers:

- ENS8_8100;
- ENS8_8000;
- AVG_8100.

### Direct seat-balanced

- ENS8_8100 vs ENS8_8000;
- ENS8_8100 vs AVG_8100.

### Transparent weak baselines

- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Compare ENS8_8100 and ENS8_8000 using identical scenarios/deals/seats/RNG.

## Sample plan

- 6 holdout seeds;
- 6000 sampler scenarios per seed;
- all realized HU scenarios;
- paired scenario-cluster confidence intervals.

## Primary PASS criteria

All must pass:

1. ecosystem absolute ENS8_8100: lower 95% CI > 0;
2. ecosystem ENS8_8100 − ENS8_8000: lower 95% CI > -3.0 chips;
3. ecosystem ENS8_8100 − AVG_8100: lower 95% CI > 0;
4. direct ENS8_8100 vs ENS8_8000: lower 95% CI > -3.0 chips;
5. direct ENS8_8100 vs AVG_8100: lower 95% CI > 0;
6. ENS8_8100 absolute EV vs each weak baseline: lower 95% CI > 0;
7. ENS8_8100 − ENS8_8000 vs each weak baseline: lower 95% CI > -3.0 chips.

No criterion may be changed after outcomes are observed.

## Failure discipline

If any primary criterion fails:

- report FAIL;
- preserve the result;
- do not tune on these holdout seeds;
- do not rerun with altered members, margins or sample selection;
- any revised candidate requires a new, untouched validation family.

## Interpretation

PASS validates the frozen HU candidate against this predefined battery. It is still not a mathematical proof of GTO optimality or exploitability.
