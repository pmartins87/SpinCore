# SpinCore — LT2 HU ENS8 final sealed holdout result

Date: 2026-09-21  
Status: **FINAL HOLDOUT PASS — HU ENS8@8100 VALIDATED UNDER FROZEN PROTOCOL**

## Candidate

TRUE_HEADS_UP current ENS8 at iteration 8100.

Required artifact pair:

- iteration-8100 ordinary checkpoint SHA256 `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`;
- iteration-8100 HU ensemble sidecar SHA256 `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`.

The holdout was executed from git commit:

`927ccaaa73cfe00edd42c6ad8279a38424146dd1`

## Integrity

Frozen holdout seeds:

- 20261001;
- 20261002;
- 20261003;
- 20261004;
- 20261005;
- 20261006.

Evaluation:

- 16,419 HU scenario clusters;
- 197,028 rows;
- no new training roots;
- no source-memory writes;
- no optimizer steps on source artifacts;
- no post-hoc tuning;
- preregistered non-inferiority margin: -3.0 chips.

## Verdict

**PASS — all 11 preregistered primary criteria passed.**

### Learned historical ecosystem

ENS8 8100 absolute:

- mean `+8.177`;
- CI95 `[+5.504,+10.850]`.

ENS8 8100 − ENS8 8000:

- mean `+1.223`;
- CI95 `[-0.488,+2.933]`;
- lower bound exceeds -3.0 margin.

ENS8 8100 − AVG8100:

- mean `+12.608`;
- CI95 `[+9.528,+15.689]`.

### Direct seat-balanced

ENS8 8100 vs ENS8 8000:

- mean `+0.510`;
- CI95 `[-1.231,+2.252]`;
- lower bound exceeds -3.0 margin.

ENS8 8100 vs AVG8100:

- mean `+7.195`;
- CI95 `[+4.342,+10.048]`.

### Transparent weak baselines

UNIFORM_LEGAL:

- ENS8 8100 absolute `+31.811`, CI95 `[+28.493,+35.130]`;
- 8100−8000 `+3.588`, CI95 `[+1.730,+5.447]`.

PASSIVE_CALLER:

- ENS8 8100 absolute `+10.288`, CI95 `[+8.063,+12.512]`;
- 8100−8000 `-0.124`, CI95 `[-1.476,+1.228]`;
- non-inferiority passes.

JAMMER:

- ENS8 8100 absolute `+13.996`, CI95 `[+11.455,+16.538]`;
- 8100−8000 `+0.377`, CI95 `[-1.062,+1.816]`;
- non-inferiority passes.

## Interpretation

The frozen ENS8 candidate passed every decision rule committed before the holdout was unsealed.

This closes the HU strategic validation phase for this candidate.

Do not rerun or tune against this holdout.

The next work is strictly deployment engineering:

- export a compact hybrid deployment artifact;
- use canonical AveragePolicy for THREE_HANDED;
- use frozen current ENS8 semantics for TRUE_HEADS_UP;
- prove exact inference parity mechanically.

No policy tuning, no new roots and no strategic reuse of the holdout are permitted during deployment integration.

This validation is finite empirical evidence, not a mathematical proof of GTO optimality.
