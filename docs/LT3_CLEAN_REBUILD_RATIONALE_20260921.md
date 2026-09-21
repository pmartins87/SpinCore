# LT3 clean rebuild from zero — rationale

Date: 2026-09-21

## Decision

The next strategic training line will start from iteration 0 rather than resume
LT2@8100 or the interrupted LT3 continuation.

LT2@8100 remains frozen and valid as the comparison baseline.

## Why continuation was originally reasonable

The mature Stage-B diagnosis did **not** support the hypothesis that its HU
reservoir was poisoned.  Fresh refits on the exact Stage-B reservoir showed that
the signal was present but the canonical 100-step HU fit was too weak to extract
it reliably.

That justified the minimal intervention used at the time:

- keep the accumulated reservoir/history;
- raise HU fitting from 100 to 400 steps;
- verify the intervention online;
- later stabilize mature HU behavior with ENS8.

That path produced the holdout-passed LT2@8100 candidate.

## Why a clean rebuild is now preferred

The training lineage is not homogeneous:

- iterations 1..7500: 3H fresh100, HU fresh100;
- iterations 7501..8000: 3H fresh100, HU fresh400, single current HU model;
- iterations 8001..8100: 3H fresh100, HU ENS8 fresh400 online behavior.

Thus the 8100 holdout proves that the **resulting candidate** passes the frozen
validation battery.  It does not prove that the same mixed schedule is the best
way to train the final system.

A clean rebuild answers the stronger engineering/scientific question: how the
current corrected training recipe behaves when it governs the learning
trajectory from the beginning.

## Important distinction about earlier 'errors'

Not every issue found during development contaminated the LT1/LT2 training
state.  Several were audit, validator, launcher, resource, or experimental
branch defects.  Core repaired action fallback and the lean functional training
path were in place before the serious LT1 campaign.

The material training-line change discovered after Stage B was the HU fitting
budget: fresh100 was insufficient at maturity.  ENS8 was later added to reduce
mature current-policy fit instability.

Therefore the old line is not discarded as invalid; it is preserved as a
valuable validated baseline.  The fresh line is about removing lineage
ambiguity and maximizing confidence.

## Before new roots

Do not launch a long fresh run yet.

First:

1. pass the Ryzen ENS8 exact-parity throughput matrix;
2. choose the fastest exact execution layout;
3. freeze whether ENS8 is active from iteration 1 or introduced at a
   preregistered maturity transition;
4. freeze milestone/checkpoint cadence and stop criteria;
5. only then launch iteration 0.

The LT2 final holdout is retired and must not be reused for these research
choices.  The LT3 sealed holdout remains untouched.
