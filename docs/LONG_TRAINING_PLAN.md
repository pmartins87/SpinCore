# SpinCore — Long-Training Plan

Status: **8100 FROZEN — CURRENT ENS8 FORENSIC PASS — INDEPENDENT LEARNED-POLICY CROSSPLAY ACTIVE**
Date: 2026-09-21

## Post-pilot conclusion

The online ENS8 mechanism survived actual feedback:

- root all-in mass dropped about 16 pp;
- POT_33 rose about 16 pp;
- Uniform EV improved significantly;
- Passive/Jammer did not regress;
- ENS8 8100 strongly exceeds the old 7600 behavior on Passive and Jammer.

AveragePolicy remains statistically flat versus 8000. This is deployment lag rather than deterioration.

## Independent design-set gate

Before any more training or holdout, evaluate on newly preregistered seeds:

`20260926..20260930`

These are outside the final sealed holdout.

Historical learned-opponent ecosystem excludes the candidate 8100 and consists of:
- AVG7600;
- AVG8000;
- BEH7600;
- ENS8_8000.

Primary paired comparisons:
- ENS8_8100 − ENS8_8000;
- AVG_8100 − AVG_8000;
- ENS8_8100 − AVG_8100.

Seat-balanced direct versions are also required.

## Decision after crossplay

If ENS8 8100 has no resolved learned-policy regression and remains competitive with/stronger than AveragePolicy 8100, freeze the HU current ensemble as a deployment candidate and prepare the final holdout protocol.

If current ENS8 fails, localize before more roots.

If current ENS8 passes but AveragePolicy remains lagging, do not automatically spend more roots; candidate deployment semantics must be decided from crossplay evidence first.
