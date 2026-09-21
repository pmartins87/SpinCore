# SpinCore — Long-Training Plan

Status: **LT3 CLEAN REBUILD DESIGN / NO LONG TRAINING AUTHORIZED YET**
Date: 2026-09-21

## Preserved LT2 baseline

LT2 ENS8@8100 remains frozen and valid as the strongest sealed-holdout-passed
baseline.

Frozen identities:

- ordinary checkpoint SHA256:
  `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`;
- HU ENS8 sidecar SHA256:
  `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`.

The LT2 final holdout is retired and must not be reused for LT3 research
decisions.

## Why LT3 will start from iteration 0

The validated LT2 candidate has a mixed training lineage:

- iterations 1..7500: 3H fresh100 / HU fresh100;
- iterations 7501..8000: 3H fresh100 / HU fresh400 single-model behavior;
- iterations 8001..8100: 3H fresh100 / HU ENS8 fresh400 behavior.

The mature Stage-B forensic work showed that HU fresh100 was an insufficient fit
budget: the reservoir still contained useful signal, but 100 optimization steps
did not reliably extract it.  The reservoir-poisoning hypothesis was not
supported, which is why continuation was a valid minimal repair at the time.

That history does not invalidate LT2@8100: the frozen candidate passed its
pre-registered final holdout.  However, it does mean LT2@8100 is not a clean
end-to-end execution of the corrected training recipe.

The next research line therefore starts from iteration 0.

## Interrupted LT3 continuation

The sequential 8100->8600 experiment was interrupted after iteration 8240.
The last durable matched checkpoint+sidecar pair is iteration 8200.

This partial run is **historical evidence only** and must not be used as the
source of the clean LT3 rebuild.

## Mandatory gates before fresh training

No new long run is authorized until all of the following are frozen:

1. exact-parity Ryzen throughput layout for independent HU fits;
2. memory/disk-safe implementation of that layout;
3. from-zero algorithmic schedule, including whether ENS8 starts at iteration 1
   or at a preregistered maturity transition;
4. checkpoint cadence and disk headroom;
5. development-set milestone schedule and stop criteria.

The performance benchmark must measure end-to-end wall time including any data
movement/serialization overhead and must not create an unsafe multi-copy
reservoir memory footprint.

## Fresh-line invariants already fixed

- empirical SpinGo 3H/HU/blind/stack sampler;
- WTA chip-EV utility;
- SPNNIV1 frozen-control representation;
- repaired all-nonpositive regret fallback;
- separate 3H and HU domains;
- 2M reservoirs per memory/domain unless a separately validated change is made;
- 600 roots/iteration;
- 3H fresh100;
- HU minimum fit budget fresh400;
- K4 off;
- LT2 production artifacts read-only;
- LT2 final holdout never reused;
- LT3 sealed holdout untouched until all research choices are frozen.

## Immediate next gate

Benchmark and freeze the Ryzen execution strategy for HU ensemble fitting.

Do **not** resume iteration 8200 and do **not** launch a fresh long run until the
performance and from-zero schedule gates are closed.
