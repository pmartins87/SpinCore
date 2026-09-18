# SpinCore — LT2 Stage-A -> Stage-B first-divergence forensic

Date: 2026-09-18
Status: **ACTIVE — DIRECT DEPLOYED-POLICY ATTRIBUTION BEFORE ANY K4 TRAINING**

## Why this gate exists

The corrected K4 mechanics smoke passed. That proves the implementation changes HU-preflop Advantage labels as intended without changing sample identity or canonical postflop labels.

It does **not** prove that future-board target noise caused the Stage-A -> Stage-B strength regression.

The confirmed practical regression was measured on the stored **AveragePolicy** against transparent weak opponents. Therefore the next causal question must begin at that deployed-policy layer:

**Where does Stage B first behave differently from Stage A on the same evaluation trajectories, and which first-divergence contexts actually contribute the negative B-minus-A chip EV?**

This prevents a diagnostic estimator improvement from being promoted into a training intervention merely because it looks promising against a benchmark observed after the fact.

## Checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Both are read only and hash-checked.

## Diagnostic seed discipline

The audit deliberately reuses the already-seen powered weak-baseline diagnostic seeds:

- 20260920
- 20260921
- 20260922
- 20260923
- 20260924
- 20260925

Default: 5,000 full-sampler scenarios per seed.

Only HU scenarios are retained for this localization because the confirmed Stage-B failure is HU Jammer.

These seeds are **forensic/design data**. They are no longer eligible as acceptance evidence for a future intervention.

Reserve a fresh untouched seed family for any future candidate acceptance:
- 20261001
- 20261002
- 20261003
- 20261004
- 20261005
- 20261006

Do not use the reserved family in this forensic design.

## Opponent families

All three transparent weak baselines are included:

- UNIFORM_LEGAL
- PASSIVE_CALLER
- JAMMER

This is intentionally broader than the Jammer failure alone.

## Pairing

For every retained HU scenario, weak baseline, and hero seat:

- Stage A and Stage B receive the same Episode;
- same solver deal seed;
- same weak opponent;
- same hero seat;
- same hero random stream;
- same opponent random streams.

The two solver states remain identical until the sampled hero actions from Stage A and Stage B first differ.

After first divergence each arm is played independently to terminal.

## Mutually exclusive first-divergence partition

Every seat-run belongs to exactly one group:

- NO_DIVERGENCE
- PREFLOP_ROOT
- PREFLOP_FACING_ALL_IN
- PREFLOP_OTHER
- FLOP
- TURN
- RIVER

For every baseline, each group's B-minus-A chip delta is converted into an additive contribution to total paired B-minus-A chip EV.

The group contributions must sum back to the overall result to numerical precision.

## Additional diagnostics at first divergence

For diverged runs report:

- Stage-A action slot -> Stage-B action slot;
- AveragePolicy TV on the identical divergence state;
- Stage-B minus Stage-A probability-mass shift for:
  - FOLD;
  - CHECK_CALL;
  - ALL_IN;
- group frequency;
- conditional B-minus-A terminal chip delta;
- additive contribution to overall B-minus-A chip EV.

## Interpretation

This audit localizes the **deployed-policy regression**.

It does not yet prove the Advantage-target cause.

Decision branches:

- If the negative HU-Jammer contribution is mainly postflop, HU-preflop K4 is not a plausible primary fix and must not be trained as the main intervention.
- If the negative contribution is preflop but not specifically tied to the same states affected by board averaging, K4 remains unproven.
- If a resolved negative contribution is concentrated in HU-preflop/FACING_ALL_IN and Stage B shifts mass in the same problematic direction previously seen in the lower-variance target diagnostics, then K4 remains a causal candidate — but one more target/Advantage overlay is still required before training.
- Cross-baseline consistency strengthens a general mechanism interpretation; Jammer-only localization is treated as benchmark-specific evidence and is not enough by itself to authorize training.

## Launcher

```bash
bash tools/run_lt2_stage_a_b_first_divergence.sh
```

Expected marker:

`LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_PASS`

No K4 training is authorized by this gate alone.
