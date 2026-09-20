# SpinCore Roadmap — active state 2026-09-19

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Jammer FAI infoset overfold — **CONFIRMED**.
- Action-gap drift — **DOMINANT CAUSAL COMPONENT**.
- Stage-B reservoir poisoning — **NOT SUPPORTED**.
- 100-step HU Advantage refit — **INSUFFICIENT**.
- 400-step HU Advantage refit — **STRUCTURAL PASS**.
- B400 broad HU generalization — **PASS**.
- Intervention — **FROZEN AS HU 400 / 3H 100 / K4 OFF**.
- 100-iteration online continuation pilot — **NEXT**.
- long continuation beyond 7600 — **NOT AUTHORIZED**.
- holdout — **SEALED**.
- DeepCrusher — **DEFERRED**.

## Broad B400 evidence

JAMMER, candidate minus production Stage B:

- B400_R0: `+16.8266`, CI95 `[+12.8183,+20.8348]`;
- B400_R1: `+10.8855`, CI95 `[+7.1343,+14.6368]`;
- B400_R2: `+10.5116`, CI95 `[+6.4837,+14.5395]`.

PASSIVE_CALLER:
- all three B400 candidates improve significantly.

UNIFORM_LEGAL:
- no B400 candidate shows resolved deterioration;
- R1 improves significantly.

The repair is therefore population-level rather than a selected-anchor artifact.

## Why HU-only

The causal chain and broad validation are HU-specific.

Raising the 3H fit budget would be an untested extra intervention and would approximately increase compute for a domain that has not been implicated.

The minimal frozen change is:

- 3H = 100 Advantage fit steps;
- HU = 400 Advantage fit steps.

## Next gate — online feedback

Create an isolated copy of Stage B and continue exactly 100 iterations:

- 7501..7600;
- 60k roots;
- 31 root workers;
- vectorized fitting;
- concurrent domain fit;
- 3H 100;
- HU 400.

Then automatically compare Stage B vs pilot on the full forensic HU population using both:

- deployed AveragePolicy;
- current Advantage behavior.

### Pass direction

Current behavior:
- resolved improvement vs Stage B against JAMMER;
- no resolved material deterioration against PASSIVE_CALLER or UNIFORM_LEGAL.

AveragePolicy:
- observe direction, but do not require full repair after only 100 new iterations because most policy-memory samples remain historical.

### If current behavior passes but AveragePolicy lags

Extend the same frozen intervention in another bounded block to refresh policy memory.

### If current behavior fails

Stop before more roots and inspect online target/reservoir feedback.

## Immediate action

Run `bash tools/run_lt2_hu_b400_online_pilot.sh`.

Do not go beyond iteration 7600 until that result is reviewed.
