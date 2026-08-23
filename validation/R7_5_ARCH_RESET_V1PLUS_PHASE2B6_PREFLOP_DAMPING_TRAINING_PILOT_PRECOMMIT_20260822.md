# R7.5 Architecture Reset — V1+ Phase2B6 Small Preflop-Damping Training Pilot

Status: **FROZEN BEFORE PHASE2B6 TRAINING OUTPUTS**  
Date: 2026-08-22

## 1. Purpose

Phase2B5 produced a precommitted `MILD_PREFLOP_DAMPING_CANDIDATE`. With the root baseline and all postflop continuation controlled, a seed-independent 25% uniform floor applied only to preflop continuation reduced pooled target-policy TV from `0.32010786853721923` to `0.21666478360495514` (absolute `0.10344308493226409`, relative `32.315%`), improved 14/15 scenarios, and reduced rather than increased dominant-action mismatch.

The oracle depth decomposition also localized most preflop feedback to the earliest continuation decisions: `DELTA1=0.1469488759211661`, `DELTA2=0.08650202295893697`, `DELTA3=0.028026058553241584`, with little additional positive contribution at greater depth.

Phase2B6 is therefore one small **causal end-to-end training pilot** of exactly the selected mild intervention. It asks whether the read-only stabilization effect survives when the regularizer is present during learning and is then removed at heldout inference.

This pilot does not reopen R7.5.3, does not authorize production, does not select an architecture, and cannot establish strategic strength by itself.

## 2. Frozen identity

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED`.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Training seeds: `1342191342`, `1801739323`.
- Heldout evaluation seeds: `2029384436`, `1150634112`.
- Iterations: `3`.
- Chance coverage: `4 x 64 = 256` roots/iteration, `768` roots/seed.
- Exact opponent levels: `2`.
- Advantage reservoir: `100000`.
- Strategy reservoir: `100000`.
- Advantage steps/member/iteration: `4096`.
- Advantage ensemble members: `4`.
- AveragePolicy steps: `16384`.
- Batch size: `256`.
- Learning rate: `0.001`.
- Heldout policy states/evaluation seed: first `1024` frozen states.
- Phase2A source execution SHA: `4bfa55d69029cd69536fa6dbfcadd162719cb887`.
- Phase2A result SHA-256: `65f691e6b9cf7fbbddf88852c5ac6e0dcd2211af45f53cc4bb3e8271dbaa6149`.
- Phase2B5 result SHA-256: `0fb028c02dbbea0c4fa7a323a3edeed5c4e12789145235be2e851452e16ab5b8`.

The exact completed Phase2A `S100K_CONTROL` artifacts are the frozen no-intervention control. **Do not retrain a native control.** Reusing the already completed matched control prevents unnecessary compute and avoids introducing a new control trajectory.

## 3. Intervention

The only causal change is the training behavior policy on a preflop policy call after at least one non-forced preflop public event has already occurred in the hand:

`p_damped = 0.75 * p_native + 0.25 * U(legal)`

where `p_native` is the existing uncertainty-damped H2 Advantage ensemble policy and `U(legal)` is uniform over the exact legal universal actions.

Frozen scope:

- no added floor at the initial root decision (`nonforced_preflop_count == 0`);
- 25% floor on preflop continuation calls (`street == PREFLOP` and `nonforced_preflop_count >= 1`);
- no added floor on flop, turn, or river;
- no changes to action abstraction, chance schedule, exact-opponent depth, network architecture, optimizer, loss, reservoirs, model-reset seeds, side-member seeds, or heldout states.

The SPNNIV3 parser must verify exact wire length `120 + 20 * history_count` before counting non-forced preflop events. Every frozen 3H scenario root must be verified to start on preflop with zero non-forced preflop events.

Iteration 1 starts with no learned Advantage ensemble, so the native behavior is uniform and the 25% floor is algebraically neutral. Any trajectory effect can enter only after the first Advantage fit.

## 4. Training and resume semantics

Use the exact Phase2A x4 root/deck schedule:

- 4 chunks of 64 roots per iteration;
- scenario index = `global_root % 15`;
- authoritative frozen `deck_seed(training_seed, global_root, iteration)`;
- Advantage fit only after all four chunks of an iteration;
- checkpoint after every chunk;
- checkpoint must preserve both reservoirs, batch RNG, completed chunk/iteration state, behavior ensemble state, and intervention telemetry;
- resume may continue only from an exact Phase2B6 checkpoint identity.

No Phase2A artifact may be modified.

## 5. Final AveragePolicy readouts

Fit the Phase2B6 Strategy memory twice per training seed exactly as Phase2A did:

1. `COMMON_LEARNER`: fixed common initialization and batch RNG across both training seeds. This is the **primary causal readout**, because it removes final-policy learner RNG as an explanation for cross-seed differences.
2. `NATIVE_LEARNER`: original seed-coupled final-policy initialization/batch RNG. This is corroborative only.

The final heldout evaluation uses the learned AveragePolicy directly. **No 25% floor is applied at inference/evaluation.** This prevents the intervention from hiding instability by flattening the evaluated policy after training.

## 6. Frozen local validity gates

A Phase2B6 result is invalid unless:

- every required Advantage iteration for both training seeds has ensemble weighted NRMSE `<= 0.75`;
- every `COMMON_LEARNER` final policy fit has weighted mean TV `<= 0.12`;
- both training seeds complete exactly `768` roots and 3 iterations;
- the two frozen heldout evaluation seeds are used without regeneration;
- the canonical variable-length legal-set -> ten-slot legal-mask conversion is used before SPNNIV3 collation.

`NATIVE_LEARNER` policy-fit gates are reported and expected to pass; a failure is a diagnostic contradiction and blocks progression.

## 7. Primary causal comparison

For each heldout evaluation seed, compute per-state cross-seed TV for:

- frozen Phase2A `COMMON_LEARNER__S100K_CONTROL`; and
- Phase2B6 `COMMON_LEARNER`.

Use paired state differences `baseline_TV - pilot_TV` and an equal-group stratified bootstrap over the two heldout evaluation seeds, `2000` replicates, 95% confidence interval.

The 25% training intervention is **causally supported** only if all of these precommitted conditions hold:

1. all local validity gates pass;
2. pooled COMMON mean-TV improvement is at least `0.02` absolute **or** at least `10%` relative;
3. bootstrap 95% CI lower bound for pooled COMMON improvement is strictly `> 0`;
4. both heldout evaluation-seed COMMON mean TVs improve (no seed may rely on compensation by the other);
5. no heldout evaluation-seed COMMON p95 TV degrades by more than `0.02` absolute;
6. NATIVE pooled mean TV does not worsen versus its exact Phase2A `S100K_CONTROL` baseline, and no NATIVE heldout mean degrades by more than `0.01`.

These are causal-effect gates, not production gates.

## 8. Stability classification

The historical cross-seed thresholds are retained as reference hard stability gates:

- mean TV `<= 0.15` on each heldout evaluation seed;
- p95 TV `<= 0.35` on each heldout evaluation seed.

Decision hierarchy:

- local validity failure -> `PHASE2B6_INVALID_LOCAL_GATES`;
- causal-effect conditions fail -> `PREFLOP_DAMPING_TRAINING_EFFECT_NOT_SUPPORTED`;
- causal effect passes but one or more hard stability gates fail -> `PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE`;
- causal effect passes and both heldout seeds satisfy both hard stability thresholds -> `PREFLOP_DAMPING_PILOT_STABILITY_PASS`.

A failed 25% pilot does **not** authorize opportunistically training the 50%, 75%, or 100% read-only B5 floors. Those were strategically intrusive diagnostic controls. A failure routes back to a precommitted anchor/lagged-target or other mechanism-specific design.

## 9. Strategic-strength firewall

Even `PREFLOP_DAMPING_PILOT_STABILITY_PASS` is only a stability result. The 25% uniform floor deliberately reduces policy sharpness during training and could reduce poker strength.

Therefore after a stability PASS, a separate precommitted strategic-strength comparison against the stable V1/R7.4 control is mandatory before architecture selection or production. Stability alone can never make this candidate the winner.

## 10. Governance

- R7.5.3 old H2/H3 admission remains `FAIL/BLOCKED/CLOSED`.
- Phase2B6 is architecture-reset causal research only.
- No H4 selection.
- No production training authorization.
- No table deployment.
- `READY FOR TABLES = NO`.
