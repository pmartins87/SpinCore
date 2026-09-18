# SpinCore Roadmap — active state 2026-09-18

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Weak-baseline regression — **CONFIRMED**.
- AveragePolicy extra-fit — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED**.
- HU-preflop target variance — **HIDDEN/CHANCE DOMINANT**.
- K4 board averaging — **VALID ESTIMATOR IMPROVEMENT**.
- Jammer K4 causal gate — **NOT MET**.
- Cross-street future-chance audit — **PASS; FUTURE-CHANCE NOT FAILURE-SPECIFIC**.
- Stage-A/B target-drift matrix — **PASS; NO UNIVERSAL TARGET-DRIFT EXPLANATION**.
- HU AveragePolicy-vs-current-behavior chain — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Target-drift verdict

Jammer:
- target A->B is invariant after opponent all-in;
- Stage-B own-target Advantage MSE does not worsen;
- deployed AveragePolicy still regresses.

This rules out target nonstationarity as the Jammer explanation and makes the policy aggregation/deployment chain the next target.

Passive flop:
- target drift is larger in controls;
- failures show a resolved Stage-B own-target fit degradation internally;
- but FAILURE-vs-CONTROL excess is unresolved;
- current Advantage action ranking is poor in both samples.

Uniform turn:
- target drift and tracking are heterogeneous;
- no failure-specific mechanism resolves.

## Next gate

Run `tools/run_lt2_hu_policy_chain.sh`.

Global HU, same forensic seeds, common random numbers.

Compare:
1. AVG_A;
2. AVG_B;
3. BEH_A;
4. BEH_B;

against each transparent weak baseline.

## Decision after policy-chain audit

- AVG B-A negative, BEH B-A neutral/positive:
  AveragePolicy aggregation/history is implicated.

- both AVG and BEH B-A negative:
  regression is already upstream in current Advantage behavior.

- baseline-dependent split:
  multiple mechanisms.

No training before this distinction is measured.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_hu_policy_chain.sh`.

Stop at PASS or first error. Do not train.
