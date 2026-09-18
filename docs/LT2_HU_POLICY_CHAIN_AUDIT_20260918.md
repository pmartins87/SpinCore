# SpinCore — LT2 HU policy-chain audit

Date: 2026-09-18
Status: **ACTIVE — TEST WHETHER STAGE-B REGRESSION IS IN CURRENT BEHAVIOR OR IN AVERAGE-POLICY AGGREGATION**

## Trigger

The Stage-A/B target-drift + model-tracking matrix showed:

- Jammer facing-all-in target drift is effectively zero;
- Jammer Stage-B own-target Advantage MSE does not worsen;
- yet deployed Stage-B AveragePolicy has a resolved negative Jammer B-A benchmark result.

Therefore target generation itself cannot explain the Jammer regression.

PassiveCaller flop shows some Stage-B own-target fit degradation, but it is not failure-specific versus controls.

UniformLegal turn remains mixed.

The highest-value next question is downstream and global:

**Does the current Stage-B Advantage-induced behavior also regress, or does the regression appear mainly after AveragePolicy aggregation?**

## Design

Use only the already-seen forensic seed family:

`20260920..20260925`.

For each seed:
- 5000 scenarios from the same empirical sampler;
- keep HU scenarios only;
- same scenario;
- same deal;
- same hero seat;
- same random streams.

Evaluate four hero policies:

1. `AVG_A` — Stage-A deployed AveragePolicy;
2. `AVG_B` — Stage-B deployed AveragePolicy;
3. `BEH_A` — Stage-A current Advantage-induced regret-matching behavior;
4. `BEH_B` — Stage-B current Advantage-induced regret-matching behavior.

Against:
- `UNIFORM_LEGAL`;
- `PASSIVE_CALLER`;
- `JAMMER`.

## Primary contrasts

For each baseline:

### Deployed-policy drift

`AVG_B - AVG_A`.

This should reproduce the known Stage-A/B weak-baseline direction on the same seed family.

### Current-behavior drift

`BEH_B - BEH_A`.

This tests whether the upstream current Advantage behavior itself worsened.

### Aggregation-chain delta

`(AVG_B - BEH_B) - (AVG_A - BEH_A)`.

This measures whether the AveragePolicy-vs-current-behavior gap became more adverse from Stage A to Stage B.

## Interpretation

If:

- `AVG_B-AVG_A < 0` but `BEH_B-BEH_A >= 0`:
  the historical AveragePolicy aggregation/deployment chain is implicated;

- both are negative:
  the regression is already present in current Advantage behavior;

- the answer differs by baseline:
  Stage-B deterioration is multi-mechanism.

This is still a weak-opponent mechanism diagnostic, not exploitability/GTO proof.

## Launcher

```bash
bash tools/run_lt2_hu_policy_chain.sh
```

Expected marker:

`LT2_HU_POLICY_CHAIN_EVAL_PASS`.

Holdout seeds `20261001..20261006` remain untouched.

No training is authorized before review.
