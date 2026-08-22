# R7.5 architecture reset — V1+ Phase 2B0 policy-algebra screen precommit

Date: 2026-08-22
Status: FROZEN_BEFORE_PHASE2B0_OUTPUT
READY FOR TABLES: NO
Production training authorized: NO

## Governance scope

This is a post-R7.5.3 architecture-reset read-only screen. R7.5.3 remains `FAIL_BLOCKED_CLOSED`; this step is not another H2/H3 readmission attempt and cannot select a representation or authorize production training.

The preceding Advantage forensic found severe target/sign disagreement on shared exact infosets and severe final Advantage behavior divergence on common heldout states. Before spending CPU on another full training trajectory, this screen tests whether part of that instability is being amplified by the order of operations in the already-frozen four-member Advantage ensemble.

## Mechanism under test

The current accepted behavior algebra is:

1. convert each ensemble member's raw Advantage vector independently through regret matching;
2. average the member policies;
3. measure member-policy disagreement around that average;
4. set `epsilon = min(0.5, 1.75 * disagreement)`;
5. mix the mean policy toward uniform legal play by epsilon.

Regret matching clips negative Advantage estimates to zero before normalization. Around zero, small noisy sign changes can therefore cause very large policy changes before ensemble averaging.

The only candidate screened here changes the exploitation aggregation order:

`RAW_MEAN_THEN_REGRET_MATCH`

1. average the four raw Advantage vectors slot-by-slot;
2. apply regret matching once to that raw mean;
3. reuse the **exact same epsilon value** computed by the frozen control algebra on the same four member outputs;
4. mix that candidate exploitation policy toward the same uniform legal policy by the same epsilon.

Thus this screen changes only whether ensemble averaging occurs before or after the discontinuous positive-regret transform. It does not change any model weight, training sample, seed, uncertainty scale/cap, legal action set or heldout state.

## Frozen inputs

Use exactly the two completed Phase 2A H2/THREE_HANDED resume checkpoints from source execution:

`4bfa55d69029cd69536fa6dbfcadd162719cb887`

Training seeds:

- `1342191342`
- `1801739323`

Each checkpoint must be completed at:

- representation `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`;
- domain `THREE_HANDED`;
- 3 iterations;
- 768 roots;
- Phase2A stage index 12;
- four final behavior/Advantage ensemble model states present in checkpoint extra data;
- no production/table authorization.

Frozen heldout corpora:

- THREE_HANDED / `2029384436`;
- THREE_HANDED / `1150634112`;
- first 1024 states from each, matching prior stability diagnostics.

## Read-only measurements

For every heldout state and each training seed:

- evaluate all four frozen Advantage ensemble members once;
- compute the exact frozen control behavior policy and its epsilon/disagreement diagnostics;
- compute `RAW_MEAN_THEN_REGRET_MATCH` using the same epsilon;
- record control-vs-candidate within-seed TV;
- record candidate raw-mean positive-regret count and dominant action.

Across the two training seeds, for CONTROL and CANDIDATE separately record:

- mean/p50/p95/max behavior-policy TV;
- dominant legal action mismatch rate;
- summaries by street;
- pooled summaries across both evaluation seeds.

Also record the control epsilon distribution and verify candidate and control use bit-identical epsilon per state/seed.

## Predeclared screening rule

`RAW_MEAN_THEN_REGRET_MATCH` is eligible for a later causal Phase 2B training ablation only if all of the following hold on this read-only screen:

1. candidate cross-seed mean behavior TV improves on **both** frozen evaluation seeds;
2. pooled candidate mean TV improves by at least `0.05` absolute **or** at least `10%` relative versus control;
3. no evaluation seed's p95 TV worsens by more than `0.02` absolute;
4. pooled dominant-action mismatch rate does not increase;
5. exact epsilon identity between control and candidate is verified for every evaluated state/seed.

Passing this screen does not prove causal training improvement. It only justifies spending CPU on one controlled Phase 2B training comparison with the candidate behavior algebra versus the current control.

If the candidate fails the screen, do not train it. The next engineering target becomes direct chance/return variance reduction or stratified chance support rather than further policy-algebra tuning.

## Strategic-strength safeguard

Even a later candidate that improves cross-seed stability cannot be selected for production on stability alone. It must eventually demonstrate strategic non-inferiority or superiority to the certified stable V1 control on precommitted paired common-deal evaluation.

## Prohibitions

- no solver traversal in this screen;
- no checkpoint mutation;
- no optimizer step;
- no model fit;
- no reservoir replay;
- no change to epsilon scale/cap;
- no additional candidate algebra chosen after seeing the result;
- no seed shopping;
- no threshold relaxation;
- no R7.5.3 reopening;
- no architecture winner declaration;
- no production authorization.
