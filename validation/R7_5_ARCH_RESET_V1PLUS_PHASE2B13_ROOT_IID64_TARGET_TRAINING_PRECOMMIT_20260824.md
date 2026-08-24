# R7.5 Architecture Reset — Phase2B13 Root IID64 Target Training Pilot

Status: **FROZEN BEFORE PHASE2B13 OUTPUTS**  
Date: 2026-08-24

## 1. Purpose

Phase2B12 showed that ordinary conditional IID chance integration at the **initial preflop root** converges materially and monotonically: pooled diagnostic root-target policy TV fell from K16 `0.33467186760867673` to K64 `0.17378003404961345`, with sign disagreement and the `TV >= 0.35` tail also falling materially for both source behavior seeds.

Phase2B13 is the smallest causal training experiment that uses exactly that supported scope. It asks:

> If the ordinary one-sample root Advantage target is replaced during training by a K64 conditional-IID estimate, while every downstream Advantage sample, every Strategy sample, the 25% preflop-continuation behavior floor, the learner, architecture and heldout inference remain unchanged, does final cross-seed AveragePolicy stability improve versus an equal-compute control?

This phase deliberately does **not** apply chance averaging to arbitrary post-history infosets. Phase2B12 did not establish a valid conditional resampling law there.

## 2. Frozen prerequisites

- representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`;
- domain: `THREE_HANDED`;
- action candidate: `PF0_CONTROL_33_75_AI`;
- training seeds: `1342191342`, `1801739323`;
- heldout evaluation seeds: the frozen two V3 heldouts;
- exact opponent levels: `2`;
- continuation behavior intervention retained in both arms: 25% uniform floor after at least one non-forced preflop event;
- root behavior: native;
- postflop behavior: native;
- heldout inference floor: `0.00`;
- Phase2B6 result SHA-256: `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`;
- Phase2B12 result SHA-256: `dbccadae5805381d0188bef41fb62a72b25b42e03e5564ca88f05d9666e6e182`.

The Phase2B12 result must have status `IID_CHANCE_EXPECTATION_CONVERGES_MATERIALLY`, `screen_pass=true`, and `small_causal_training_pilot_precommit_allowed=true`.

## 3. Pilot budget

This is a deliberately smaller causal screen, not a full x4 confirmation.

Per arm and training seed:

- 3 Deep-CFR iterations;
- 2 chunks per iteration;
- 64 logical roots per chunk;
- 128 logical roots per iteration;
- 384 logical roots per seed;
- ordinary Strategy collection and ordinary downstream Advantage collection once per logical root;
- one K64 conditional-IID root-target auxiliary estimate per logical root.

Across both arms and both seeds:

- logical roots: `2 arms × 2 seeds × 384 = 1536`;
- auxiliary root-target traversals: `1536 × 64 = 98,304`.

The two arms must have the same logical-root schedule and the same 64 explicit deals and traversal RNG namespace for each `(training_seed, iteration, global_root)`.

## 4. Arms

### `IID1_OF_64_EQUAL_COMPUTE_CONTROL`

For each logical root:

1. construct the ordinary scheduled anchor using the frozen `deck_seed(training_seed, global_root, iteration)`;
2. identify the initial root actor and exact SPNNIV3 root observation;
3. generate 64 legal conditional-IID deals holding that actor's exact two hole cards fixed while sampling opponent private cards and future board from the legal remaining deck;
4. evaluate all 64 root targets with one fixed traversal RNG seed for that logical root;
5. use **only sample 0** as the replacement initial-root Advantage target.

The other 63 targets are deliberately computed and discarded. This makes compute, chance support generation and worker scheduling equal to the candidate while preserving a single-sample root-target estimator.

### `IID64_MEAN_CANDIDATE`

The same 64 legal deals and the same traversal RNG are evaluated. The ten-slot raw Advantage targets are arithmetic-averaged before insertion, exactly matching the Phase2B12 K64 estimator at the root.

## 5. Exact intervention boundary

The ordinary training root collection still runs once for every logical root for **all traversers and Strategy collection**.

A deterministic memory proxy suppresses exactly one sample: the Advantage sample whose observation bytes equal the initial root SPNNIV3 observation and whose iteration equals the current iteration. The collector must attempt to add exactly one such sample per logical root. Zero or multiple suppressions abort the run.

After ordinary collection:

- control inserts one replacement root sample using IID sample 0;
- candidate inserts one replacement root sample using the mean of all 64 IID targets;
- legal mask, weight and iteration must match the suppressed root sample contract (`weight == iteration` at the initial root).

Everything else from ordinary collection is retained byte-for-byte in mechanism:

- downstream Advantage samples are **not** replaced or averaged;
- Strategy samples are unchanged in mechanism;
- reservoir capacity and reservoir replacement semantics are unchanged;
- Advantage reset seeds, side-member seeds, batch RNG rules, optimizer, loss and number of fit steps are unchanged;
- AveragePolicy fitting uses the same COMMON and NATIVE learner protocols as Phase2B6;
- heldout inference uses the learned AveragePolicy directly with no floor or target averaging.

## 6. Conditional-IID generation

The acting player's two root hole cards are fixed. Opponent private cards and all five future board cards are sampled legally from the remaining deck using deterministic, precommitted seed namespaces independent of arm.

Every explicit variant must preserve:

- exact root SPNNIV3 bytes;
- root actor;
- universal legal set and ten-slot legal mask.

Any mismatch aborts the task.

The 64 variants use the same fixed traversal RNG within a logical root so the auxiliary estimator isolates hidden/future chance rather than traversal-action RNG.

## 7. Equal-compute and pairing guarantees

For a fixed training seed and logical root, the two arms use identical:

- anchor deck seed;
- conditional-IID private/public chance seeds;
- number of auxiliary root traversals (`64`);
- traversal RNG seed;
- normal logical-root collection budget;
- optimizer steps and policy-fit steps.

Only the statistic inserted as the **single replacement root Advantage target** differs: first IID target versus mean of 64 IID targets.

After the arms diverge, their learned behavior policies are naturally allowed to differ. Chance seeds remain paired; target values may differ because the source behavior has causally diverged.

## 8. Local validity gates

For each arm and seed:

- exactly 384 logical roots;
- 3 completed iterations;
- exactly one suppressed/replaced root sample per logical root;
- exactly 64 auxiliary target traversals per logical root;
- all three iteration Advantage ensemble NRMSE gates pass;
- COMMON and NATIVE AveragePolicy fit gates pass;
- no solver/target worker error or root-identity drift.

If either arm fails local validity, Phase2B13 is invalid and no causal interpretation is accepted.

## 9. Primary causal stability gate

Primary learner: `COMMON_LEARNER`.

Compare candidate minus equal-compute control on the exact frozen heldout descriptors.

The candidate is causally supported only if **all** are true:

1. pooled COMMON mean-TV improvement is `>= 0.02` absolute **or** `>= 10%` relative;
2. equal-group stratified bootstrap 95% CI of `(control TV - candidate TV)` has `ci_low > 0`;
3. candidate mean TV is lower on both heldout evaluation seeds;
4. candidate p95 TV on either heldout does not exceed control by more than `0.02`;
5. NATIVE learner pooled mean TV is non-worse and no individual NATIVE heldout mean degrades by more than `0.01`.

The historical hard stability gates remain separate:

- each COMMON heldout mean TV `<= 0.15`;
- each COMMON heldout p95 TV `<= 0.35`.

Because Phase2B13 uses only 384 logical roots per seed, even a hard-gate PASS does not select an architecture. It requires a full-budget confirmation before strategic-strength testing.

## 10. Frozen classification

- invalid local gate or identity failure -> `PHASE2B13_INVALID_STOP_AUDIT`;
- causal gate fails -> `ROOT_IID64_TRAINING_EFFECT_NOT_SUPPORTED` and route `REASSESS_CONTINUATION_CONDITIONAL_CHANCE_OR_REPRESENTATION_SUPPORT_NO_SCALEUP`;
- causal gate passes but historical hard stability does not -> `ROOT_IID64_CAUSAL_EFFECT_SUPPORTED_SMALL_PILOT` and route `PRECOMMIT_FULL_X4_ROOT_IID64_CONFIRMATION`;
- causal gate and historical hard stability both pass -> `ROOT_IID64_SMALL_PILOT_HARD_STABILITY_PASS` and route `PRECOMMIT_FULL_X4_ROOT_IID64_CONFIRMATION_BEFORE_STRENGTH`.

A PASS authorizes only the stated full-budget confirmation precommit. It does not authorize production training or strategic-strength selection.

## 11. Guardrails

- no K sweep after B12;
- no K128/K256 in this pilot;
- no factorized estimator retry;
- no averaging of arbitrary post-history infosets;
- no higher preflop continuation floor;
- no Huber or lag-anchor tuning;
- no gate relaxation;
- no seed shopping or dropped scenario;
- no production training;
- no architecture winner selection;
- `READY FOR TABLES = NO`.

## 12. Strategic firewall

Stability and strategic strength remain distinct. Even a later full-budget hard stability PASS must be followed by a separately precommitted strength comparison against the certified stable V1 control before any representation/architecture can be selected.
