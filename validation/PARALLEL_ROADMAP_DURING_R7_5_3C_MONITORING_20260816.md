# SpinCore parallel roadmap during R7.5.3C stability monitoring — 2026-08-16

Status: **ACTIVE PARALLEL ENGINEERING PLAN**

Purpose: use the wall-clock time of the long R7.5.3C H2/H3 x4 chance-coverage readmission to finish every useful task that does **not** depend on the monitoring result, so that when the active stability run closes the remaining critical path contains only genuine result-dependent work.

This document does **not** add new canonical roadmap phases. It is a scheduling overlay over `ROADMAP.md` and is subordinate to `validation/R7_5_FINITE_CLOSURE_AND_COMPUTE_POLICY_20260816.md`.

## Non-negotiable boundaries

- The active H2/H3 x4 experiment remains unchanged: independent training seeds, unchanged `deck_seed`, 256 roots per iteration, unchanged Advantage/AveragePolicy budgets and unchanged hard gates.
- No H2/H3 winner may be selected from preparation work.
- No threshold may be relaxed.
- H4 is forbidden while H2/H3 admission remains unresolved.
- R7.5.4 strategic action/sizing execution is forbidden until a provisional representation winner exists.
- A material action-space change in R7.5.4 reopens the frozen H2/H3 representation comparison under the selected action abstraction.
- R8.3+ official training is forbidden until R7.5.5 and R8.0 both close their prerequisites.
- `READY FOR TABLES = NO`; only R12.9 may eventually change it.

## Parallel lanes

| ID | Work item | May execute now? | Completion condition | What remains result-dependent? |
|---|---|---:|---|---|
| P0.1 | Canonical dependency/frontier map | YES | This document exists and is linked from `ROADMAP.md` | Nothing |
| P0.2 | Governance staleness audit (`STATUS.json`, `VERSION.json`) | YES | Stale fields are explicitly inventoried; no false production/SPNNIV3 claim is introduced | Canonical mutation can follow after exact schema-preserving patch is prepared |
| P0.3 | Artifact persistence/certification contract for x4 stability | YES | Exact result bytes, SHA-256, GitHub run/artifact identity and gate result have a predeclared persistence format | Filling the values requires x4 completion |
| P0.4 | Post-x4 Phase-2 strategic evaluator readiness audit | YES | Existing frozen evaluator blobs/semantics and required inputs are inventoried; execution harness is prepared fail-closed | Actual strategic run requires x4 `STABILITY_PASS` |
| P0.5 | R7.5.4 SPNNIV3 action/sizing transfer readiness audit | YES | Historical reusable pieces vs mandatory SPNNIV3 reruns are bound to the existing transfer precommit | Running R7.5.4 requires provisional H2/H3 winner |
| P0.6 | Ryzen heavy-compute handoff | YES | Frozen local runner, manifest rules and exact-offload checklist are ready and smoke-tested | Concrete command/commit depends on the workload selected after current gate |
| P0.7 | R8.0 selected-state acquisition tooling audit | YES | Capture→packet→validation path and missing first-party facts are explicit | Exact selected-state data still has to be acquired from authoritative source/client evidence |
| P0.8 | R8.2 physical Ryzen calibration package preparation | YES, PREPARE ONLY | Calibration command/input/output contract can be staged without selecting architecture | Physical calibration must use final R7.5 architecture and therefore cannot run yet |
| P0.9 | R7.3 exact-reproducibility debt closure preparation | YES | Reproduction inputs, environment contract and acceptance rule are consolidated | Any new heavy rerun may be scheduled independently, but this debt only gates final release/R12 |
| P0.10 | Downstream R9–R12 prerequisite consistency audit | YES | Frozen designs are checked for references to obsolete V1/action assumptions and only mechanical/schema preparation is made | Strategic executions remain blocked by upstream gates |
| P1.1 | x4 stability result persistence | NO, READY-TO-FILL | Persist exact x4 result and evidence without reinterpretation | Requires current run result |
| P1.2 | Complete frozen Phase-2 strategic evaluation on admitted x4 policies | NO | Strategic metrics + hard gates + frozen selection rule produce winner or BLOCKED | Requires x4 `STABILITY_PASS` |
| P1.3 | R7.5.4 action/sizing strategic revalidation | NO | Frozen transfer protocol completes under provisional SPNNIV3 winner | Requires provisional representation winner |
| P1.4 | Representation recheck if action set changes | NO | H2/H3 comparison repeated under newly selected action abstraction | Only required if R7.5.4 changes PF0/PR0 materially |
| P1.5 | R7.5.5 production representation/action freeze | NO | Encoder + action abstraction + runtime identities become immutable production contract | Requires R7.5.3 and R7.5.4 closure |
| P1.6 | R8.2 physical Ryzen calibration | NO | Exact selected architecture is calibrated with semantic-equivalence gate | Requires R7.5.5 |
| P1.7 | R8.3–R8.5 official production training/freeze | NO | Official HU/3H policies trained and immutable | Requires R7.5.5 + R8.0 + R8.2 |

## Lane P0 execution order while monitoring runs

The priority order is deliberately chosen to minimize post-monitoring latency rather than to create more research:

1. **Freeze the dependency map and persistence contracts.** This prevents waiting for a long run and only then deciding how its evidence will be admitted.
2. **Prepare the post-x4 strategic evaluator path.** Reuse the already-frozen Phase-2 evaluator semantics; do not invent a new metric or protocol.
3. **Prepare R7.5.4 transfer plumbing parametrically.** Everything may be written/tested against a placeholder representation identifier, but execution must fail closed until a provisional H2/H3 winner artifact is supplied.
4. **Audit R8.0 acquisition tooling and selected-state missing facts.** This is independent of which H2/H3 representation wins and is a real downstream blocker.
5. **Prepare the Ryzen handoff/calibration package.** Heavy workloads should be executable from an exact frozen commit without rebuilding procedures after the strategic gate closes.
6. **Audit governance/downstream contracts for stale V1 assumptions.** Mechanical updates are allowed; strategic reinterpretation is not.

## Exact boundary at x4 completion

### If x4 returns `STABILITY_PASS`

Immediate critical path becomes:

```text
persist/certify exact x4 result
-> run complete frozen Phase-2 strategic evaluation on x4 policies
-> apply already-frozen H2/H3 selection rule
-> provisional representation winner OR BLOCKED
-> R7.5.4 SPNNIV3 action/sizing revalidation
```

Preparation work in P0 must ensure that there is no avoidable design or tooling pause between those arrows.

### If x4 returns `STABILITY_BLOCKED`

The finite-closure policy applies. Exactly one final winner-independent chance-variance remediation may be frozen/executed; it may not relax gates or shop seeds. If that final permitted remediation also fails, R7.5.3 closes `FAIL/BLOCKED`. Parallel work must not be used as a justification to create additional R7.5.3 sub-stages.

## Predeclared x4 evidence persistence contract

When the active x4 run completes, persist at minimum:

```text
schema
source_run_id
source_execution_sha
source_artifact_id
source_artifact_name
source_artifact_zip_digest
exact_result_file_sha256
result_status
all_training_cells_present
cross_seed_rows_expected = 8
cross_seed_rows_observed
cross_seed_mean_tv_max = 0.15
cross_seed_p95_tv_max = 0.35
all_cross_seed_rows_pass
representation_winner = null
selection_rule_changed = false
production_training_authorized = false
ready_for_tables = false
```

The stability result itself is not permitted to choose H2/H3. `STABILITY_PASS` only unlocks the already-frozen strategic evaluation.

## Post-x4 strategic evaluator reuse contract

The existing Phase-2 strategic evaluator remains the semantic authority. Reuse, unless an independently proven mechanical incompatibility requires a frozen correction:

```text
tools/r7_5_3c_phase2_extract_final_policies.py
tools/r7_5_3c_phase2_eval_matrix.py
tools/r7_5_3c_phase2_eval_worker.py
tools/r7_5_3c_phase2_eval_aggregate.py
```

Authoritative evaluator identities already established in the prior admission lineage:

```text
extractor blob: b2cc858fb1c23b7ea441caece08547515b79cbd1
matrix blob:    276089aa337e6c83d0e8fe347cfb029db9404784
worker blob:    293c530e1337ee0c67f6310d4b8285868fcdd106
aggregate blob: 891cd16e19e485965c1a45a3360470296c06fff2
```

The post-x4 harness must bind the new x4 policy checkpoints while preserving the frozen evaluator RNG namespace, heldout corpora, local-deviation semantics, pairwise semantics, common-reference diagnostic-only role, bootstrap count, materiality floor and selection cascade. Local deviation must continue to be described as `ONE_STEP_SELF_CONTINUATION_DEVIATION_GAIN`, **not exact exploitability**.

## R7.5.4 transfer preparation boundary

The authority remains `validation/R7_5_4_SPNNIV3_ACTION_SIZING_TRANSFER_PRECOMMIT_20260815.json`.

Reusable now without a new strategic claim:

```text
exact no-limit betting semantics
universal primitive/action resolver
state-local exact-action deduplication
candidate identities
sizing arithmetic/seed derivation
learning-quality thresholds
referee/crossplay metric definitions
160/320/640 escalation semantics
```

Not reusable as SPNNIV3 strategic sufficiency:

```text
historical V1-lineage winner/result itself
historical strategic action conclusion without SPNNIV3 rerun
any claim that PF0/PR0 is globally optimal merely because it was the Phase-2 control
```

## Governance staleness found during this parallel audit

`ROADMAP.md` is current enough to represent the active R7.5.3 state, but the machine-readable governance files are stale:

```text
STATUS.json
  version = 1.12.1-recovery.31
  R7.5 detail still describes legacy 184/H1/H2-era work
  r7_5_3 = PENDING
  primary_candidate text predates current SPNNIV3 H2/H3 admission

VERSION.json
  version = 1.12.1-recovery.26
  neural_schema = SPNNIV1
  status still describes R7.3/R7.4 verification era
```

These stale fields must **not** be “fixed” by relabelling SPNNIV3 as production while admission is open. A future schema-preserving governance refresh should distinguish:

```text
production/fallback neural schema = SPNNIV1 until R7.5.5
candidate successor schema = SPNNIV3
R7.5.3 = IN_PROGRESS_X4_CHANCE_COVERAGE_READMISSION
production_training_authorized = false
ready_for_tables = false
```

## R8.0 parallel work boundary

The repository already contains a capture-to-selected-state packet builder (`tools/build_r8_selected_state_packet.py`) and the production evidence machinery. The remaining blocker is not choosing H2 or H3; it is obtaining exact authoritative selected-state facts for the production economics/rules. Therefore R8.0 evidence acquisition can proceed independently while R7.5.3 runs, but pilot constants or inferred defaults may not substitute for missing first-party selected-state evidence.

## Definition of success for this parallel roadmap

The parallel roadmap is complete when all `P0.*` items are either:

- completed with durable repository evidence/tooling; or
- proven to require an unavailable external fact and reduced to one explicit acquisition dependency.

At that point the only remaining work after monitoring must be the genuinely sequential chain:

```text
x4 result
-> strategic H2/H3 decision
-> action/sizing decision
-> production freeze
-> exact R8 inputs / calibration / production training
-> R9 -> R10 -> R11 -> R12
```

No parallel task may turn `READY FOR TABLES` true.
