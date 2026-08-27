# R7.5.4A — WSL2 dense-3H recovery milestone: i2c1

Date: 2026-08-27
Status: **I2C1 COMPLETE / RECOVERY NOT FINALIZED / NON-STRATEGIC MILESTONE**
Scope: `PF_DENSE_REFERENCE × THREE_HANDED`, three frozen training seeds
Branch purpose: documentation / continuity only. This record does not alter the frozen R7.5.4A experiment, candidate, thresholds, seeds, budgets, policy semantics, optimizer semantics, reservoir semantics, root order, or production authorization.

## Executive result

The Ryzen 9 / WSL2 local recovery path successfully completed the **first durable collection root of iteration 2 (`i2c1`) for all three frozen seeds**. All three workers returned code 0 and the bootstrap validator reported `i2c1 PASS` for each seed followed by `I2C1 COMPLETE for all three seeds`.

This is a mechanical recovery milestone only. It is **not** completion of iteration 2, not completion of the three missing `PF_DENSE_REFERENCE × THREE_HANDED` cells, not the historical 36/36 closure, and not a strategic PASS. Each worker report explicitly had `finalized=false`, `final_report=null`, `production_training_authorized=false`, and `ready_for_tables=false`.

## Frozen provenance validated locally

- Original frozen source execution SHA: `457996944f76e9f1fa0475691df978f450259641`
- Recovery execution SHA: `a7eb746b0ac32ef730568150e1e2c2757bb212d2`
- Source training run: `31804178848`
- Candidate: `PF_DENSE_REFERENCE`
- Domain: `THREE_HANDED`
- Target iteration: `2`
- Operation: `collect`
- Root budget for this durable chunk: `1`
- Recovery intervention: `MECHANICAL_MID_ITERATION_CHECKPOINT_ONLY`

The bootstrap revalidated frozen source identities for:

- `python/spincore/r7_5_action_stage.py`
- `python/spincore/r7_5_action_training.py`
- `python/spincore/r7_5_action_checkpoint.py`
- `python/spincore/r7_5_action_cfr.py`
- `python/spincore/r7_5_action_uncertainty.py`
- `python/spincore/r7_5_action_contract.py`
- `python/spincore/r7_5_action_stage_contract.py`

The local mid-iteration equivalence gate passed: **9 tests passed in 4.67 s**.

## Frozen local runtime

- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- Python: `3.11.15`
- Torch: `2.13.0+cpu`
- NumPy: `2.3.5`
- `torch_threads`: `2`

The initial bootstrap attempt stopped safely because `unzip` was absent. After installing the required Ubuntu packages, the same bootstrap was rerun and all identity/equivalence checks passed before collection began.

## Per-seed i2c1 evidence

### Seed `1737995611`

- Input checkpoint SHA-256: `ea598ec624ee2e4e72fc8c3780c53863d6f116d5d9baa9495bcbbfe7cfadea2c`
- Output checkpoint SHA-256: `0a7e88af09b3cfd2352cdf76e3a882a416ceb8a2a6edc946b576078bbca4e172`
- Roots added / collected after: `1 / 1`
- Nodes added: `17,483`
- Strategy seen added: `3,059`
- Advantage seen added: `3,297`
- Advantage decision visits: `8,642`
- Effective unique aggressive branches: `19,082`
- Effective aggressive branches / decision: `2.208053691275168`
- Nominal aggressive branches: `67,444`
- Nominal aggressive branches / decision: `7.804211987965749`
- Tree collection seconds: `16245.249215712998`
- Wall seconds: `16248.520952469002`
- Worker result: PASS / return code 0

### Seed `645939859`

- Input checkpoint SHA-256: `ba02b8a6b27da27b891c51a2e90bb437810ac2c44db6ca498375ca83be8cde09`
- Output checkpoint SHA-256: `1e37a635e2b763c95cc42158cf6f7ed33924eba42ad7d0c27bf2ad024a528987`
- Roots added / collected after: `1 / 1`
- Nodes added: `10,295`
- Strategy seen added: `36,076`
- Advantage seen added: `1,957`
- Advantage decision visits: `5,038`
- Effective unique aggressive branches: `10,944`
- Effective aggressive branches / decision: `2.172290591504565`
- Nominal aggressive branches: `39,452`
- Nominal aggressive branches / decision: `7.830885271933307`
- Tree collection seconds: `34397.004131497`
- Wall seconds: `34400.145772817`
- Worker result: PASS / return code 0

### Seed `1311335590`

- Input checkpoint SHA-256: `064713c596b6e860f25240c6b649aba00126346363aa5c6790c179ddb5e2e5ac`
- Output checkpoint SHA-256: `0b356aa5eef8dc61de55509afdb9ed8fbf7c6728ea34a08b80f29ff60c873e9b`
- Roots added / collected after: `1 / 1`
- Nodes added: `14,159`
- Strategy seen added: `154,398`
- Advantage seen added: `2,616`
- Advantage decision visits: `7,127`
- Effective unique aggressive branches: `15,844`
- Effective aggressive branches / decision: `2.223095271502736`
- Nominal aggressive branches: `55,480`
- Nominal aggressive branches / decision: `7.784481549038866`
- Tree collection seconds: `21045.57075047`
- Wall seconds: `21048.721765089`
- Worker result: PASS / return code 0

Descriptive totals across the three first roots: `41,937` nodes added, `193,533` strategy samples seen, and `7,870` advantage samples seen. These totals are operational observations only and have no strategic PASS/FAIL meaning.

## Semantic non-intervention evidence

All three reports asserted:

- `deck_seed_formula_changed=false`
- `optimizer_semantics_changed=false`
- `policy_semantics_changed=false`
- `reservoir_semantics_changed=false`
- `root_order_changed=false`

Therefore the completed work is classified as a durable mechanical continuation of the frozen source, not as a new experiment.

## Exact recovery sequence implied by the frozen recovery workflow

The recovery workflow at recovery SHA `a7eb746...` partitions iteration 2 into **32 one-root durable collection stages**: `i2c1` through `i2c32`. Each stage consumes the previous stage's checkpoint and increments `expected_roots_collected` by one. After `i2c32`, `i2finish` performs the iteration-2 `fit` operation with `expected_roots_collected=32` and produces stage `i2`.

Thus the completed local work is currently **root 1 of 32 collection roots in iteration 2**. The next mechanical stage is `i2c2`, consuming each seed's validated `i2c1/checkpoint.pt`, with the same source/recovery identities and frozen semantics.

The currently inspected recovery workflow defines the iteration-2 recovery chain through `i2finish`; no later-stage local command should be invented from assumption. Before iteration 3 or any later continuation, the exact frozen protocol must be read/derived from repository evidence and documented.

## Local durable state — preservation rule

The bootstrap explicitly instructed:

`Do not delete /home/rz9/spincore_r754_dense3h_recovery.`

That directory is now part of the recovery evidence/state because it contains the durable per-seed checkpoints required for exact continuation. Do not recreate or replace it while the current recovery is active.

The bootstrap exported:

`/mnt/c/SpinCoreAI/SpinCore/SpinCore_R7_5_4A_WSL2_BOOTSTRAP/results_i2c1`

The machine-readable `SUMMARY.json` from that export remains the preferred evidence package to ingest/preserve alongside this transcript-derived record. Once supplied, its hashes/fields should be cross-checked against this record before launching the next local stage.

## Governance consequences

1. Do not treat `i2c1 PASS` as strategic evidence.
2. Do not modify seeds, K, thresholds, policy semantics, optimizer semantics, reservoir semantics, deck-seed formula, root order, or source code to accelerate recovery.
3. Preserve all three `i2c1` output checkpoints and the WSL ext4 recovery tree.
4. Ingest and verify `results_i2c1/SUMMARY.json` before sealing the machine-readable milestone.
5. Continue with the exact next durable stage (`i2c2`) only through a script derived from the frozen recovery worker/workflow contract, not by manual reconstruction.
6. The broader frozen order remains: complete the three missing dense-3H historical cells → validate historical 36/36 → run frozen R7.5.4A-160 strategic evaluation → non-gating V1+ sidecar diagnosis → R7.5.5 decision/freeze.

## Context preservation / handoff statement

If this work is resumed in another chat, the key fact is: **local WSL2 recovery has been validated and `i2c1` is durably complete for all three dense-3H seeds; recovery itself remains incomplete.** The next required evidence input is the exported `SUMMARY.json`, and the next compute stage is `i2c2` under the exact same frozen identities and semantics.
