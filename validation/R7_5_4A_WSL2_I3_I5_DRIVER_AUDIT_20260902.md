# R7.5.4A — WSL2 iterations 3–5 recovery driver audit

Date: 2026-09-02
Status: **PREPARED / STATICALLY AUDITED / NOT YET RUN ON THE RYZEN**

## Frozen evidence

The exact historical workflow at source execution SHA `457996944f76e9f1fa0475691df978f450259641` freezes five training iterations and 32 roots per iteration at root level 160. It chains `i2 -> i3 -> i4 -> i5`; iteration 5 performs finalization and the exact final AveragePolicy fit. The original final inventory requires `target_iteration=5`, `finalized=true`, `roots=160`, `average_policy_optimizer_steps=16384`, and `side_advantage_optimizer_steps=3*4096*5`.

The immutable recovery worker at recovery SHA `a7eb746b0ac32ef730568150e1e2c2757bb212d2` explicitly permits target iterations 2, 3, 4, and 5. Its collection mode preserves a one-root budget and deterministic partial collection continuation. Its fit mode consumes a completed 32-root partial checkpoint; at target iteration 5 it invokes `finalize_stage_seed` and writes the final report. For target iterations greater than 2 it requires the prior recovered fitted checkpoint to carry the same recovery execution SHA and exact original iteration-1 provenance.

## Sealed local starting point

The local fitted iteration-2 result is frozen by exact checkpoint SHA-256:

- seed `1737995611`: `a51e15b355de8bdb41cc69c03e0d37facfc10124e9eeadd0c935606344c806f2`
- seed `645939859`: `7139ff4d50df87695d177cde7371ae9994ad732367182b3a240c8683799f5df3`
- seed `1311335590`: `21e620ed270945e3a88d3a5f664be4440adf499c90b78a8c9f057b9459ed74cb`

The driver refuses to start scientific work unless all three fitted i2 checkpoints/reports validate against these identities and the frozen provenance.

## Driver contract

`tools/run_r7_5_4a_dense3h_wsl2_i3_i5.sh` preserves the frozen source SHA, recovery SHA, three seeds, original-i1 artifact/checkpoint provenance, Python 3.11.15, Torch 2.13.0+cpu, NumPy 2.3.5, and two Torch threads. It executes exactly 32 one-root collection barriers followed by one fit for each of iterations 3, 4, and 5. The three seeds may run in parallel within a root stage, but the next root cannot start until all three outputs validate.

Every reusable checkpoint is verified against its report, its direct predecessor hash, source/recovery identities, seed/domain/candidate, recovery provenance, and no-semantic-change flags. An interrupted output containing only one of checkpoint/report is moved to quarantine before recomputation. A complete-looking but contract-invalid output fails closed instead of being overwritten. Progress is exported after every durable barrier outside the WSL ext4 state root.

At iteration 5 fit, the driver additionally requires the historical final-report invariants: schema `SPINCORE_R7_5_ACTION_DOMAIN_FINAL_REPORT_V1`, 160 roots, 16384 AveragePolicy optimizer steps, `3*4096*5` side-advantage optimizer steps, no strategic selection permission at 160, and no production/table authorization.

## Governance

This driver only recovers the three historically missing `PF_DENSE_REFERENCE × THREE_HANDED` cells. Completion does not by itself authorize tables or production. After the three i5 final cells exist, the next required sequence remains: combine them with the 33 historical finals, require a complete 36/36 inventory, execute the frozen R7.5.4A-160 strategic evaluation, and only then advance to the R7.5.5 representation/action freeze decision.
