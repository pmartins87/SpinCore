# AGENTS.md — SpinCore mandatory operating instructions

These instructions are **binding for any AI/agent working on this repository**.

Before proposing or executing training, architecture changes, action-abstraction changes, sampler changes, evaluator/state changes, runtime integration, or Ryzen heavy compute:

1. Read `PROJECT_CONTRACT.yaml`, the relevant files under `contracts/`, and `contracts/AUDIT_STATUS.yaml` first.
2. Read `docs/LEGACY_BASELINE_AND_QUALITY_POLICY.md` and `docs/RYZEN_OPTIMIZATION_POLICY.md`.
3. Read `CURRENT_WORK.md`.
4. Read `ROADMAP.md` and `README.md`.
5. Treat `Tentativas anteriores de SpinGo.zip` as the historical baseline of years of prior work, not as optional reference material.
6. Inspect the legacy implementation of any component before replacing or redesigning that component.
7. Do not run heavy compute until game specification and sampling realism are confirmed.
8. **Any substantial workload run on the user's Ryzen must be optimized for that Ryzen before the long run begins.** Reuse the proven DeepPot many-worker/one-thread-per-worker pattern when tasks are independent, run a short worker-count benchmark on the actual machine, persist the selected profile, and do not knowingly launch a long serial/low-utilization job merely because it is functionally correct.
9. Do not use a single blind level (including 10/20) to select the final SpinGo policy or global action abstraction.
10. Preserve the real 3-handed/HU blind ladder and empirically derived blind/stack distributions from the legacy project unless a better evidence-based replacement is justified.
11. Optimize for actual poker-playing strength and correctness, not certification ceremony. Run only tests that can catch a meaningful bug or change a strategic decision; use adaptive stopping instead of test-after-test.
12. Prior DeepSpin trained for roughly three months and still made gross errors. Never default to 'more training' before checking evaluator semantics, state representation, sampling, objective/traversal, action mapping, and training/runtime parity.
13. Historical board-only/made-hand interpretation bugs are a permanent warning: distinguish board texture from Hero-contributed hand strength wherever strategically required.
14. CPU utilization is a diagnostic, not the goal: maximize useful solver/training throughput without changing strategy semantics or creating oversubscription/waste. Reprofile the next real bottleneck when acceleration shifts it.
15. If any implementation or document conflicts with an active `PROJECT_CONTRACT` invariant, stop and reconcile the conflict before affected work proceeds; no lower-priority document silently overrides the contract.
16. New substantial work must identify the active contract IDs it relies on, challenges, changes, or supersedes.
17. Read the current contract status rather than assuming a migration state. If it is ever `BOOTSTRAP_AUDIT_REQUIRED`, no new long training or architecture-changing stage may start unless the affected domains are audited. Under `COMPLETE`, new substantial work still must satisfy the relevant invariant IDs, stage/performance manifests, and fail-closed gates.
18. On a new chat/session, re-read the contract and these files before continuing so directives do not depend on conversational memory.

Current governance state (2026-09-29): `PROJECT_CONTRACT.yaml` is COMPLETE. Historical fixed-10/20 R7.5.4 evidence remains localized/superseded for global selection. The active 10115->10315 semantic long run is grandfathered to finish untouched; its sequential fitter is not reusable for restart/extension until the contract-required parity/throughput repair is satisfied.
