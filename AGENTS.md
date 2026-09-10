# AGENTS.md — SpinCore mandatory operating instructions

These instructions are **binding for any AI/agent working on this repository**.

Before proposing or executing training, architecture changes, action-abstraction changes, sampler changes, evaluator/state changes, or runtime integration:

1. Read `docs/LEGACY_BASELINE_AND_QUALITY_POLICY.md`.
2. Read `ROADMAP.md` and `README.md`.
3. Treat `Tentativas anteriores de SpinGo.zip` as the historical baseline of years of prior work, not as optional reference material.
4. Inspect the legacy implementation of any component before replacing or redesigning that component.
5. Do not run heavy compute until game specification and sampling realism are confirmed.
6. Do not use a single blind level (including 10/20) to select the final SpinGo policy or global action abstraction.
7. Preserve the real 3-handed/HU blind ladder and empirically derived blind/stack distributions from the legacy project unless a better evidence-based replacement is justified.
8. Optimize for actual poker-playing strength and correctness, not certification ceremony. Run only tests that can catch a meaningful bug or change a strategic decision; use adaptive stopping instead of test-after-test.
9. Prior DeepSpin trained for roughly three months and still made gross errors. Never default to 'more training' before checking evaluator semantics, state representation, sampling, objective/traversal, action mapping, and training/runtime parity.
10. Historical board-only/made-hand interpretation bugs are a permanent warning: distinguish board texture from Hero-contributed hand strength wherever strategically required.
11. If another document conflicts with the legacy-first quality policy, stop and reconcile the conflict before compute.
12. On a new chat/session, re-read these files before continuing so these directives do not depend on conversational memory.

Current corrective state (2026-09-09): R7.5.4 results produced only at SB=10/BB=20 are localized evidence and must not be promoted to a global final-policy/action-abstraction decision until re-evaluated under the representative legacy SpinGo tournament-state distribution.
