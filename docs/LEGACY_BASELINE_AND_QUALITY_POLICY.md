# SpinCore Legacy Baseline and Quality Policy

Status: **MANDATORY / CANONICAL**
Date: 2026-09-09

## Purpose

SpinCore is an evolution of the user's multi-year SpinGo/DeepSpin work, not a greenfield replacement. The archive `Tentativas anteriores de SpinGo.zip` is the historical baseline that must be consulted before redesigning any component already addressed there.

The project goal is a SpinGo AI that plays **extremely competently**. Scientific ceremony, certification, exhaustive proof, and repeated tests are not goals by themselves. Work is justified only when it can materially improve strategic quality, correctness, robustness, runtime viability, or a decision that affects those properties.

## Non-negotiable rules

1. **Legacy-first preflight.** Before creating or replacing any sampler, state representation, action abstraction, evaluator, traversal, training loop, buffer, network, checkpointing mechanism, runtime/inference path, or OpenHoldem integration, inspect how the same concern was handled in `Tentativas anteriores de SpinGo.zip` and record what is preserved, changed, and why.
2. **Evolution, not restart.** A new SpinCore component may replace a legacy component only when there is a concrete reason to expect improvement or correctness repair. Do not discard mature legacy behavior merely because a cleaner/newer implementation exists.
3. **Real SpinGo distribution is mandatory for final strategic decisions.** The final training/evaluation distribution must represent both 3-handed and heads-up play across the actual blind ladder and realistic stack distributions. A single-blind experiment such as 10/20 may be used only as a localized diagnostic; it cannot select or certify the final policy or global action abstraction.
4. **Preserve empirical scenario knowledge.** The historical scenario sampler's blind frequencies, HU/3H separation, blind-conditioned stack distributions, total-chip constraints, dealer/seat randomization, and any other empirically derived tournament-state distributions are project assets. They must not be silently replaced by simplistic fixed scenarios.
5. **No compute before game-spec sanity.** Before any heavy Ryzen/GitHub training, confirm that the sampled game distribution, legal actions, hand evaluation, utility/payout semantics, hidden/public state, and runtime action mapping correspond to the intended SpinGo game.
6. **Quality-first, bureaucracy-light.** Keep tests that can catch bugs or change a strategic choice. Remove or demote tests whose only purpose is formal certification/reproducibility and that cannot plausibly change play quality. Use adaptive evidence: cheap screen first, scale only when uncertainty can change the decision.
7. **Stop repeated testing when the decision is already clear.** Do not run test-after-test merely because a roadmap contains another gate. Every expensive test must state what decision it can change before it runs.
8. **Historical failures are design constraints.** The prior DeepSpin trained for roughly three months on the Ryzen and still made gross errors. Therefore 'more training' is never an acceptable default explanation for bad play. First investigate game semantics, evaluator correctness, state/representation, sampling distribution, objective/traversal, training/runtime mismatch, action mapping, and integration.
9. **Hand-category semantics must reflect Hero contribution where strategically required.** Historical bugs included board-only made-hand categories being treated as if Hero had made the hand; these errors can poison both hardcoded logic and learned targets. Evaluator/feature semantics must distinguish board texture from Hero-contributed hand strength.
10. **Preserve useful legacy assets even when not used unchanged.** The old DeepSpin contains scenario sampling, network/training infrastructure, traversal, buffers, C++ poker environment, OpenHoldem runtime, hardcoded Crusher material, and solver-v2 assets. Treat them as source material and regression references, not disposable artifacts.
11. **Do not let repository documents become dead policy.** `AGENTS.md`, `README.md`, and `ROADMAP.md` must point to this policy. Any future roadmap or implementation plan that conflicts with it is subordinate and must be revised before compute begins.
12. **When a new chat/session starts, re-anchor from the repository before continuing.** Read `AGENTS.md`, this file, and the current roadmap/status before proposing training or structural changes.

## Known historical lesson from the failed DeepSpin line

The prior DeepSpin line accumulated large training volume without translating that volume into reliable play. The user's own conclusion was that the failure could not be treated as simple undertraining: after about three months of continuous Ryzen training the agent still made gross mistakes. The project also uncovered basic semantic/evaluator defects, including 'two pair' / made-hand interpretations that could be satisfied by the board rather than by meaningful Hero contribution. A sophisticated trainer cannot learn the intended strategy if the game/state labels feeding it are wrong.

Accordingly, the preferred order is:

`correct game specification -> correct evaluator/state/action semantics -> representative tournament-state sampling -> efficient learning architecture -> strategic evaluation -> runtime integration`.

Not:

`train longer -> add gates -> retest indefinitely`.

## Current corrective direction

The 2026-09-09 discovery that R7.5.4 action-abstraction work was restricted to SB=10 / BB=20 means those results are **localized evidence only**. They must not drive the final global SpinGo policy/action-abstraction choice until re-evaluated under the recovered real tournament distribution from the legacy project.

The immediate project task is to inventory the complete `Tentativas anteriores de SpinGo.zip`, identify preserved strengths and known defects, and merge only demonstrated SpinCore improvements into that historical baseline before any new heavy training.
