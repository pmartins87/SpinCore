# SpinCore — Lean High-Quality Execution Policy

Date: 2026-09-02
Status: proposed operating policy for project simplification without intentional strategic-quality reduction.

## Product objective

Build an extremely competent Spin & Go agent for the offline simulator. The project is an engineering product, not a reproducibility paper, benchmark paper, certification exercise, or formal homologation program.

## Governing rule

A step remains mandatory only when its outcome can materially change at least one of these:

1. the selected poker strategy / representation / action abstraction;
2. expected strategic quality or exploitability;
3. correctness of poker rules, state transitions, payouts, observations, legal actions, or inference;
4. ability to train enough data/model capacity within available compute;
5. ability to run complete matches reliably.

If a step only strengthens provenance, publication-grade reproducibility, formal certification, documentary completeness, or proof of an already decision-insensitive fact, it is non-blocking or removed.

## Decision policy

Use progressive evidence and adaptive stopping instead of exhaustive fixed matrices.

- Start with the cheapest experiment capable of changing the decision.
- Escalate compute only when candidates remain practically close or uncertainty can plausibly reverse the choice.
- Stop evaluating a candidate once it is clearly dominated on strategically material criteria and there is no credible mechanism by which more samples would make it preferable.
- Do not require every cell in a historical matrix merely for matrix completeness when the missing cell cannot plausibly alter the engineering decision.
- Use theory and established poker/ML structure as priors. Empirical proof is reserved for uncertain, high-impact claims.
- Reuse valid existing evidence aggressively.

## Keep because it directly protects playing quality

- exact poker rules, legal-action resolution, stack/pot/SPR and tournament payout semantics;
- HU and 3-handed coverage;
- tests that catch state/action identity errors, impossible actions, observation corruption, or inference/runtime mismatches;
- comparison of action abstractions when the choice can materially affect EV;
- enough held-out/crossplay evaluation to distinguish competitive candidates;
- training stability checks only to the extent instability can degrade the selected policy;
- production-scale training of the selected architecture;
- strategic evaluation against strong baselines / self-play / crossplay;
- end-to-end simulator play and runtime correctness;
- exploitation/adaptation only when it demonstrably improves expected value without creating larger strategic weaknesses.

## Demote or remove as blocking requirements

- bit-for-bit or 1e-9 fresh-process reproducibility when the policy quality is unchanged;
- repeated independent confirmations after the engineering decision is already robust;
- all-seed/all-cell completion for its own sake;
- publication-style bootstrap counts or confidence rituals when a much smaller adaptive test resolves the decision with a large margin;
- frozen-precommit bureaucracy that prevents reacting rationally to strong new evidence;
- formal release certification / homologation gates that do not change strategy or runtime correctness;
- documentary gates whose only output is permission to proceed;
- hash/provenance ceremony beyond what is useful for resumability, preventing accidental data mixing, and identifying the model actually being run.

## Immediate R7.5.4 rule

The active recovery of the three missing PF_DENSE_REFERENCE × THREE_HANDED cells is not automatically entitled to consume weeks merely because the historical design expected a 36/36 matrix.

While the current root is allowed to continue, the existing 33 final cells and referee design must be re-evaluated under this policy. Continue the dense-3H recovery only if completing it has a credible chance of changing the selected action abstraction or materially improving the final strategy. Otherwise preserve the completed work, stop at a durable barrier, and select/confirm the strongest practical action abstraction with an adaptive decision-focused comparison.

The original R7.5.4 referee remains useful as a source of high-quality evaluation ideas (exact-action identity, common random numbers, held-out play, crossplay), not as an immutable obligation to execute every originally frozen sample count.

## Lean path to a strong playing agent

1. Select representation + action abstraction using existing evidence plus only decision-relevant additional experiments.
2. Train the strongest selected architecture at meaningful production scale on HU and 3H.
3. Evaluate strategic quality with adaptive held-out crossplay and strong baselines; increase sample count only near close decisions.
4. Integrate the selected policy into the offline simulator runtime immediately and run complete matches throughout development.
5. Fix observed strategic/runtime defects and retrain when the expected-value benefit justifies it.
6. Declare the product ready when it plays complete matches reliably and the strategic evidence supports excellent performance; no separate formal homologation ceremony is required.

## Quality safeguard

"Lean" means removing work that does not affect the final agent, not lowering the ambition for the agent. A shortcut is forbidden when it creates a credible risk of choosing a materially weaker strategy, masking a poker-engine error, training the wrong objective, or shipping a broken runtime.
