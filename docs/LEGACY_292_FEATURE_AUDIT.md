# Legacy DeepSpin 292-feature observation audit

Date: 2026-09-11
Status: **DECISION RECORDED — DO NOT REINTRODUCE THE 292-FLOAT VECTOR AS THE FIRST-RELEASE INPUT**

## What the legacy vector contained

The archived DeepSpin v60 observation is exactly 292 floats:

- cards: 104 (52 Hero one-hot + 52 board one-hot)
- numeric state: 20
- street: 4
- position: 11
- hand strength: 40
- draws: 12
- board texture: 29
- action context: 26
- history summaries: 39
- legal mask: 7

The observation was not arbitrary; it encoded a great deal of strategically relevant poker knowledge. In particular, the later archived implementation repaired earlier hand-strength semantics by distinguishing board-only made hands from hands that actually use Hero hole cards.

## Why 292 itself was not the main compute problem

The legacy AdvantageNet was a flat MLP with hidden layers `[1024, 1024, 512, 512]`, producing about **2,140,679 parameters**. The legacy PolicyNet used `[1024, 512, 512]`, producing about **1,091,079 parameters**. Together that is about **3.23 million parameters**.

Reducing only the flat input from 292 to 128 dimensions while leaving those hidden layers unchanged would still leave about **2.90 million parameters** across the two networks — only about a 10% reduction. Therefore the legacy cost was driven much more by the very wide MLPs and Deep-CFR traversal volume than by the number 292 alone.

## Why the 292 representation was nevertheless too complicated for the first repaired SpinCore

The vector duplicates information heavily:

- raw cards already determine hand class, draws and board texture;
- 81 separate dimensions (`40 hand strength + 12 draws + 29 board texture`) manually re-encode information already present in the cards;
- 65 more dimensions (`26 action context + 39 history`) are hand-engineered summaries of betting history;
- legal actions are both encoded and separately masked by the policy machinery.

Hand-crafted derived features can improve sample efficiency, but they also create a large semantic bug surface. The historical two-pair/board-only problem demonstrates that a wrong derived feature can poison learning even when the underlying cards are correct.

## Current SpinCore V1 is a materially leaner representation

The current `SPNNIV1` neural payload represents the same decision state structurally rather than as a 292-float flat vector:

- 7 card tokens (2 hole + up to 5 board)
- 16 numeric values
- 8 categorical values
- 6 legal-action flags
- up to 32 public-history tokens

The V1 network embeds cards/categories/history and uses a small GRU for history. Its recovered default architecture is about **152,438 parameters per model**, roughly an order of magnitude smaller than the legacy pair of wide MLPs.

The exact underlying `CanonicalInfoset` still retains hole cards, board, domain, street, dealer relation, stacks, commitments, statuses, pot, to-call, current bet, blinds, blind index, legal actions and public history. Therefore choosing V1 as the neural boundary does not mean discarding the exact game state.

## Decision

For the first functional SpinCore:

1. **Do not restore the full 292-float observation as the training input.**
2. Use the compact exact-state-derived V1 representation as the default first-release neural representation.
3. Preserve the legacy 292-feature definitions as a **semantic checklist and diagnostic library**, not as mandatory neural inputs.
4. Restore a legacy derived feature only when there is a concrete reason that the compact representation cannot learn or distinguish an important strategic situation efficiently.
5. Do not launch another broad representation tournament. Previous richer SpinCore representations already failed to demonstrate a robust strategic improvement over the V1 fallback; added representation complexity is not presumed beneficial.
6. Any feature added later must be directly computable from the exact canonical state and must have identical training/runtime semantics.

This is a scope reduction intended to improve both training efficiency and correctness without throwing away the poker knowledge accumulated in the legacy feature definitions.

## Important caveat

This decision does **not** claim that every one of the 292 legacy features is useless. Many are strategically meaningful. The decision is that the same information should not automatically be supplied through 292 hand-coded floats when a compact exact representation can carry the state with far fewer model parameters and a smaller semantic failure surface.
