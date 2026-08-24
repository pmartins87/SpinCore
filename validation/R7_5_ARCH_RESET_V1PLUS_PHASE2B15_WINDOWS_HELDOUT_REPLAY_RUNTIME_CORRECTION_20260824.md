# R7.5 Architecture Reset — Phase2B15 Windows Heldout Replay Runtime Correction

Date: 2026-08-24

## Trigger

The first Windows/Ryzen Phase2B15 execution at HEAD `553f2df3e0d924b30d396b6ec707369e726c0ab8` passed all frozen prerequisite, model, solver ABI, explicit-deal, and synthetic gates, then failed before a scientific result with:

`RuntimeError: Phase2B15 canonical heldout observation drift`

The frozen heldout V3 corpora themselves remained hash-identical and Phase2B14 had reproduced all Phase2B13 metrics exactly.  The failure was therefore localized to B15's attempt to reconstruct a historical heldout state from `scenario_index + deck_seed + action_path` on Windows.

## Root cause

The immutable heldout V3 corpus was generated on Ubuntu/Linux at the frozen heldout execution.  `HandEngine` constructs a seeded deal by feeding `std::mt19937_64` into `std::shuffle`.  The C++ standard fixes the random-engine sequence but does not require different standard-library implementations to use the same `std::shuffle` permutation algorithm.  Replaying the same stored `deck_seed` under MSVC on Windows is therefore not a portable identity contract for a corpus originally materialized under libstdc++ on Linux.

This is a transport/reconstruction defect in the new B15 diagnostic, not evidence against the B15 posterior hypothesis and not a failure of the frozen heldout corpus.

## Scientific correction

Phase2B15 does not need the historical hidden opponent cards or historical hidden board.  Its conditional-IID proposal law conditions only on the current actor's preflop private cards and then resamples all opponent private cards and the hidden board.

SPNNIV3 already transports the current actor's two private cards losslessly up to true suit symmetry:

- ordered rank tokens for hole slots 0 and 1;
- the same-suit relation for hole slots (0,1);
- no physical suit labels.

The runtime correction therefore:

1. derives a deterministic canonical suit-isomorphic physical two-card representative directly from the frozen SPNNIV3 bytes;
2. creates a valid explicit 3-handed deal with those actor cards and arbitrary distinct filler cards for hidden opponents/board;
3. replays the frozen public `action_path` using the authoritative current action abstraction;
4. requires byte-identical frozen SPNNIV3, actor, active mask, and legal slots at the target continuation state;
5. passes that canonical snapshot to the existing B11 conditional-IID generator, which consumes only the current actor's fixed hole cards and resamples all other cards.

Thus the probability space is unchanged up to poker suit symmetry, while the invalid cross-platform dependence on `std::shuffle` is removed.

## Non-changes

This correction does **not** change:

- selected anchors or heldout hashes;
- K=64;
- two independent blocks;
- two final Phase2B13 candidate behavior seeds;
- 30-worker ceiling;
- posterior likelihood definition;
- 25% continuation behavior floor inherited from Phase2B13;
- traversal RNG namespaces;
- target iteration;
- all precommitted scientific gates and decision hierarchy;
- training authorization (`false`);
- production authorization (`false`);
- architecture-winner status (`unset`).

Any partial from the failed pre-correction execution is rejected unless it carries the new runtimefix schema marker, so corrected and pre-correction partials cannot be silently mixed.

## Runtime preflight added

Before the 16,384 target traversals begin, the corrected launcher must reconstruct **all 64 selected anchors** using the canonical explicit-deal path and prove byte-identical SPNNIV3/actor/legal identity.  A failure at that gate blocks the run.

## Governance

The failed `553f2df...` execution produced no admissible Phase2B15 scientific result.  Phase2B15 remains the same frozen read-only experiment, now with a compute/runtime reconstruction correction only.  No threshold relaxation, seed change, anchor change, or scientific redesign is authorized.
