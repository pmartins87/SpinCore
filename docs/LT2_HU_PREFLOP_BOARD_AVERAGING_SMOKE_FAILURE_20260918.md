# SpinCore — LT2 board-averaging smoke first-run failure

Date: 2026-09-18
Status: **DIAGNOSED AND FIXED — IMPLEMENTATION FAILURE, NOT STRATEGY EVIDENCE**

## Observed failure

The first K1-vs-K4 mechanics smoke stopped with:

`RuntimeError: HU board averaging changed preflop sample count across boards`.

No training memory was written, no optimizer step ran, and the preserved Stage-B checkpoint was not modified.

## Root cause

The original K4 implementation reset the same global external-sampling RNG state before each board variant and assumed that this would keep the preflop sampled opponent actions identical.

That assumption is false for the recursive Deep-CFR traversal.

The traversal is depth-first. At a traverser preflop node, one action branch can enter postflop and consume a board-dependent number of opponent-action RNG draws before recursion returns and explores a later preflop action branch.

Therefore two different future boards can consume different postflop RNG histories before a later preflop opponent node is reached.

Resetting the RNG only at traversal start does not prevent this feedback.

The result is that alternate boards can sample different later preflop opponent actions, producing a different number/order of preflop Advantage samples.

## Why the smoke was valuable

The invariant check caught exactly the kind of hidden semantic contamination the gate was designed to detect.

The failure does **not** show that board averaging is ineffective. It shows that the first integration did not isolate future-board chance cleanly.

## Fix

The canonical board-0 traversal now records every sampled **preflop opponent action** together with its preflop observation and legal action set.

For each alternate board:

- the same canonical preflop opponent-action sequence is replayed;
- the observation and legal set must match at every replayed node;
- one dummy RNG draw is consumed for every replayed preflop opponent action, matching the local sampling call count;
- postflop opponent sampling remains ordinary;
- board 0 remains a canonical traversal;
- after all board variants, the RNG state is restored to the state reached by canonical board 0.

This makes later preflop sampling independent of board-specific postflop RNG consumption while preserving the canonical K1 path.

A small overridable opponent-sampling hook was added to the generic collector. Its default implementation is exactly the previous direct `sample_action` behavior, so canonical training semantics remain unchanged when board averaging is disabled.

## Next action

Rerun:

```bash
bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh
```

The smoke must pass before any further causal inference.

Even if it passes, do **not** immediately start a K4 training continuation. The next scientific gate is Stage-A -> Stage-B causal attribution: determine whether target-noise/sign errors actually explain the observed HU-Jammer regression rather than merely improving a convenient proxy.
