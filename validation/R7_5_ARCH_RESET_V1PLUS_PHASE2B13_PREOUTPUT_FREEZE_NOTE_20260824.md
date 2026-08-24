# Phase2B13 pre-output freeze note

Date: 2026-08-24  
Status: **BEFORE ANY PHASE2B13 RYZEN OUTPUT**

After the implementation audit and before any Phase2B13 execution, the precommit wording for the root-sample replacement mechanism was clarified to match the audited implementation exactly.

The scientific design, arms, K=64 budget, chance seeds, equal-compute comparison, gates and routing were not changed. The clarification replaces an ambiguous suppress/insert phrasing with the actual fail-closed mechanism: the memory proxy intercepts the ordinary initial-root Advantage `add` call and immediately delegates the replacement sample at the same reservoir insertion position.

This preserves reservoir `seen`, replacement-RNG call position and ordering relative to downstream Advantage samples. The change was committed before outputs and is not a response to Phase2B13 results.

Frozen Phase2B12 source result SHA-256 remains `dbccadae5805381d0188bef41fb62a72b25b42e03e5564ca88f05d9666e6e182`.

`PRODUCTION TRAINING = NO`; `READY FOR TABLES = NO`.
