# SpinCore — LT2 OpenHoldem runtime integration preflight

Date: 2026-09-21  
Status: **LEGACY-FIRST REVIEW COMPLETE — NATIVE C++ INFERENCE PARITY NEXT**

## Mandatory legacy review

Before starting runtime integration, the historical archive
`Tentativas anteriores de SpinGo.zip` was inspected, especially:

`deepspin/user_deepspin.cpp`

The OpenHoldem source snapshot in
`repositorio_completo_openholdem.txt` was also checked for the user-DLL lifecycle.

## Legacy behavior worth preserving

The historical DeepSpin user-DLL already established useful runtime patterns:

1. inference is triggered on `DLLUpdateOnMyTurn()`;
2. the result is cached for later `ProcessQuery()` calls;
3. `DLLUpdateOnHeartbeat()` does not recompute strategy;
4. hand reset/new-round callbacks explicitly reset or refresh runtime state;
5. OpenPPL-facing outputs distinguish fold, call/check, raises and all-in;
6. model inference is kept local to the runtime rather than depending on a remote service.

These patterns reduce stale-state and duplicate-inference risk and should be preserved.

## Legacy behavior that must not be copied unchanged

The old runtime manually rebuilt a 292-feature observation from OpenHoldem symbols and contained its own hand/draw/board feature logic.

That path is not acceptable for LT2 because:

- the validated LT2 observation is canonical `SPNNIV1`, 126 bytes;
- the old duplicated feature path can diverge from the solver;
- the project has historical evidence that hand-category semantics can be wrong when board-only made hands are confused with Hero contribution;
- the legacy bridge frequently used defensive zero defaults for missing symbols, while the new runtime must fail closed when a missing value can change state, domain or action legality.

The new runtime must therefore reconstruct the canonical solver state and consume the exact solver-generated observation/action semantics rather than maintaining a second poker-state interpretation.

## Action semantics

The old DeepSpin DLL exposed seven contiguous action IDs.

LT2 preserves the same seven active action labels:

- FOLD;
- CHECK_CALL;
- POT_33;
- POT_50;
- POT_75;
- POT_100;
- ALL_IN.

However, LT2 carries them in the ten-slot universal network:

- 0 FOLD;
- 1 CHECK_CALL;
- 3 POT_33;
- 5 POT_50;
- 7 POT_75;
- 8 POT_100;
- 9 ALL_IN.

Slots 2, 4 and 6 are dormant.

The OpenHoldem bridge must use the canonical lean resolver for exact action amount/type semantics rather than reimplementing the old sizing formulas independently.

## Model-runtime change

The historical runtime embedded a simple exported neural model.

The frozen LT2 model family uses:

- card embeddings;
- categorical embeddings;
- history embeddings;
- one GRU layer;
- two ReLU dense layers;
- ten-output head.

For HU, eight independent Advantage models are evaluated and their raw outputs are averaged before unchanged lean regret matching.

Therefore the next gate is native C++ numeric inference parity. It is a prerequisite for embedding the frozen strategy in the Windows user-DLL.

## Native parity contract

The C++ implementation must:

- load only the frozen deployment weights;
- preserve THREE_HANDED AveragePolicy;
- preserve TRUE_HEADS_UP ENS8 raw-output averaging;
- preserve the unchanged regret-matching fallback;
- consume canonical SPNNIV1 observations;
- consume the external ten-slot legal mask;
- produce zero illegal-action mass;
- match Python probabilities within a small floating-point tolerance;
- produce zero argmax mismatches on the parity fixture corpus.

No strategic EV, holdout reuse or policy tuning is allowed.

After native parity passes, proceed to the actual OpenHoldem lifecycle/state-acquisition bridge.
