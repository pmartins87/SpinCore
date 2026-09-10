# SpinCore Current Work

Date: 2026-09-09
Status: **CORRECTIVE LEGACY-FIRST AUDIT — NO HEAVY TRAINING**

## Why work is paused

R7.5.4 action-abstraction training/evaluation was discovered to be restricted to SB=10 / BB=20. That can be useful as localized evidence but cannot represent the full SpinGo tournament distribution and therefore cannot select the final policy or global action abstraction.

The project had already been given a multi-year legacy archive, `Tentativas anteriores de SpinGo.zip`, containing a substantially richer SpinGo-specific foundation. The archive must now be treated as the baseline rather than as optional historical material.

## Immediate task

Audit the complete legacy archive and build a preservation/migration map:

- what must be preserved unchanged in semantics;
- what was known to be defective and must not be revived;
- what SpinCore has genuinely improved;
- what SpinCore accidentally lost;
- what the single consolidated final architecture should be before further heavy compute.

## Legacy assets already confirmed

The legacy `deepspin/scenario.py` contains:

- both 3-handed and true-HU episode sampling;
- nine blind levels from 10/20 through 100/200;
- separate empirical blind-frequency weights for 3H and HU;
- blind-conditioned stack distributions;
- 1500 total chips;
- randomized live seats/dealer and one dead seat for true HU;
- fallback approximations for sparse late-blind samples rather than silently deleting those states.

The legacy DeepSpin stack also contains:

- a C++ poker environment;
- Deep CFR trainer/traversal;
- advantage and average-policy networks;
- seven actions (fold, check/call, B33, B50, B75, B100, all-in);
- 292-feature neural observation;
- reservoir buffers;
- checkpoint/RNG continuity;
- multiprocessing rollout workers;
- OpenHoldem inference/runtime code;
- Crusher Framework 5 + hardcoded C++ strategy material;
- solver-v2 assets including the historical 184-flop abstraction.

## Historical failure lesson

The earlier DeepSpin trained continuously for roughly three months on the Ryzen and still made gross mistakes. The user's conclusion was that this was not adequately explained by insufficient training. Historical debugging also found basic semantic/hand-strength mistakes, including treating board-created made hands/two-pair structures as if they represented meaningful Hero hand strength. Therefore additional training is never the first remedy for poor play; first audit game semantics, evaluator/features, sampling distribution, traversal/objective, action mapping, and training/runtime parity.

## Do not do next

- Do not resume dense 3H i3-i5 merely to complete an old matrix.
- Do not run the PF0-PF4 10/20-only comparison as a final selector.
- Do not start R8 heavy training.
- Do not add new certification/reproducibility gates unless they can materially change playing quality or catch a real correctness problem.

## Required next deliverable

A concise legacy-vs-SpinCore architecture map followed by one consolidated training/runtime plan. Only after that map is complete should compute resume.
