# R7.5.4 — Action-abstraction source and implementation audit

Date: 2026-08-13
Status: **SOURCE/IMPLEMENTATION AUDIT COMPLETE; NO ACTION CANDIDATE PASS**

`READY FOR TABLES = NO`.

## 1. Why this audit exists

R7.5.3 selects how poker state is represented to the neural network. R7.5.4 selects which legal aggressive actions are exposed to Deep CFR. These are different questions and must not be conflated.

The exact SpinCore game state, betting rules, chance, showdown and terminal utility remain authoritative. R7.5.4 changes only the **lossy action abstraction at the traversal/neural boundary**.

No R7.4 numerical evidence can be relabelled as evidence for a changed action set.

## 2. Current SpinCore control — physical code audit

Current source:

```text
include/spincore/action_abstraction.hpp
src/action_abstraction.cpp
src/hand_infoset_adapter.cpp
src/spin_traversal_state.cpp
tests/test_action_abstraction.cpp
```

Current six output slots are:

```text
0 Fold
1 CheckCall
2 ContextRaise
3 SmallPot
4 LargePot
5 AllIn
```

Current semantics:

### Preflop

```text
Fold          when facing a call
CheckCall     legal whenever actor has chips
ContextRaise  minimum legal raise-to
AllIn         exact all-in
```

No intermediate non-all-in preflop size is exposed.

### Postflop

```text
Fold          when facing a call
CheckCall     legal whenever actor has chips
SmallPot      33% of pot-after-call raise increment, clamped to legal raise range
LargePot      75% of pot-after-call raise increment, clamped to legal raise range
AllIn         exact all-in
```

For a fractional aggressive action, current code computes conceptually:

```text
call_target    = current street contribution + amount_to_call
pot_after_call = current total pot + amount actually called
raw_raise_to   = call_target + ceil(pot_after_call * fraction)
exact_raise_to = clamp(raw_raise_to, minimum legal raise-to, maximum legal raise-to)
```

The exact betting engine then validates/applies the resulting exact action.

Therefore the present control is correctly described as:

```text
preflop  = MIN / AI
postflop = 33 / 75 / AI
```

plus Fold/CheckCall when legal.

## 3. Legacy DeepSpin evidence

The previously audited DeepSpin action output had seven actions:

```text
Fold
Check/Call
33%
50%
75%
100%
AllIn
```

Postflop the percentage labels were distinct sizes. Preflop they were mapped through context-dependent real-raise logic rather than naively copied as identical postflop semantics.

This is historical design evidence only. It is not proof that the five aggressive postflop choices are optimal.

## 4. Full Crusher Framework 5 evidence

Authoritative source audit:

```text
validation/R7_5_CRUSHER_FRAMEWORK_FULL_SEMANTIC_AUDIT_20260813.md
source sha256 = 7ec68e2efc9790bfc02f47da690faa1856ab2be47a8d10bd178a2963fa78ce08
```

The physical framework contains 1,195 named blocks, including `#12 BETSIZES` with eight historical sizing concepts:

```text
Min
33
40
50
66
75
100
Max
```

It also distinguishes continuous faced-sizing / raise geometry throughout its 297 scenario blocks. The reusable lesson is that sizing is strategically relevant; the old thresholds/actions are not production truth.

The full Crusher audit explicitly assigns `40`, `66` and `Min` to possible R7.5.4 diagnostic/additional candidates if they are frozen before outputs and if any gain justifies branching cost.

### Max is not a separate SpinCore action

In the authoritative no-limit betting engine, the maximum legal commitment is the actor's all-in commitment. Exposing historical `Max` and exact `AllIn` as separate neural branches would create two labels for the same terminal chip commitment in states where maximum raise means all-in.

R7.5.4 therefore represents this concept once, as `ALL_IN`.

## 5. Problem in comparing action sets naively

Adding branches can appear to improve a solver simply because:

- it creates more output parameters;
- it changes sampling frequency;
- several nominal sizes clamp to the same exact raise in shallow-stack states;
- duplicate exact actions can split probability/regret mass across aliases;
- larger trees receive more compute if runtime rather than roots is held fixed.

The R7.5.4 comparison must explicitly prevent all five confounders.

## 6. Universal action vocabulary for the ablation

Every R7.5.4 candidate will use the same universal output vocabulary and model output width:

```text
0 FOLD
1 CHECK_CALL
2 MIN_RAISE
3 POT_33
4 POT_40
5 POT_50
6 POT_66
7 POT_75
8 POT_100
9 ALL_IN
```

Candidate action abstractions are **masks over this common 10-slot vocabulary**. This keeps output dimensionality/model capacity identical across candidates.

`ALL_IN` is exact. `MIN_RAISE` is exact minimum legal raise-to. Percentage primitives use the same pot-after-call formula as the current SpinCore control and are clamped by the exact betting engine.

## 7. Exact-action deduplication is mandatory

Nominal actions that resolve to the same exact action in a particular state must not remain as separate legal neural branches.

For each state:

1. generate every active candidate primitive;
2. resolve it to exact action type + exact raise-to amount;
3. suppress fractions that resolve to exact all-in when `ALL_IN` is active;
4. group remaining identical exact raise targets;
5. retain exactly one deterministic representative per group;
6. preserve Fold/CheckCall normally.

For a duplicate group of fractional candidates, choose the primitive whose *unclamped raw target* is closest to the realized exact target; break exact ties by smaller nominal fraction. If `MIN_RAISE` is active and the realized target is exactly the legal minimum, `MIN_RAISE` wins the alias group.

This rule is state-local and strategy-neutral. It prevents artificial probability splitting.

## 8. Separate postflop and preflop ablations

A Cartesian product of every preflop and postflop candidate would waste compute and make attribution difficult. R7.5.4 therefore has two sequential subgates.

### R7.5.4A — postflop sizing

Preflop is frozen to the exact current control:

```text
MIN_RAISE / ALL_IN
```

Postflop candidates:

```text
PF0_CONTROL_33_75_AI
    33 / 75 / AI

PF1_33_50_75_AI
    33 / 50 / 75 / AI

PF2_33_50_75_100_AI
    33 / 50 / 75 / 100 / AI

PF3_COMPACT_33_66_100_AI
    33 / 66 / 100 / AI

PF4_CRUSHER_COMPACT_40_66_100_AI
    40 / 66 / 100 / AI

PF_DENSE_REFERENCE
    MIN / 33 / 40 / 50 / 66 / 75 / 100 / AI
```

The first five are eligible to win. `PF_DENSE_REFERENCE` is an expensive diagnostic/referee and is **not automatically eligible to become production**. Its role is to detect whether every compact tree leaves material sizing value on the table.

The original precommitted A/B/C candidates are exactly retained as PF0/PF1/PF2. PF3/PF4 and the dense diagnostic are added before any R7.5.4 result exists, using the permission explicitly recorded in the full Crusher audit.

### R7.5.4B — preflop sizing

The R7.5.4A postflop winner is frozen for every preflop candidate.

Preflop candidates:

```text
PR0_CONTROL_MIN_AI
    MIN / AI

PR1_MIN_75_AI
    MIN / 75 / AI

PR2_MIN_50_75_AI
    MIN / 50 / 75 / AI

PR3_MIN_75_100_AI
    MIN / 75 / 100 / AI

PR4_MIN_50_75_100_AI
    MIN / 50 / 75 / 100 / AI
```

Percentage semantics remain pot-after-call raise increments, not hard-coded BB constants. This automatically adapts to open, isolation, 3-bet, HU and multiway pot geometry while retaining the exact realized size in public history.

This is deliberately smaller than importing every historical Crusher preflop branch.

## 9. Structural screen before strategic training

Every candidate must first pass a deterministic exact-state structural audit:

```text
legal exact actions valid                 100%
no duplicated exact neural actions        100%
all-in represented exactly                100%
minimum raise represented when specified  100%
monotonic fraction targets after dedup     100%
exact game state / settlement unchanged   100%
```

Record for HU and 3H separately:

```text
raw nominal aggressive branches/state
effective unique aggressive branches/state
fraction of nominal actions suppressed as exact aliases
fraction clamped to MIN
fraction clamped to ALL_IN
raise-to / pot-after-call target distribution
SPR and stack distributions of collisions
```

A candidate with an implementation correctness failure is ineligible regardless of strategic score.

## 10. Strategic comparison principles

R7.5.4 will not select the largest tree by default. The selected representation from R7.5.3 will be used for every candidate, and all candidate networks will share the same 10-output architecture.

Both subgates must use:

- exact same domain scenario cycle;
- exact same deal/chance seeds across candidates;
- exact same training initialization seeds;
- exact same optimizer/batch contract;
- exact same ICM terminal utility;
- exact same active/inactive universal action vocabulary;
- fixed roots, not fixed wall-clock time;
- no candidate-specific early stopping;
- no post-result candidate additions.

Tree cost must be reported rather than hidden:

```text
nodes/root
seconds/root
peak RSS
effective unique branches/decision
samples/root
```

Strategic evidence must include both learning stability and an exact-action omission/referee metric; fit quality alone cannot prove that an action abstraction is strategically sufficient.

## 11. Current gate meaning

This source/implementation audit does not select an action abstraction and does not emit R7.5 PASS.

At this moment:

```text
R7.5.3 representation experiment     ACTIVE on immutable SHA
R7.5.4 source/action audit           COMPLETE
R7.5.4 candidate outputs             NONE observed
R7.5.4 candidate selection           NOT STARTED
R7.5.5 production freeze             BLOCKED
READY FOR TABLES                     NO
```
