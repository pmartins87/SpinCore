# Strategic action sentinels — finite release gate design

`READY FOR TABLES = NO`

This document freezes the role of action-level canonical/extreme sentinels before production policies are available. The purpose is to detect two different failure classes that aggregate fit/stability metrics cannot exclude:

1. **integrity/runtime failure** — the wrong model, wrong runtime, wrong observation bytes or wrong action output is being loaded/executed;
2. **strategic plausibility failure** — the policy produces an action distribution that violates an explicitly precommitted expectation at a named canonical/extreme state.

These two layers must never be conflated.

## Implementation

```text
python/spincore/strategic_sentinel.py
schema = SPINCORE_STRATEGIC_ACTION_SENTINELS_V1
```

Each observation binds:

```text
sentinel_id
× exact production profile_id
× strategy domain
× exact model SHA-256
× exact neural observation SHA-256
× canonical sorted legal-action set
× exact six-action policy vector
```

The action fingerprint is SHA-256 over a canonical payload that serializes every probability with `float.hex()`. This is an integrity fingerprint, not a tolerance-based strategic metric.

## Layer 1 — exact integrity sentinel

For the same frozen model/runtime/profile and the same observation bytes, an accepted release must reproduce the precommitted action fingerprint exactly. A mismatch fails closed and is treated as a loader/runtime/model/observation identity problem until explained.

Integrity alone **cannot** authorize release. A system that perfectly reproduces an absurd policy is still absurd.

## Layer 2 — strategic plausibility sentinel

Plausibility rules are explicit bounds on named action probabilities at named sentinel states. The framework supports minimum and/or maximum probability bounds and requires a rationale for each rule.

The framework deliberately does **not** hard-code poker opinions such as `AA shove >= X%`. Those bounds may only be frozen after:

- the exact R8.0 production profile and payout semantics are known;
- the sentinel state can be reproduced exactly and described unambiguously;
- the rule is justified independently of the production policy result being audited.

This prevents post-result threshold tuning and prevents a generic intuition from being applied to a profile where ICM/payout semantics differ.

## Fail-closed completeness rule

A strategic sentinel gate is PASS only when:

- every required sentinel observation exists;
- every required sentinel has an exact integrity expectation;
- every required sentinel is covered by at least one precommitted plausibility rule;
- every integrity fingerprint matches exactly;
- every plausibility rule passes.

If plausibility rules are empty or incomplete, the sentinel gate is **NOT PASS**, even if all integrity fingerprints match.

The sentinel gate itself always reports `ready_for_tables = false`; final table authorization remains an R12 decision.

## Required future population

Before READY FOR TABLES, the final sentinel set must include both domains and must cover canonical/extreme states sufficient to detect gross strategic/pathology failures. At minimum the population work must consider:

- premium versus trash preflop holdings at shallow and deeper effective stacks;
- facing no raise, a raise and an all-in where those actions are legal in the frozen abstraction;
- HU and 3H separately;
- materially different payout/ICM states where the production profile permits them;
- short-stack, medium-stack and large-stack tournament states;
- exact action labels from the frozen action abstraction rather than informal words such as `call` or `shove` detached from action IDs.

The final list and numerical plausibility bounds must be committed **before inspecting the audited production policy's outputs on those sentinel states**. That precommit is required to avoid tuning the audit around the answer.

## Current status

```text
sentinel framework implementation = IMPLEMENTED
framework regression = PENDING at document creation
production sentinel state set = NOT YET POPULATED
production integrity baselines = NOT YET FROZEN
production plausibility bounds = NOT YET FROZEN
strategic sentinel gate = NOT PASS
READY FOR TABLES = NO
```

Population is intentionally deferred until exact R8.0 profile evidence and frozen R8.5 production policy identities exist. The framework itself can be regression-tested now without inventing production strategy thresholds.
