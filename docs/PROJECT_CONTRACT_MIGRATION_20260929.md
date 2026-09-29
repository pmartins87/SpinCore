# SpinCore Project Contract Migration

Status: **ACTIVE / NOT YET COMPLETE**  
Date: 2026-09-29

## Decision

SpinCore will no longer rely on conversational memory, ROADMAP chronology, or
human recollection to preserve active design decisions.

`PROJECT_CONTRACT.yaml` is the root of one logical project contract. The
contract may be physically split into machine-readable modules, but its scope is
**all active invariants**, not only performance.

The user is not expected to reconstruct or enumerate prior decisions.

## Required audit scope

The migration must inspect the full repository history needed to recover active
constraints, including at minimum:

- ROADMAP.md and CURRENT_WORK.md;
- README and long-training plans;
- frozen precommit / validation artifacts;
- training and evaluation specs;
- Ryzen optimization policy and benchmark results;
- action-space / abstraction / representation documents;
- scenario generation and game-domain contracts;
- chance-sampling and RNG decisions;
- model / ensemble / specialist decisions;
- checkpoint / resume / lineage rules;
- validation gates, sealed holdouts, statistical rules and stop criteria;
- DeepCrusher benchmark contracts and DC0/DC1/DC2 progression;
- OpenHoldem/OpenPPL runtime and deployment semantics;
- regression, sanity and pathology guards;
- source code and tests where they encode a stronger current invariant than old
  prose;
- Git history when needed to distinguish active, superseded and historical
  rules.

## Migration method

Each recovered rule must receive:

- stable contract ID;
- status (ACTIVE / CONDITIONAL / SUPERSEDED / RETIRED / EXPERIMENT_ONLY);
- exact scope;
- normative statement;
- provenance;
- implementation/enforcement mapping;
- supersession relation when relevant.

A result from an experiment is not automatically an invariant. It becomes one
only when the project used it to freeze a decision, admission rule, architecture,
protocol, or operational constraint.

Historical rules must not be copied blindly: the audit must reconcile later
decisions and mark superseded rules explicitly.

## Fail-closed transition rule

Until migration reaches COMPLETE, the contract must advertise
`BOOTSTRAP_AUDIT_REQUIRED`.

The currently running 10115->10315 semantic long run is grandfathered because
changing its scientific contract mid-run would be worse than allowing it to
finish. No new long training or architecture-changing stage should begin until
the affected domains have been audited and their active invariants are covered.

## End state

The contract bundle becomes the first source checked for every substantial
change. ROADMAP and CURRENT_WORK remain chronological/contextual records, not
the only guard against forgetting.

CI/preflight must eventually check:

1. contract syntax/schema;
2. referenced contract IDs exist and are ACTIVE/CONDITIONAL as appropriate;
3. long-run manifests cite current performance contracts;
4. immutable/frozen artifacts and hashes match required provenance;
5. changes touching governed areas declare affected contract IDs;
6. no silent supersession;
7. contract/document/code conflicts fail closed.

The migration is complete only when every required coverage domain in
`PROJECT_CONTRACT.yaml` is audited and there are zero unresolved conflicts.
