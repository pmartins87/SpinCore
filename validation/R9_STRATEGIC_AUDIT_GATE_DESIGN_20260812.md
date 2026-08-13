# R9 Strategic Audit — finite gate precommit

`READY FOR TABLES = NO`

R9 begins only after R8.5 has frozen the official production policy artifacts for every accepted production profile/domain. R9 audits those immutable artifacts; it does not retrain, retune or select a replacement policy after seeing audit results.

R9 PASS authorizes **R10 OpenHoldem runtime integration only**. It never authorizes table use.

## Finite R9 sequence

### R9.0 — immutable artifact / identity admission

For every policy entering audit, verify fail-closed:

- exact R8 production profile identity;
- domain identity (`TRUE_HEADS_UP` or `THREE_HANDED`);
- ruleset, action-abstraction, utility-model and learning-profile identities;
- exact model bytes and SHA-256;
- architecture/configuration identity;
- training/checkpoint lineage and counters;
- source commit/tree provenance;
- R7.4 prerequisite provenance;
- preservation of any still-open R7.3 exact-reproducibility debt;
- `ready_for_tables = false`.

Any missing or mismatched identity rejects the artifact before strategic testing.

### R9.1 — structural / numerical policy audit

Exercise the frozen artifact on deterministic sampled states covering its accepted production support and prove:

- six-action output shape;
- finite non-negative probabilities;
- zero probability mass on illegal actions;
- normalized legal-action mass;
- no model/runtime exceptions on accepted support;
- no silent HU/3H or production-profile fallback;
- exact deterministic replay under the frozen inference runtime;
- action-abstraction labels/IDs match the artifact identity.

This is a correctness gate, not a poker-quality substitute.

### R9.2 — canonical/extreme action sentinels

Use `SPINCORE_STRATEGIC_ACTION_SENTINELS_V1` plus deterministic catalog states generated from exact `(episode, action_prefix, deck_seed)` identities.

Two layers are mandatory:

1. exact integrity fingerprints for the frozen model/runtime/observation;
2. independently precommitted strategic plausibility bounds.

Integrity-only PASS cannot satisfy R9.2.

The final sentinel population must cover HU and 3H separately, strong and weak holdings, materially different stack depths, and relevant facing-action / payout-ICM situations supported by the exact production profile.

**Anti-tuning rule:** numerical plausibility bounds and the exact sentinel set must be frozen before inspecting the audited production policy outputs on those states. A failed sentinel is investigated; its bound is not weakened merely because the production artifact failed it.

### R9.3 — independent held-out strategic stability

Run an independent, mechanically derived hold-out audit that was not used for R8 training or model selection. It must cover every official profile/domain policy.

Before the physical R9.3 audit begins, freeze in machine-readable form:

- hold-out seed derivation;
- scenario/profile support schedule;
- sample/root counts;
- all quantitative acceptance thresholds;
- inference/runtime identity;
- stop/failure rules.

Thresholds may not be selected after seeing R9.3 results. R9.3 reports both aggregate metrics and per-domain/per-profile results so one strong policy cannot mask another failed policy.

### R9.4 — action-distribution pathology / collapse audit

Audit broad held-out state samples for gross policy pathologies that aggregate fit metrics can miss, including:

- action probability NaN/Inf or illegal mass;
- accidental one-action collapse across heterogeneous states;
- unreachable action IDs caused by runtime/action-map mismatch;
- identical HU/3H outputs caused by wrong artifact routing;
- identical outputs across distinct production profiles caused by fallback/misrouting;
- discontinuities attributable to malformed encoded state fields rather than strategic state changes.

Any quantitative collapse/pathology criterion that is not a pure structural invariant must be precommitted before the audit result is observed.

### R9.5 — strategic value / best-response audit

Evaluate the frozen policy using an independently defined strategic-value / best-response methodology appropriate to the production domain and payout semantics. The exact methodology, opponent model(s), seeds, sampling budget, uncertainty reporting and PASS thresholds must be frozen before execution.

R9.5 may estimate exploitability or best-response loss, but an estimate without uncertainty / sample-budget provenance cannot be used as a release gate. HU and 3H are reported separately.

No R9.5 result may trigger silent retraining inside R9. A failed official policy returns the project to an explicit R8 training/revision cycle with a new artifact identity.

### R9.6 — finite audit summary

Materialize one machine-readable R9 final gate that binds all audited policy hashes and all R9.0–R9.5 evidence.

R9 PASS requires every required production profile/domain artifact to pass every applicable subgate. There is no averaging of failed profiles/domains into an overall PASS.

R9 final output must state:

```text
r9_pass
r9_ready_to_advance_to_r10
all_required_policy_artifacts_audited
strategic_sentinel_gate_pass
heldout_stability_gate_pass
action_pathology_gate_pass
best_response_gate_pass
r7_3_exact_reproducibility_debt_status
ready_for_tables = false
```

## Fail-closed / anti-drift rules

R9 remains FAIL/BLOCKED if any of the following occurs:

- R8.5 official artifact freeze is absent;
- any audit loads model bytes different from the frozen R8.5 hash;
- profile/domain routing falls back to another policy;
- a threshold or seed is changed after the corresponding result is inspected;
- integrity sentinels are substituted for strategic plausibility sentinels;
- HU PASS is used to cover missing/failed 3H evidence or vice versa;
- a failed policy is silently retrained/replaced within the same audit identity;
- a best-response estimate lacks a frozen budget/methodology and uncertainty evidence;
- any R9 output claims table readiness.

## Finite transition

```text
R8.5 frozen official policies
-> R9.0 identity admission
-> R9.1 structural/numerical audit
-> R9.2 canonical/extreme action sentinels
-> R9.3 independent held-out stability
-> R9.4 action-distribution pathology audit
-> R9.5 strategic value / best-response audit
-> R9.6 final R9 gate
-> R10 OpenHoldem runtime integration
```

No additional R9 stage may be silently inserted after results are observed. A genuinely new safety finding can amend this roadmap only through an explicit documented gate-design revision that states why the original finite design was insufficient.
