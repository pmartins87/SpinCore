# R12 Operational Homologation — finite final gate precommit

R12 is the final roadmap stage. It begins only after R11 PASS. R12 does not invent or retune strategy; it proves that the complete approved system can be deployed, observed, recovered and operated on the intended machines without violating the audited strategic/runtime contracts.

**Only R12.9 may set `READY FOR TABLES = YES`.** Every earlier stage and artifact must remain `ready_for_tables = false`.

## R12.0 — immutable release candidate

Build one release candidate whose manifest binds at minimum:

- source commit/tree;
- OpenHoldem executable/version identity accepted by R10;
- bridge/user-DLL bytes and SHA-256;
- all baseline policy model hashes;
- all enabled exploit model/configuration hashes;
- production profile manifest/hash;
- runtime dependency versions;
- action/observation schema identities;
- opponent DB/schema/migration identity where exploitation uses it;
- table/scrape configuration identity;
- startup/configuration files;
- R7–R11 evidence/gate hashes;
- explicit unresolved-debt list;
- target machine/architecture requirements.

After R12.0, homologation tests run against those exact release-candidate bytes. Any code/model/config change creates a new release-candidate identity and invalidates downstream R12 evidence that depends on changed bytes.

## R12.1 — installation / deployment parity

Prove a deterministic installation/update procedure for every intended playing PC. At minimum verify:

- expected files all present and no stale alternative DLL/model is loadable by accident;
- file hashes match the release manifest;
- required runtime dependencies are present at accepted versions;
- local configuration points to the exact production profile/model manifest;
- DB/schema migrations are complete and idempotent;
- machine-specific paths/settings do not change strategic identity;
- restart/reboot preserves the intended configuration;
- an incomplete/corrupt deployment fails closed before play.

If multiple PCs are used, parity is proved per PC. Copying only a subset of binaries is not assumed safe unless the manifest explicitly proves those are the only changed components.

## R12.2 — scrape/table-state homologation

On the exact target client/table configuration, validate the R10 canonical snapshot against independently observable game state across representative hands and failure fixtures.

Cover at minimum:

- hero seat and opponent seats;
- HU transition from 3H and normal 3H;
- hole/public cards;
- button/dealer identity;
- live/dead/eliminated players;
- stacks, pot, bets, amount to call;
- blind/ante/level/profile/multiplier identity;
- fold/check/call/raise/all-in sequences;
- simultaneous elimination/tie situations relevant to the ruleset;
- hand-boundary/reset behavior;
- observer/not-seated state where applicable.

Any ambiguous or stale scrape must trigger the R10 safety barrier, not a strategic action.

## R12.3 — historical log / deterministic replay

Replay a substantial corpus of captured real or homologation hands through the release candidate offline. For every decision where source state is sufficiently complete, compare:

```text
captured state
-> reconstructed canonical snapshot
-> observation hash
-> routed profile/domain/model
-> policy output
-> selected abstract action
-> translated runtime action
```

Repeated replay of the same exact input/runtime/RNG provenance must reproduce the same decision record. Disagreements are classified and resolved before progression.

## R12.4 — observer / shadow homologation

Run the exact release candidate against the live target environment with action emission disabled or otherwise prevented from controlling play. Record the full R10/R11 decision audit stream and verify:

- state resets correctly across many hands/tables/sessions;
- profile/domain routing remains correct;
- no stale opponent/hand state leaks into the next hand;
- baseline/exploit activation semantics match offline expectations;
- safety barriers fire on deliberately unsupported/invalid states;
- latency remains inside the precommitted operational budget;
- process/resource behavior is stable for long sessions.

Shadow evidence cannot be replaced by unit tests alone.

## R12.5 — fault, recovery and rollback homologation

Inject operational failures representative of the actual deployment:

- OpenHoldem restart;
- bridge/DLL restart/reload;
- client/table close/reopen;
- network/client interruption affecting scrape freshness;
- opponent DB temporarily unavailable/locked/corrupt copy;
- model/profile manifest unavailable or hash-mismatched;
- partial file update;
- abrupt process termination during persistence/update;
- machine reboot;
- clock/session/log rotation boundaries;
- transfer/update between intended PCs where applicable.

For each fault prove:

1. no unverified strategic action is emitted during invalid state;
2. durable data remain internally consistent or fail closed;
3. restart restores only a manifest-valid configuration;
4. rollback to the last known-good release is deterministic and documented;
5. opponent/exploitation state cannot silently contaminate another identity/profile/domain.

## R12.6 — endurance / resource / latency gate

Run long-session soak tests on every target machine using the release candidate. Precommit before the authoritative soak:

- minimum duration/decision count;
- maximum acceptable decision latency and timeout behavior;
- memory/resource growth criteria;
- log/audit durability expectations;
- allowed runtime error/barrier categories and counts;
- restart/recovery acceptance rules.

Measure tail latency, not only average latency. A timeout must resolve through the safety barrier rather than a different strategic action.

## R12.7 — end-to-end action verification

Using exact homologation fixtures, compare intended abstract action to the action actually presented/emitted by the OpenHoldem integration, including chip amount and all-in semantics. Include edge cases around:

- check versus fold availability;
- call amount equal/near stack;
- minimum raise;
- stack-limited raises;
- integer/chip rounding;
- all-in abstraction;
- HU/3H transitions;
- hand reset.

No discrepancy is waived because the action is "close enough" strategically. Runtime translation is exact-contract work.

## R12.8 — final evidence/debt closure audit

Before table authorization, materialize and inspect the complete release-debt register. It must explicitly include every debt deferred from earlier stages.

In particular, the R7.3 historical exact-reproducibility debt may **not** remain merely deferred at READY FOR TABLES. R12.8 must contain an explicit accepted resolution that satisfies the release standard established when the debt was deferred. The debt cannot be closed by loosening tolerance, changing frozen strategic gates, changing seeds after results, or relabeling provisional evidence as strict exact certification.

Also require:

- R7.4 final PASS;
- R8 final PASS and immutable official policy freeze;
- R9 strategic audit PASS including populated strategic sentinels;
- R10 OpenHoldem runtime PASS;
- R11 safe exploitation PASS for every exploit feature enabled in the release; features not passing R11 must be disabled and baseline-only operation re-homologated under the exact release manifest;
- no unresolved critical/high safety defect;
- every target PC/install hash matches the release candidate.

## R12.9 — READY FOR TABLES gate

Create one machine-readable final release gate. It is the **only** artifact allowed to emit `ready_for_tables = true`.

Required fields include:

```text
schema = SPINCORE_R12_TABLE_READINESS_GATE_V1
r12_pass
ready_for_tables
release_candidate_id
source_commit_sha
openholdem_runtime_identity
bridge_binary_sha256
production_profile_manifest_sha256
baseline_policy_hashes
exploit_feature_set_and_hashes
r7_4_final_pass
r8_final_pass
r9_final_pass
r10_final_pass
r11_final_pass_or_explicitly_disabled_exploitation
r7_3_exact_reproducibility_debt_closed
installation_parity_pass
scrape_state_homologation_pass
deterministic_replay_pass
shadow_homologation_pass
fault_recovery_rollback_pass
endurance_latency_pass
end_to_end_action_verification_pass
all_release_debts_closed
```

`ready_for_tables = true` iff all mandatory conditions are true for the exact immutable release candidate. Otherwise it is false.

## Anti-shortcut rules

R12 cannot PASS if:

- evidence was produced by binaries/models different from the final release candidate;
- any target PC has unknown/stale component hashes;
- a scrape anomaly can still yield a strategic action;
- replay is not deterministic under the frozen execution/RNG contract;
- runtime action differs from intended abstract action;
- a required shadow/soak/fault test was replaced by reasoning alone;
- exploitation is enabled without R11 PASS;
- an earlier release debt remains deferred rather than resolved;
- the R7.3 exact-reproducibility debt remains open;
- a gate was weakened after observing a failing homologation result.

## Finite endpoint

```text
R11.9 PASS
-> R12.0 immutable release candidate
-> R12.1 installation/deployment parity
-> R12.2 scrape/table-state homologation
-> R12.3 deterministic historical replay
-> R12.4 live observer/shadow homologation
-> R12.5 fault/recovery/rollback homologation
-> R12.6 endurance/resource/latency gate
-> R12.7 end-to-end action verification
-> R12.8 all debts closed, including R7.3 exact reproducibility
-> R12.9 final release gate
-> READY FOR TABLES = YES only if R12.9 PASS
```

This is the finite endpoint of the current SpinCore roadmap. A failed R12 subgate returns to the stage responsible for the defect; R12 does not silently redefine acceptance criteria.
