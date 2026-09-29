# SpinCore

Canonical SpinCore recovery repository for the Spin & Go / All-in-or-Fold expansion project.

> **MANDATORY PRE-FLIGHT FOR ALL FUTURE WORK:** read [`PROJECT_CONTRACT.yaml`](PROJECT_CONTRACT.yaml), the relevant machine-readable modules under `contracts/`, [`AGENTS.md`](AGENTS.md), and [`docs/LEGACY_BASELINE_AND_QUALITY_POLICY.md`](docs/LEGACY_BASELINE_AND_QUALITY_POLICY.md) before proposing or executing training, architecture changes, action-abstraction changes, sampler changes, evaluator/state changes, or runtime integration. `PROJECT_CONTRACT.yaml` is the first normative source for active invariants. The multi-year archive `Tentativas anteriores de SpinGo.zip` is the historical baseline; SpinCore is an evolution of that work, not a greenfield replacement.

The repository was rebuilt after the original R5 Git bundle and part of the R6/R7 transient checkout were lost with a ChatGPT runtime. The current tree is intentionally self-contained and is **not claimed to be byte-for-byte identical** to the lost R5 checkout. It reimplements the preserved contracts and re-certifies them with clean Release, ASan/UBSan, and Python tests.

Current state: R0-R6 and R7.0-R7.2 physically rebuilt/recertified; later roadmap state is tracked in `ROADMAP.md`. **As of 2026-09-09, single-blind 10/20 R7.5.4 evidence is localized only and must not select the final global SpinGo policy/action abstraction until re-evaluated under the representative legacy tournament-state distribution.**

Permanent/current invariants are represented and scope-resolved in `PROJECT_CONTRACT.yaml` and its modules. Historical documents remain evidence/provenance; they do not silently override the active contract.


## Project contract

As of 2026-09-29 the exhaustive migration is complete. The machine-readable
logical contract rooted at `PROJECT_CONTRACT.yaml` is the first normative
source for active project invariants and has status `COMPLETE`.

The closure is auditable through `contracts/AUDIT_STATUS.yaml`,
`contracts/SOURCE_AUDIT_INDEX.json`, `contracts/SOURCE_AUDIT_OVERRIDES.yaml`,
`contracts/SOURCE_FAMILY_ADJUDICATIONS.yaml`, and
`contracts/ENFORCEMENT_MAP.yaml`. Contract CI fails on missing scopes,
provenance/enforcement bindings, unaudited new source material, or illegal
COMPLETE-state drift.

The already-running 10115->10315 semantic long training remains the explicit
grandfathered execution. It may finish untouched, but its sequential fitter is
not reusable for a restart/extension without the current performance contract.
