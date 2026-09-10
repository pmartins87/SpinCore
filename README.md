# SpinCore

Canonical SpinCore recovery repository for the Spin & Go / All-in-or-Fold expansion project.

> **MANDATORY PRE-FLIGHT FOR ALL FUTURE WORK:** read [`AGENTS.md`](AGENTS.md) and [`docs/LEGACY_BASELINE_AND_QUALITY_POLICY.md`](docs/LEGACY_BASELINE_AND_QUALITY_POLICY.md) before proposing or executing training, architecture changes, action-abstraction changes, sampler changes, evaluator/state changes, or runtime integration. The multi-year archive `Tentativas anteriores de SpinGo.zip` is the historical baseline; SpinCore is an evolution of that work, not a greenfield replacement.

The repository was rebuilt after the original R5 Git bundle and part of the R6/R7 transient checkout were lost with a ChatGPT runtime. The current tree is intentionally self-contained and is **not claimed to be byte-for-byte identical** to the lost R5 checkout. It reimplements the preserved contracts and re-certifies them with clean Release, ASan/UBSan, and Python tests.

Current state: R0-R6 and R7.0-R7.2 physically rebuilt/recertified; later roadmap state is tracked in `ROADMAP.md`. **As of 2026-09-09, single-blind 10/20 R7.5.4 evidence is localized only and must not select the final global SpinGo policy/action abstraction until re-evaluated under the representative legacy tournament-state distribution.**

Permanent invariants include true-HU vs 3H domain separation, exact cloneable hidden state, explicit-payout ICM continuation utility for production Deep CFR, fail-closed ambiguous simultaneous elimination, external-sampling advantage targets, own-reach average-policy collection, full-reservoir deterministic audits, and exact mid-iteration checkpoint/resume. These invariants are subordinate to the mandatory legacy-first quality policy where the older roadmap contains certification-only requirements that do not affect actual playing quality.
