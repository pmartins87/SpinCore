# SpinCore

Canonical SpinCore recovery repository for the Spin & Go / All-in-or-Fold expansion project.

The repository was rebuilt after the original R5 Git bundle and part of the R6/R7 transient checkout were lost with a ChatGPT runtime. The current tree is intentionally self-contained and is **not claimed to be byte-for-byte identical** to the lost R5 checkout. It reimplements the preserved contracts and re-certifies them with clean Release, ASan/UBSan, and Python tests.

Current state: R0-R6 and R7.0-R7.2 physically rebuilt/recertified; R7.3 multi-seed stability remains the active failing gate. `READY FOR TABLES = NO`.

Permanent invariants include true-HU vs 3H domain separation, exact cloneable hidden state, explicit-payout ICM continuation utility for production Deep CFR, fail-closed ambiguous simultaneous elimination, external-sampling advantage targets, own-reach average-policy collection, full-reservoir deterministic audits, and exact mid-iteration checkpoint/resume.
