# R7.4 active-run checkout integrity audit — 2026-08-12

## Scope

This audit records the repository-integrity risk discovered while GitHub Actions run `31661899987` (`SpinCore R7.4 three-handed 640 confirmation`) is active.

The run was created at head SHA:

```text
92e783344e81e3af4cefee8a30598c554b6b03fd
```

The workflow currently performs `actions/checkout@v4` with `ref: main` separately in each staged job. Therefore later stages can in principle observe a newer `main` than earlier stages. This is an orchestration/provenance weakness even when the frozen strategy source itself is separately identified.

## Active-run semantic audit

At the time this audit was performed, `main` had advanced beyond the run head, but every critical file used by the staged R7.4 3H640 execution remained byte-identical between the run head and the then-current `main`:

```text
tools/r7_4_staged_domain_worker.py
  blob = 4d0dfdafba70506b19ca3ed62041a559ab4ba286

tools/r7_4_stability_pilot_worker.py
  blob = 0a4789f2972e6c659dee50d47b9e97e6ab7e20fa

tools/run_with_heartbeat.py
  blob = d033a52b84ad119e6c1527b866cabb664e087e4d

validation/R7_4_RULESET_EXTENSION_FREEZE.json
  blob = 2500144889204c27adb4da5dcb7eaf427c44a6ec

validation/R7_3_CANDIDATE_SEMANTIC_FREEZE.json
  blob = c27dcdbe6f255aac66fc8cd87ee7f10812bb52f1
```

No semantic contamination of the active R7.4 run was therefore established by the branch movement observed up to this audit.

This finding is not a blanket waiver for later stages. Because the active workflow definition still checks out moving `main`, critical blob identity must be rechecked before accepting any later stage that starts after additional branch movement.

## Required hardening for future executions

A future revision of the staged R7.4 workflow must freeze one immutable execution checkout SHA after the gate has read the required latest durable evidence, then make every compute/finalize/aggregate job check out that exact SHA rather than `main`.

The hardening must satisfy all of the following:

- the gate may read the latest authorized `main` state needed to observe upstream durable evidence;
- immediately after gate authorization, one exact repository SHA is captured and exported as a job output;
- every stage, finalize job, and aggregate computation uses that immutable SHA;
- evidence persistence may rebase/push onto current `main` only after the strategic calculation has completed;
- no seed, strategic threshold, RNG contract, strategy parameter, sample count, or accepted source identity changes;
- the change is orchestration/provenance-only and must be regression-checked before use.

Do not alter the workflow definition merely to affect the already-running experiment. The currently active run must finish under its loaded workflow graph, with stage-by-stage critical-blob verification as needed.

## Release implication

This audit does not promote any strategic gate. In particular:

```text
R7.4 FINAL PASS = false
READY FOR TABLES = false
```

The checkout weakness is recorded for correction, while the active computation remains acceptable only so long as no critical execution-input drift is demonstrated.