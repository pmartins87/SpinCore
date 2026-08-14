# Artifact persistence ordering incident — 2026-08-13

Status: **ROOT CAUSE IDENTIFIED / SCIENTIFIC EVIDENCE UNAFFECTED / ENGINEERING RULE FROZEN**

`READY FOR TABLES = NO`.

## Incident

R7.5.4 structural audit run `31769178705` computed and uploaded valid immutable evidence at execution SHA:

```text
95f885048fdc1ce1b8468ec5fdddf79a8522d1ea
```

The compute/audit job succeeded and artifact `9207470459` was uploaded with artifact digest:

```text
sha256:c8125bd4d7aef25a2528048666ca1ffb0e0cddfd305ee9414dbc486dbb03337a
```

The subsequent persistence job failed for a purely mechanical reason:

```text
1. actions/download-artifact wrote the immutable artifact under computed/
2. actions/checkout@v4 then checked out main with its default clean behavior
3. checkout deleted computed/
4. cp failed because the already-downloaded evidence no longer existed in the workspace
```

Observed error:

```text
cp: cannot stat 'computed/R7_5_4_ACTION_STRUCTURAL_AUDIT.json': No such file or directory
```

The solver, structural audit, selection rules and evidence bytes were not recomputed or modified by this failure.

## Recovery

The immutable artifact was downloaded independently and inspected. Its exact logical evidence was persisted to main without rerunning the scientific computation:

```text
validation/R7_5_4_ACTION_STRUCTURAL_AUDIT.json
validation/R7_5_4_ACTION_STRUCTURAL_EXECUTION_MANIFEST.json
```

The durable evidence retains:

```text
workflow_run_id = 31769178705
execution_sha   = 95f885048fdc1ce1b8468ec5fdddf79a8522d1ea
structural_gate_pass = true
production_training_authorized = false
ready_for_tables = false
```

## Frozen engineering rule

For every workflow that persists an artifact into Git:

```text
CORRECT:
    checkout destination branch
    THEN download immutable artifact
    THEN validate provenance
    THEN copy/add/commit/push

FORBIDDEN:
    download artifact into workspace
    THEN actions/checkout with clean=true
    THEN attempt to persist downloaded artifact
```

Equivalent safe alternatives are allowed only when the artifact is downloaded outside the checkout-cleaned workspace and its provenance is validated after checkout.

## Active R7.5.3 protection

The already-running R7.5.3 workflow `31767822186` loaded its YAML before this incident was discovered and therefore cannot be safely mutated in-flight.

A separate persistence-only recovery workflow has been installed:

```text
.github/workflows/r7_5_3_persist_recovery_current_run.yml
```

It is hard-bound to:

```text
source run    = 31767822186
execution SHA = 66923cf50ece3aa9ac5632a11bcd865eb154f3e4
```

It performs no solver work and no aggregation. On completion of that source run it:

1. checks out current main first;
2. looks specifically for the final immutable `r7-5-3-evidence-<execution_sha>` artifact;
3. does nothing if no final artifact exists;
4. if it exists, downloads it after checkout;
5. verifies manifest run id, execution SHA, selected candidate and all `ready_for_tables=false` / production-training guards;
6. persists the exact computed evidence.

Therefore an infrastructure-only persistence failure cannot silently erase a completed R7.5.3 result or cause a scientific rerun.

## Gate meaning

This incident does not change any strategic gate, seed, sample, candidate, tolerance or result.

```text
R7.5.4 structural compute       PASS
R7.5.4 structural evidence      DURABLE
R7.5.4 strategic action gate    NOT RUN
R7.5.3 representation gate      ACTIVE
READY FOR TABLES                NO
```
