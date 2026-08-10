# R7.3 replicated-candidate V1 deck-control correction — 2026-08-10

`READY FOR TABLES = NO`. No frozen gate is changed.

## Finding

During post-run review of the three 640 replicated-path candidates, the candidate runner's hidden-deal schedule was compared line-by-line against the authoritative generation-2 R7.3 runner.

The authoritative runner (`tools/run_r7_3_diagnostic.py`) uses a **global root index continuous across CFR iterations**:

```text
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
```

with `global_root = 0..639` for a 5 x 128 run.

The replicated-candidate V1 runner instead used:

```text
deck_seed = (seed << 32) ^ (iteration << 16) ^ root_index_within_iteration
```

and `root_index` restarted at zero at every iteration.

Therefore the V1 comments and JSON field that described its deck schedule as matching the acceptance runner were incorrect.

## Scope of the correction

The physical V1 results remain valid **experiments on independent hidden-deal streams**. Their solver execution, neural fits and reported metrics are not discarded:

- separated Advantage x4: `0.459596 / 0.898250`, evidence `94b5e423fa51e1dad8445e6ce36b8832d8161648`;
- separated both x4: `0.458853 / 0.908883`, evidence `871967f777f7cec17479ed3ec9f476543452912d`;
- coupled Advantage x4: `0.451112 / 0.893292`, evidence `87547311076fd6a015b7d855de1a9c26124b924f`.

However, these V1 values **must not be treated as deck-identical paired comparisons** against corrected 640 (`0.477649 / 0.902403`) or strong-Advantage 640 (`0.464474 / 0.886204`). Some of their apparent delta may come from the different deal sample.

This does not reverse the broader conclusion that none of the V1 x4 runs passed the frozen gates. It does change the strength of any claim about the exact percentage improvement caused by x4 at acceptance scale.

## V2 correction

`tools/run_r7_3_replicated_640_candidate_v2.py` now reuses the already-smoke-certified V1 collection/fitting machinery but replaces only the deal schedule with the exact authoritative formula. It also self-checks boundary points across iteration transitions so global-root continuity cannot silently regress.

V2 JSON explicitly records:

```text
GENERATION2_AUTHORITATIVE_GLOBAL_ROOT_FORMULA_EXACT
seed*1000003 + global_root*97 + iteration
global_root_continuous_across_iterations = true
```

Workflow `r7_3_advantage_x4_coupled_640_v2.yml` runs the highest-priority corrected candidate: coupled Advantage x4, 640 roots/seed, 4 Advantage trajectories, 1 strategy trajectory, strong Advantage fitting, enlarged 400k reservoir, and unchanged frozen gates.

## Engineering lesson

R7.3 diagnostics must distinguish three concepts explicitly:

1. **same distribution** of hidden deals;
2. **independent samples** from that distribution;
3. **the exact same deterministic deal schedule** used by a reference run.

Only (3) permits a tightly paired claim about a candidate's delta against the reference. Future acceptance-candidate reports must carry the exact deck formula and global-index semantics in their evidence payload.
