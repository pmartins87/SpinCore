from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from spincore.solver import SolverLibrary
from spincore.r7 import stratified_audit_indices

from run_r7_3_advantage_fit_sign_sensitivity import (
    DEFAULT_DECK_STREAM_SEED,
    _pair_metrics,
    _predict,
    collect_common_memory,
    train_replica,
)


INIT_A = 0x11111
INIT_B = 0x22222
BATCH_X = 0xAAAA1
BATCH_Y = 0xBBBB2
SPECS = {
    "A_X": (INIT_A, BATCH_X),
    "A_Y": (INIT_A, BATCH_Y),
    "B_X": (INIT_B, BATCH_X),
    "B_Y": (INIT_B, BATCH_Y),
}


def _avg(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Factor Advantage fit variance into init and minibatch components")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_FIT_FACTORIAL_256.json"))
    ap.add_argument("--roots", type=int, default=256)
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_DECK_STREAM_SEED)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--chunk-steps", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=4096)
    ap.add_argument("--fit-target", type=float, default=0.50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--eval-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    args._solver_obj = solver
    started = time.time()
    memory, collection = collect_common_memory(
        solver=solver,
        roots=int(args.roots),
        deck_stream_seed=int(args.deck_stream_seed),
        reservoir_capacity=int(args.reservoir_capacity),
        device=args.device,
    )
    state = memory.state_dict()
    ids = stratified_audit_indices(len(memory.items), int(args.eval_size), 0xFAC70)
    samples = [memory.items[i] for i in ids]
    observations = [x.observation for x in samples]
    legal_masks = [tuple(int(v) for v in x.legal) for x in samples]

    replicas = {}
    predictions = {}
    for name, (init_seed, batch_seed) in SPECS.items():
        bundle, report = train_replica(
            memory_state=state,
            init_seed=int(init_seed),
            batch_seed=int(batch_seed),
            args=args,
        )
        replicas[name] = report
        predictions[name] = _predict(bundle.advantage, observations, args.device)

    pairs = {}
    for a, b in (("A_X", "A_Y"), ("B_X", "B_Y"), ("A_X", "B_X"), ("A_Y", "B_Y"), ("A_X", "B_Y"), ("A_Y", "B_X")):
        pairs[f"{a}_vs_{b}"] = _pair_metrics(predictions[a], predictions[b], legal_masks)

    same_init = [pairs["A_X_vs_A_Y"], pairs["B_X_vs_B_Y"]]
    same_batch = [pairs["A_X_vs_B_X"], pairs["A_Y_vs_B_Y"]]
    both_different = [pairs["A_X_vs_B_Y"], pairs["A_Y_vs_B_X"]]
    init_component = _avg(same_batch, "regret_matching_mean_tv")
    batch_component = _avg(same_init, "regret_matching_mean_tv")
    both = _avg(both_different, "regret_matching_mean_tv")

    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_FIT_FACTORIAL_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "same_memory_for_all_replicas": True,
        "replica_specs": {name: {"init_seed": init, "batch_seed": batch} for name, (init, batch) in SPECS.items()},
        "replicas": replicas,
        "pairwise": pairs,
        "summary": {
            "same_init_different_batch_mean_tv": float(batch_component),
            "different_init_same_batch_mean_tv": float(init_component),
            "different_init_and_batch_mean_tv": float(both),
            "init_to_batch_component_ratio": float(init_component / max(batch_component, 1e-12)),
            "diagnosis": (
                "ADVANTAGE_INIT_VARIANCE_DOMINANT"
                if init_component >= batch_component * 1.25
                else "ADVANTAGE_MINIBATCH_VARIANCE_DOMINANT"
                if batch_component >= init_component * 1.25
                else "ADVANTAGE_INIT_AND_MINIBATCH_VARIANCE_MIXED"
            ),
        },
        "interpretation_note": (
            "Exact 2x2 factorial on one frozen Advantage reservoir. A_X vs A_Y and B_X vs B_Y "
            "change only minibatch RNG; A_X vs B_X and A_Y vs B_Y change only initialization; "
            "cross pairs change both. All policy disagreement is measured after production hard "
            "regret matching. No production training or acceptance gate is changed."
        ),
        "acceptance_gate_changed": False,
        "production_training_rng_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
