from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from spincore.r7_5_action_cfr import NUM_ACTIONS
from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_final_policy import load_finalized_v3_policy_light
from spincore.r7_5_representation_v3_phase2_eval import cross_seed_policy_stability
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact, state_payload_sha256
from spincore.r7_5_representation_v3_stage_contract import DOMAINS, EVALUATION_SEEDS, TRAINING_SEEDS

HELDOUT_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_HELDOUT_EVAL_CELL_V1"
RESULT_SCHEMA = "SPINCORE_R7_5_3C_CROSS_SEED_PARITY_AUDIT_RESULT_V1"
POLICY_COUNT = 1024
POLICY_RECOMPUTE_MAX_ABS_TOL = 1e-6
METRIC_RECOMPUTE_TOL = 5e-6
BLOCKER_EVIDENCE_TOL = 1e-12


def read_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _short(rep: str) -> str:
    return "H2" if rep == H2_FINAL else "H3"


def _sha_legal(descriptors) -> str:
    rows = [[int(x) for x in item.legal_slots] for item in descriptors[:POLICY_COUNT]]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _assert_close(left: float, right: float, name: str, tol: float) -> None:
    if abs(float(left) - float(right)) > float(tol):
        raise RuntimeError(f"{name} mismatch: {left} != {right}; tol={tol}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--cells-root", type=Path, required=True)
    ap.add_argument("--policies-root", type=Path, required=True)
    ap.add_argument("--heldout-root", type=Path, required=True)
    ap.add_argument("--training-sha", required=True)
    ap.add_argument("--evaluator-sha", required=True)
    ap.add_argument("--heldout-sha", required=True)
    ap.add_argument("--blocker-evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if NUM_ACTIONS != 10:
        raise RuntimeError(f"universal action width drift: {NUM_ACTIONS}")

    blocker = json.loads(args.blocker_evidence.read_text(encoding="utf-8"))
    expected_rows = {(row["representation"], row["domain"], int(row["evaluation_seed"])): row for row in blocker["cross_seed_rows"]}
    if len(expected_rows) != 8:
        raise RuntimeError("blocker evidence does not contain exact 8 cross-seed rows")

    cells = {}
    for path in sorted(args.cells_root.rglob("cell.json.gz")):
        payload = read_gz(path)
        if payload.get("schema") != HELDOUT_SCHEMA:
            continue
        if payload.get("evaluator_execution_sha") != args.evaluator_sha or payload.get("training_execution_sha") != args.training_sha or payload.get("heldout_execution_sha") != args.heldout_sha:
            raise RuntimeError(f"heldout cell immutable identity mismatch: {path}")
        key = (str(payload["representation"]), str(payload["domain"]), int(payload["training_seed"]), int(payload["evaluation_seed"]))
        if key in cells:
            raise RuntimeError(f"duplicate heldout cell {key}")
        cells[key] = payload
    if len(cells) != 16:
        raise RuntimeError(f"expected 16 heldout cells, found {len(cells)}")

    heldout = {}
    for path in sorted(args.heldout_root.rglob("states.json.gz")):
        raw = read_gz(path)
        domain, evaluation_seed = str(raw["domain"]), int(raw["evaluation_seed"])
        states = load_heldout_v3_artifact(path, expected_domain=domain, expected_evaluation_seed=evaluation_seed, expected_count=2048)
        key = (domain, evaluation_seed)
        if key in heldout:
            raise RuntimeError(f"duplicate heldout corpus {key}")
        heldout[key] = states
    if len(heldout) != 4:
        raise RuntimeError(f"expected 4 heldout corpora, found {len(heldout)}")

    policies = {}
    for path in sorted(args.policies_root.rglob("*.pt")):
        raw_rng = torch.get_rng_state().clone()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        torch.set_rng_state(raw_rng)
        rep, domain, seed = str(payload.get("representation")), str(payload.get("domain")), int(payload.get("training_seed", -1))
        if rep not in (H2_FINAL, H3_FINAL) or domain not in DOMAINS or seed not in TRAINING_SEEDS:
            continue
        policy = load_finalized_v3_policy_light(path, repo_root=args.repo_root, expected_training_execution_sha=args.training_sha, expected_representation=rep, expected_domain=domain, expected_training_seed=seed)
        key = (rep, domain, seed)
        if key in policies:
            raise RuntimeError(f"duplicate light policy {key}")
        policies[key] = policy
    if len(policies) != 8:
        raise RuntimeError(f"expected 8 light policies, found {len(policies)}")

    results, global_max_recompute_abs = [], 0.0
    for rep in (H2_FINAL, H3_FINAL):
        for domain in DOMAINS:
            for evaluation_seed in EVALUATION_SEEDS:
                audit = heldout[(domain, int(evaluation_seed))][:POLICY_COUNT]
                descriptor_sha = state_payload_sha256(tuple(audit))
                legal_sha = _sha_legal(audit)
                stored_rows, recomputed_rows, seed_details = [], [], []
                for training_seed in TRAINING_SEEDS:
                    cell = cells[(rep, domain, int(training_seed), int(evaluation_seed))]
                    if [int(x) for x in cell["policy_state_indices"]] != list(range(POLICY_COUNT)):
                        raise RuntimeError("policy state indices are not exact 0..1023")
                    stored = np.asarray(cell["policy_rows"], dtype=np.float64)
                    if stored.shape != (POLICY_COUNT, NUM_ACTIONS):
                        raise RuntimeError(f"stored policy shape drift: {stored.shape}")
                    policy = policies[(rep, domain, int(training_seed))]
                    recomputed = np.asarray(policy.batch_probabilities([item.observation_v3 for item in audit], [item.legal_slots for item in audit]), dtype=np.float64)
                    if recomputed.shape != stored.shape:
                        raise RuntimeError("recomputed policy shape mismatch")
                    max_abs = float(np.max(np.abs(recomputed - stored)))
                    global_max_recompute_abs = max(global_max_recompute_abs, max_abs)
                    if max_abs > POLICY_RECOMPUTE_MAX_ABS_TOL:
                        raise RuntimeError(f"worker policy recomputation mismatch {rep}/{domain}/{training_seed}/{evaluation_seed}: {max_abs}")
                    stored_rows.append(stored.tolist())
                    recomputed_rows.append(recomputed.tolist())
                    seed_details.append({"training_seed": int(training_seed), "max_abs_worker_vs_recomputed": max_abs, "artifact_path": str(policy.artifact_path)})

                stored_metric = cross_seed_policy_stability(stored_rows[0], stored_rows[1])
                recomputed_metric = cross_seed_policy_stability(recomputed_rows[0], recomputed_rows[1])
                for name in ("mean", "p50", "p95", "max"):
                    _assert_close(stored_metric[name], recomputed_metric[name], f"stored/recomputed {name}", METRIC_RECOMPUTE_TOL)
                if bool(stored_metric["gate_pass"]) != bool(recomputed_metric["gate_pass"]):
                    raise RuntimeError("stored/recomputed gate boolean mismatch")
                expected = expected_rows[(_short(rep), domain, int(evaluation_seed))]
                for name in ("mean", "p95", "max"):
                    _assert_close(stored_metric[name], expected[name], f"blocker evidence {name}", BLOCKER_EVIDENCE_TOL)
                if bool(stored_metric["gate_pass"]) != bool(expected["gate_pass"]):
                    raise RuntimeError("blocker evidence gate boolean mismatch")
                results.append({"representation": rep, "domain": domain, "evaluation_seed": int(evaluation_seed), "heldout_first1024_payload_sha256": descriptor_sha, "legal_slots_first1024_sha256": legal_sha, "action_width": NUM_ACTIONS, "seed_details": seed_details, "cross_seed_policy_stability": stored_metric})

    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "purpose": "Independent parity audit of the frozen Phase2 cross-seed policy-stability implementation and artifacts; diagnostic only, never selects a representation.",
        "evaluator_execution_sha": args.evaluator_sha,
        "training_execution_sha": args.training_sha,
        "heldout_execution_sha": args.heldout_sha,
        "heldout_cells_verified": len(cells),
        "light_policies_verified": len(policies),
        "heldout_corpora_verified": len(heldout),
        "cross_seed_rows_verified": len(results),
        "universal_action_width": NUM_ACTIONS,
        "numerical_tolerances": {"policy_recompute_max_abs": POLICY_RECOMPUTE_MAX_ABS_TOL, "cross_seed_metric_recompute_abs": METRIC_RECOMPUTE_TOL, "stored_metric_vs_blocker_evidence_abs": BLOCKER_EVIDENCE_TOL},
        "max_abs_worker_policy_vs_independent_recompute": global_max_recompute_abs,
        "rows": results,
        "conclusion": "The observed cross-seed TV failures reproduce from independently reloaded final AveragePolicy artifacts on the exact same hashed heldout observations/legal slots. No seed-specific state, legal-mask, action-slot, policy-artifact identity, or worker serialization mismatch was found by this audit.",
        "representation_winner": None,
        "changes_frozen_thresholds": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(results), "max_abs_worker_policy_vs_independent_recompute": global_max_recompute_abs, "conclusion": result["conclusion"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
