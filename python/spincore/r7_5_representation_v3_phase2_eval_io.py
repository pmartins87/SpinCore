from __future__ import annotations

import gzip
import json
from pathlib import Path

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_phase2_eval import H2, H3, cross_seed_policy_stability, validate_training_final_report
from spincore.r7_5_representation_v3_stage_contract import DOMAINS, EVALUATION_SEEDS, TRAINING_SEEDS

HELDOUT_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_HELDOUT_EVAL_CELL_V1"
COMMONREF_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_COMMONREF_CELL_V1"
PAIRWISE_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_PAIRWISE_CELL_V1"
REPRESENTATIONS = (H2_FINAL, H3_FINAL)
REP_SHORT = {H2_FINAL: H2, H3_FINAL: H3}


def read_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def load_cells(root: Path, *, evaluator_sha: str, training_sha: str) -> tuple[list[dict], list[dict], list[dict]]:
    heldout, commonref, pairwise = [], [], []
    for path in sorted(root.rglob("*.json.gz")):
        payload = read_gz(path)
        schema = payload.get("schema")
        if schema not in (HELDOUT_SCHEMA, COMMONREF_SCHEMA, PAIRWISE_SCHEMA):
            continue
        if payload.get("evaluator_execution_sha") != evaluator_sha:
            raise RuntimeError(f"evaluator execution SHA mismatch in {path}")
        if payload.get("training_execution_sha") != training_sha:
            raise RuntimeError(f"training execution SHA mismatch in {path}")
        if payload.get("production_training_authorized") or payload.get("ready_for_tables"):
            raise RuntimeError(f"evaluation artifact illegally authorizes production/table use: {path}")
        {HELDOUT_SCHEMA: heldout, COMMONREF_SCHEMA: commonref, PAIRWISE_SCHEMA: pairwise}[schema].append(payload)
    return heldout, commonref, pairwise


def require_exact_inventory(heldout: list[dict], commonref: list[dict], pairwise: list[dict]) -> None:
    expected = {(r, d, int(ts), int(es)) for r in REPRESENTATIONS for d in DOMAINS for ts in TRAINING_SEEDS for es in EVALUATION_SEEDS}
    got = {(p["representation"], p["domain"], int(p["training_seed"]), int(p["evaluation_seed"])) for p in heldout}
    if got != expected or len(heldout) != len(expected):
        raise RuntimeError(f"heldout evaluation inventory mismatch: missing={sorted(expected-got)} extra={sorted(got-expected)} rows={len(heldout)}")
    got = {(p["representation"], p["domain"], int(p["training_seed"]), int(p["evaluation_seed"])) for p in commonref}
    if got != expected or len(commonref) != len(expected):
        raise RuntimeError(f"common-reference inventory mismatch: missing={sorted(expected-got)} extra={sorted(got-expected)} rows={len(commonref)}")
    expected_pairwise = {(d, int(es), int(h2s), int(h3s)) for d in DOMAINS for es in EVALUATION_SEEDS for h2s in TRAINING_SEEDS for h3s in TRAINING_SEEDS}
    got_pairwise = {(p["domain"], int(p["evaluation_seed"]), int(p["h2_training_seed"]), int(p["h3_training_seed"])) for p in pairwise}
    if got_pairwise != expected_pairwise or len(pairwise) != len(expected_pairwise):
        raise RuntimeError(f"pairwise inventory mismatch: missing={sorted(expected_pairwise-got_pairwise)} extra={sorted(got_pairwise-expected_pairwise)} rows={len(pairwise)}")


def recompute_hard_gates(heldout: list[dict]) -> tuple[dict, dict[str, bool]]:
    by_key = {(p["representation"], p["domain"], int(p["training_seed"]), int(p["evaluation_seed"])): p for p in heldout}
    final_gates, sentinel_rows, cross_seed_rows = {}, [], []
    for representation in REPRESENTATIONS:
        for domain in DOMAINS:
            for training_seed in TRAINING_SEEDS:
                reports = [by_key[(representation, domain, int(training_seed), int(es))]["final_report"] for es in EVALUATION_SEEDS]
                if json.dumps(reports[0], sort_keys=True) != json.dumps(reports[1], sort_keys=True):
                    raise RuntimeError("final training report drifted across evaluation-seed artifacts")
                final_gates[f"{representation}|{domain}|{training_seed}"] = validate_training_final_report(reports[0])
            for evaluation_seed in EVALUATION_SEEDS:
                rows = [by_key[(representation, domain, int(seed), int(evaluation_seed))] for seed in TRAINING_SEEDS]
                identities = [tuple(int(x) for x in row["policy_state_indices"]) for row in rows]
                if identities[0] != identities[1] or identities[0] != tuple(range(1024)):
                    raise RuntimeError("cross-seed heldout state identity mismatch")
                cross_seed_rows.append({
                    "representation": representation,
                    "domain": domain,
                    "evaluation_seed": int(evaluation_seed),
                    **cross_seed_policy_stability(rows[0]["policy_rows"], rows[1]["policy_rows"]),
                })
                for row in rows:
                    sentinel_rows.append({
                        "representation": representation,
                        "domain": domain,
                        "training_seed": int(row["training_seed"]),
                        "evaluation_seed": int(evaluation_seed),
                        **dict(row["sentinel_gate"]),
                    })
    candidate_pass = {}
    for representation in REPRESENTATIONS:
        report_pass = all(gate["gate_pass"] for key, gate in final_gates.items() if key.startswith(representation + "|"))
        sentinel_pass = all(row.get("gate_pass") for row in sentinel_rows if row["representation"] == representation)
        cross_seed_pass = all(row.get("gate_pass") for row in cross_seed_rows if row["representation"] == representation)
        candidate_pass[REP_SHORT[representation]] = bool(report_pass and sentinel_pass and cross_seed_pass)
    return {
        "final_report_gates": final_gates,
        "sentinel_gates": sentinel_rows,
        "cross_seed_policy_stability": cross_seed_rows,
        "candidate_pass": candidate_pass,
    }, candidate_pass


def validate_training_inventory(path: Path, training_sha: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "SPINCORE_R7_5_3C_PHASE2_TRAINING_INVENTORY_V1":
        raise RuntimeError("wrong Phase2 training inventory schema")
    if payload.get("execution_sha") != training_sha:
        raise RuntimeError("Phase2 training inventory SHA mismatch")
    if int(payload.get("expected_cells", -1)) != 8 or int(payload.get("observed_cells", -1)) != 8:
        raise RuntimeError("Phase2 training inventory is incomplete")
    if payload.get("integrity_complete") is not True:
        raise RuntimeError("Phase2 training inventory integrity is not complete")
    return payload
