from __future__ import annotations

"""Read-only Phase 1B projection decomposition for final x16 reservoirs.

No training, traversal, optimizer step, reservoir mutation, model mutation, or
selection is performed. The purpose is to separate card/current-state support
from public-history support before designing the first V1+ causal ablation.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_stage_contract import DOMAINS, MODEL_FINGERPRINTS, TRAINING_SEEDS
from spincore_nn.codec_v3 import decode_spnniv3

SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_PHASE1B_PROJECTION_DECOMPOSITION_V1"
REPRESENTATIONS = (H2_FINAL, H3_FINAL)
MEMORIES = (("ADVANTAGE", "adv_mem"), ("STRATEGY", "pol_mem"))
PROJECTION_NAMES = (
    "exact_observation",
    "cards_only",
    "geometry_only",
    "fixed_state_no_history",
    "history_exact",
    "history_structured",
    "history_v1_like",
    "no_cards_plus_exact_history",
    "no_cards_plus_structured_history",
    "no_cards_plus_v1_history",
)


def _hash_material(material) -> bytes:
    payload = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _history_exact(decoded):
    return tuple((tuple(int(v) for v in event.categorical), tuple(float(v) for v in event.numeric)) for event in decoded.history)


def _history_structured(decoded):
    return tuple(tuple(int(v) for v in event.categorical) for event in decoded.history)


def _history_v1_like(decoded):
    return tuple((int(event.categorical[1]), int(event.categorical[2])) for event in decoded.history[-32:])


def _projection_digests(observation: bytes) -> dict[str, bytes]:
    decoded = decode_spnniv3(observation)
    cards = (tuple(int(v) for v in decoded.rank_tokens), tuple(int(v) for v in decoded.same_suit))
    geometry = (
        tuple(int(v) for v in decoded.categorical),
        tuple(float(v) for v in decoded.numeric),
        tuple(int(v) for v in decoded.primitive_legal),
    )
    exact_history = _history_exact(decoded)
    structured_history = _history_structured(decoded)
    v1_history = _history_v1_like(decoded)
    fixed = (
        tuple(int(v) for v in decoded.categorical),
        tuple(int(v) for v in decoded.rank_tokens),
        tuple(int(v) for v in decoded.same_suit),
        tuple(float(v) for v in decoded.numeric),
        tuple(int(v) for v in decoded.primitive_legal),
    )
    return {
        "exact_observation": hashlib.sha256(bytes(observation)).digest(),
        "cards_only": _hash_material(cards),
        "geometry_only": _hash_material(geometry),
        "fixed_state_no_history": _hash_material(fixed),
        "history_exact": _hash_material(exact_history),
        "history_structured": _hash_material(structured_history),
        "history_v1_like": _hash_material(v1_history),
        "no_cards_plus_exact_history": _hash_material((geometry, exact_history)),
        "no_cards_plus_structured_history": _hash_material((geometry, structured_history)),
        "no_cards_plus_v1_history": _hash_material((geometry, v1_history)),
    }


def _find_final_checkpoints(input_root: Path) -> dict[tuple[str, str, int], Path]:
    found = {}
    for checkpoint in sorted(input_root.rglob("checkpoint.pt")):
        report_path = checkpoint.parent / "report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("finalized")):
            continue
        key = (str(report.get("representation")), str(report.get("domain")), int(report.get("training_seed", -1)))
        if key in found:
            raise RuntimeError(f"duplicate finalized checkpoint identity: {key}")
        found[key] = checkpoint
    expected = {(r, d, int(s)) for r in REPRESENTATIONS for d in DOMAINS for s in TRAINING_SEEDS}
    if set(found) != expected:
        raise RuntimeError(f"final checkpoint inventory mismatch missing={sorted(expected-set(found))} extra={sorted(set(found)-expected)}")
    return found


def _load_memory_state(checkpoint: Path, *, representation: str, domain: str, training_seed: int, payload_key: str, training_execution_sha: str) -> dict:
    rng_state = torch.get_rng_state().clone()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        torch.set_rng_state(rng_state)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"wrong checkpoint schema: {checkpoint}")
    if str(payload.get("execution_sha")) != str(training_execution_sha):
        raise RuntimeError(f"checkpoint execution SHA mismatch: {checkpoint}")
    if str(payload.get("representation")) != representation or str(payload.get("domain")) != domain or int(payload.get("seed", -1)) != int(training_seed):
        raise RuntimeError(f"checkpoint identity mismatch: {checkpoint}")
    if payload.get("architecture_fingerprint_sha256") != MODEL_FINGERPRINTS[representation]:
        raise RuntimeError(f"checkpoint architecture fingerprint mismatch: {checkpoint}")
    memory = dict(payload[payload_key])
    del payload
    gc.collect()
    return memory


def _sets_for_memory(memory: dict) -> tuple[dict[str, set[bytes]], dict]:
    items = list(memory.get("items") or [])
    sets = {name: set() for name in PROJECTION_NAMES}
    for index, item in enumerate(items, start=1):
        digests = _projection_digests(bytes(item.observation))
        for name in PROJECTION_NAMES:
            sets[name].add(digests[name])
        if index % 25000 == 0:
            print(f"[Phase1B] decoded {index}/{len(items)} retained samples", flush=True)
    capacity = int(memory.get("capacity", 0))
    seen = int(memory.get("seen", 0))
    summary = {
        "capacity": capacity,
        "seen": seen,
        "retained": len(items),
        "saturation_factor_seen_over_capacity": float(seen / capacity) if capacity > 0 else None,
        "retention_fraction_retained_over_seen": float(len(items) / seen) if seen > 0 else None,
        "unique_by_projection": {name: len(sets[name]) for name in PROJECTION_NAMES},
    }
    return sets, summary


def _jaccard(left: set[bytes], right: set[bytes]) -> dict:
    inter = len(left & right)
    union = len(left | right)
    return {"left_unique": len(left), "right_unique": len(right), "intersection": inter, "union": union, "jaccard": float(inter / union) if union else 1.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V1+ Phase1B reservoir projection decomposition")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--training-execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    input_root = args.input_root.resolve()
    output = args.out.resolve()
    diagnostic_sha = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    checkpoints = _find_final_checkpoints(input_root)
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    cell_summaries = []
    overlap_rows = []
    for representation in REPRESENTATIONS:
        for domain in DOMAINS:
            for memory_name, payload_key in MEMORIES:
                pair_sets = {}
                for training_seed in (seed_a, seed_b):
                    print(f"[Phase1B] {representation} {domain} {memory_name}: seed {training_seed}", flush=True)
                    memory = _load_memory_state(
                        checkpoints[(representation, domain, training_seed)],
                        representation=representation,
                        domain=domain,
                        training_seed=training_seed,
                        payload_key=payload_key,
                        training_execution_sha=str(args.training_execution_sha),
                    )
                    sets, summary = _sets_for_memory(memory)
                    pair_sets[training_seed] = sets
                    cell_summaries.append({"representation": representation, "domain": domain, "memory": memory_name, "training_seed": training_seed, **summary})
                    del memory
                    gc.collect()
                overlap_rows.append({
                    "representation": representation,
                    "domain": domain,
                    "memory": memory_name,
                    "training_seed_pair": [seed_a, seed_b],
                    "projection_overlap": {name: _jaccard(pair_sets[seed_a][name], pair_sets[seed_b][name]) for name in PROJECTION_NAMES},
                })
                del pair_sets
                gc.collect()
    result = {
        "schema": SCHEMA,
        "status": "PHASE1B_COMPLETE_NO_ARCHITECTURE_SELECTED",
        "purpose": "Separate chance/card/current-state support from public-history support using existing final x16 reservoirs only.",
        "diagnostic_execution_sha": diagnostic_sha,
        "training_execution_sha": str(args.training_execution_sha),
        "representations": list(REPRESENTATIONS),
        "domains": list(DOMAINS),
        "training_seeds": [seed_a, seed_b],
        "projection_names": list(PROJECTION_NAMES),
        "cell_summaries": cell_summaries,
        "cross_seed_overlap": overlap_rows,
        "guardrails": ["No training or traversal performed.", "No model or reservoir mutated.", "No architecture winner selected.", "Overlap is diagnostic and is not a strategic-strength metric."],
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(json.dumps({"status": result["status"], "cell_summaries": len(cell_summaries), "overlap_rows": len(overlap_rows), "out": str(output)}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
