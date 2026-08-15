from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import time
from pathlib import Path

import torch

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.solver import SolverLibrary
from spincore.solver_v3 import neural_bytes_v3
from spincore_nn.action_models import (
    collate_action_observations,
    make_advantage_action_model,
)
from spincore_nn.codec_v3 import decode_spnniv3
from spincore_nn.models_v3_final import (
    collate_v3_observations,
    make_h2_final_v3,
    make_h3_final_v3,
)

SCHEMA = "SPINCORE_R7_5_3C_PHASE2_RESOURCE_PREFLIGHT_REPORT_V1"
FREEZE_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_RESOURCE_PREFLIGHT_FREEZE_V1"
H0 = "H0_V1_RESOURCE_CONTROL"
H2 = "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL"
H3 = "H3_HYBRID_EXACT_SEMANTIC_FINAL"
REPRESENTATIONS = (H0, H2, H3)
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
PF0 = "PF0_CONTROL_33_75_AI"


def _hash_u64(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _distribution(values) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        return {"min": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": min(data),
        "mean": sum(data) / len(data),
        "p95": _percentile(data, 0.95),
        "max": max(data),
    }


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _legal_mask(legal: tuple[int, ...]) -> tuple[int, ...]:
    legal_set = {int(action) for action in legal}
    if not legal_set:
        raise RuntimeError("resource state has empty universal legal set")
    return tuple(1 if action in legal_set else 0 for action in range(10))


def _choose_walk_action(legal: tuple[int, ...], rng: random.Random) -> int:
    candidates = list(legal)
    if len(candidates) > 1 and 0 in candidates:
        candidates.remove(0)  # avoid voluntary early fold when a continuation exists
    if len(candidates) > 1 and 9 in candidates:
        candidates.remove(9)  # avoid all-in when another non-fold continuation exists
    if not candidates:
        candidates = list(legal)
    return int(candidates[rng.randrange(len(candidates))])


def collect_states(
    *,
    solver: SolverLibrary,
    repo_root: Path,
    domain: str,
    target: int,
) -> tuple[list[dict], dict]:
    action_spec = postflop_candidate_specs(repo_root)[PF0]
    scenarios = action_scenario_cycle(domain)
    samples: list[dict] = []
    root_index = 0
    street_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    roots_used = 0
    nodes_walked = 0
    started = time.perf_counter()

    while len(samples) < target:
        if root_index >= 4096:
            raise RuntimeError("resource preflight exhausted root safety bound")
        episode = scenarios[root_index % len(scenarios)]
        deck_seed = _hash_u64(f"R7.5.3C:RESOURCE:{domain}:{root_index}")
        walk_rng = random.Random(
            _hash_u64(f"R7.5.3C:RESOURCE:WALK:{domain}:{root_index}")
        )
        state = solver.create(episode, deck_seed)
        roots_used += 1
        try:
            for _depth in range(64):
                if state.terminal or len(samples) >= target:
                    break
                v3_payload = neural_bytes_v3(state)
                decoded = decode_spnniv3(v3_payload)
                street = int(decoded.categorical[1])
                active_mask = action_spec.active_mask(street)
                legal = state.universal_legal_actions(active_mask)
                if not legal:
                    raise RuntimeError("nonterminal resource state has no PF0 legal action")
                samples.append(
                    {
                        "v1": state.neural_bytes(),
                        "v3": v3_payload,
                        "legal": _legal_mask(legal),
                        "street": street,
                        "history_len": int(decoded.history_len),
                    }
                )
                street_counts[street] += 1
                nodes_walked += 1
                action = _choose_walk_action(legal, walk_rng)
                state.apply_universal(active_mask, action)
        finally:
            state.close()
        root_index += 1

    collection_seconds = time.perf_counter() - started
    if len(samples) != target:
        raise RuntimeError(f"resource state count mismatch: {len(samples)} != {target}")
    if any(street_counts[street] < 1 for street in range(4)):
        raise RuntimeError(f"resource preflight missed street coverage: {street_counts}")
    return samples, {
        "states": len(samples),
        "roots_used": roots_used,
        "walk_nodes": nodes_walked,
        "street_counts": {str(key): int(value) for key, value in street_counts.items()},
        "collection_seconds": float(collection_seconds),
    }


def _make_model(representation: str):
    if representation == H0:
        cfg, model = make_advantage_action_model(
            "C0_V1_FROZEN_CONTROL", device="cpu", seed=0x753C
        )
        return cfg, model, False
    if representation == H2:
        cfg, model = make_h2_final_v3(device="cpu", seed=0x753C)
        return cfg, model, False
    if representation == H3:
        cfg, model = make_h3_final_v3(device="cpu", seed=0x753C)
        return cfg, model, True
    raise ValueError(representation)


def _collate(representation: str, rows: list[dict], *, with_semantics: bool):
    masks = [row["legal"] for row in rows]
    if representation == H0:
        return collate_action_observations(
            "C0_V1_FROZEN_CONTROL",
            [row["v1"] for row in rows],
            masks,
            device="cpu",
        )
    return collate_v3_observations(
        [row["v3"] for row in rows],
        masks,
        with_semantics=with_semantics,
        device="cpu",
    )


def benchmark_representation(
    representation: str,
    rows: list[dict],
    *,
    warmup_passes: int,
    batch1_passes: int,
    batch64_passes: int,
) -> dict:
    cfg, model, with_semantics = _make_model(representation)
    model.eval()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if representation in (H2, H3) and parameter_count > 500_000:
        raise RuntimeError(f"{representation} exceeds frozen 500k parameter cap: {parameter_count}")

    selected = [rows[index % len(rows)] for index in range(batch1_passes)]

    # Preprocess/collation timing. Keep model work out of this measurement.
    for index in range(min(warmup_passes, len(rows))):
        _collate(representation, [rows[index]], with_semantics=with_semantics)
    started = time.perf_counter()
    single_batches = [
        _collate(representation, [row], with_semantics=with_semantics)
        for row in selected
    ]
    preprocess_seconds = time.perf_counter() - started

    # Model-only batch1 timing from already-collated tensors.
    with torch.no_grad():
        for index in range(warmup_passes):
            model(single_batches[index % len(single_batches)])
        started = time.perf_counter()
        for batch in single_batches:
            out = model(batch)
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError(f"nonfinite {representation} batch1 output")
        model_batch1_seconds = time.perf_counter() - started

    # Online batch1 combines decode/semantic/orbit/collation plus model forward.
    with torch.no_grad():
        for index in range(warmup_passes):
            batch = _collate(
                representation,
                [selected[index % len(selected)]],
                with_semantics=with_semantics,
            )
            model(batch)
        started = time.perf_counter()
        for row in selected:
            batch = _collate(representation, [row], with_semantics=with_semantics)
            out = model(batch)
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError(f"nonfinite {representation} online output")
        online_batch1_seconds = time.perf_counter() - started

    # Training-style batch64 model throughput, with preprocessing excluded.
    batch64_rows = [rows[index % len(rows)] for index in range(64)]
    batch64 = _collate(representation, batch64_rows, with_semantics=with_semantics)
    with torch.no_grad():
        for _ in range(warmup_passes):
            model(batch64)
        started = time.perf_counter()
        for _ in range(batch64_passes):
            out = model(batch64)
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError(f"nonfinite {representation} batch64 output")
        model_batch64_seconds = time.perf_counter() - started

    # Full frozen-corpus finite-output audit in chunks.
    with torch.no_grad():
        for first in range(0, len(rows), 64):
            batch = _collate(
                representation,
                rows[first : first + 64],
                with_semantics=with_semantics,
            )
            out = model(batch)
            if out.shape[1] != 10 or not bool(torch.isfinite(out).all()):
                raise RuntimeError(f"invalid {representation} full-corpus output")

    return {
        "config": cfg.to_dict(),
        "parameter_count": parameter_count,
        "preprocess_seconds_per_sample": preprocess_seconds / len(single_batches),
        "model_batch1_seconds_per_sample": model_batch1_seconds / len(single_batches),
        "online_batch1_seconds_per_sample": online_batch1_seconds / len(single_batches),
        "model_batch64_seconds_per_sample": model_batch64_seconds / (batch64_passes * 64),
        "finite_output_audit": True,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    freeze_path = root / "validation" / "R7_5_3C_PHASE2_RESOURCE_PREFLIGHT_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("status") != "FROZEN_BEFORE_RESOURCE_RESULTS":
        raise RuntimeError("resource preflight freeze contract mismatch")
    if freeze.get("selection_authority") is not False:
        raise RuntimeError("resource preflight illegally gained selection authority")
    if freeze["state_collection"]["observations_per_domain"] != 256:
        raise RuntimeError("resource preflight state count drift")
    timing = freeze["timing_protocol"]
    if timing != {
        "warmup_passes": 16,
        "batch1_timed_passes": 128,
        "batch64_timed_passes": 32,
        "state_order": "fixed collection order; no timing-dependent reshuffle",
        "notes": "CI absolute wall time is not treated as Ryzen wall time. Relative ratios and gross feasibility are diagnostic only."
    }:
        raise RuntimeError("resource preflight timing contract drift")

    torch.set_num_threads(2)
    if torch.get_num_threads() != 2:
        raise RuntimeError("resource preflight torch thread contract not applied")
    solver = SolverLibrary(args.solver)
    rows, collection = collect_states(
        solver=solver,
        repo_root=root,
        domain=args.domain,
        target=256,
    )
    result = benchmark_representation(
        args.representation,
        rows,
        warmup_passes=16,
        batch1_passes=128,
        batch64_passes=32,
    )

    v3_bytes = [len(row["v3"]) for row in rows]
    v1_bytes = [len(row["v1"]) for row in rows]
    history_lengths = [row["history_len"] for row in rows]
    files = [
        "python/spincore_nn/models_v3_final.py",
        "python/spincore_nn/card_orbit_v3.py",
        "python/spincore_nn/codec_v3.py",
        "python/spincore_nn/semantics_v3.py",
        "python/spincore/r7_5_representation_v3.py",
        "python/spincore/solver_v3.py",
    ]
    payload = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "representation": str(args.representation),
        "domain": str(args.domain),
        "selection_authority": False,
        "collection": collection,
        "observation_bytes": {
            "spnniv1": _distribution(v1_bytes),
            "spnniv3": _distribution(v3_bytes),
        },
        "history_length": _distribution(history_lengths),
        "benchmark": result,
        "source_sha256": {
            path: _sha256_file(root / path)
            for path in files
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
