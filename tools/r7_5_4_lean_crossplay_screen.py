from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_checkpoint import load_action_checkpoint
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_action_stage_contract import PAYOUT
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations, representation_wire


SCHEMA = "SPINCORE_R7_5_4_LEAN_CROSSPLAY_SCREEN_V1"
ELIGIBLE = (
    "PF0_CONTROL_33_75_AI",
    "PF1_33_50_75_AI",
    "PF2_33_50_75_100_AI",
    "PF3_COMPACT_33_66_100_AI",
    "PF4_CRUSHER_COMPACT_40_66_100_AI",
)
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")


@dataclass
class LoadedPolicy:
    candidate_id: str
    domain: str
    training_seed: int
    bundle: object
    spec: object
    wire: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _keyed_u64(*parts: object) -> int:
    text = "|".join(str(x) for x in ("SpinCore", "R7.5.4", "LEAN-XPLAY-V1", *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def _keyed_uniform(*parts: object) -> float:
    return (_keyed_u64(*parts) + 0.5) / float(1 << 64)


def _street(state) -> int:
    payload = state.neural_bytes_v2()
    if len(payload) != 830 or not payload.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("lean crossplay requires valid SPNNIV2 state metadata")
    return int(payload[112])


def _policy_probabilities(loaded: LoadedPolicy, state) -> tuple[int, tuple[int, ...], tuple[float, ...]]:
    active_mask = int(loaded.spec.active_mask(_street(state)))
    legal = tuple(int(x) for x in state.universal_legal_actions(active_mask))
    if not legal:
        raise RuntimeError("nonterminal state has no legal universal action")
    observation = state.neural_bytes() if loaded.wire == "SPNNIV1" else state.neural_bytes_v2()
    batch = collate_action_observations(
        loaded.bundle.selected_representation,
        [observation],
        [legal_mask(legal)],
        device="cpu",
    )
    loaded.bundle.policy.eval()
    with torch.no_grad():
        logits = loaded.bundle.policy(batch).masked_fill(~batch["legal"], -1e9)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
    return active_mask, legal, tuple(float(x) for x in probs)


def _sample_with_uniform(probabilities: tuple[float, ...], legal: tuple[int, ...], u: float) -> int:
    total = sum(float(probabilities[a]) for a in legal)
    if not math.isfinite(total) or total <= 0.0:
        return legal[min(int(u * len(legal)), len(legal) - 1)]
    cumulative = 0.0
    for action in legal:
        p = max(0.0, float(probabilities[action])) / total
        cumulative += p
        if u < cumulative:
            return int(action)
    return int(legal[-1])


def _live_seats(episode) -> tuple[int, ...]:
    return tuple(i for i, stack in enumerate(episode.stacks) if int(stack) > 0)


def _simulate(
    solver: SolverLibrary,
    episode,
    *,
    deck_seed: int,
    assignment: dict[int, LoadedPolicy],
    domain: str,
    training_seed: int,
    hand_index: int,
) -> tuple[float, float, float]:
    state = solver.create(episode, int(deck_seed))
    ordinals = {0: 0, 1: 0, 2: 0}
    decisions = 0
    try:
        while not state.terminal:
            actor = int(state.actor)
            try:
                loaded = assignment[actor]
            except KeyError as exc:
                raise RuntimeError(f"no policy assigned to live actor {actor}") from exc
            active_mask, legal, probabilities = _policy_probabilities(loaded, state)
            ordinal = int(ordinals[actor])
            u = _keyed_uniform("policy", domain, training_seed, hand_index, actor, ordinal)
            action = _sample_with_uniform(probabilities, legal, u)
            state.apply_universal(active_mask, action)
            ordinals[actor] = ordinal + 1
            decisions += 1
            if decisions > 512:
                raise RuntimeError("decision guard exceeded")
        return tuple(float(x) for x in state.terminal_icm_delta(PAYOUT))
    finally:
        state.close()


def _normal_ci(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan, math.nan
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan, math.nan, math.nan
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def _prepare_cells(bundle_path: Path, cache: Path) -> dict[tuple[str, str, int], Path]:
    cache.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[str, str, int], Path] = {}
    with zipfile.ZipFile(bundle_path, "r") as outer:
        manifest = json.loads(outer.read("MANIFEST.json"))
        expected = {row["file"]: row for row in manifest["files"]}
        for member in sorted(n for n in outer.namelist() if n.startswith("artifacts/") and n.endswith(".zip")):
            name = Path(member).name
            blob = outer.read(member)
            row = expected.get(name)
            if row is None:
                raise RuntimeError(f"bundle manifest missing {name}")
            if _sha256_bytes(blob) != str(row["sha256"]):
                raise RuntimeError(f"nested artifact hash mismatch: {name}")
            with zipfile.ZipFile(io.BytesIO(blob), "r") as inner:
                report_name = next(n for n in inner.namelist() if n.endswith("report.json"))
                checkpoint_name = next(n for n in inner.namelist() if n.endswith("checkpoint.pt"))
                report = json.loads(inner.read(report_name))
                final = report.get("final_report") or {}
                candidate = str(report["candidate_id"])
                domain = str(report["domain"])
                seed = int(report["training_seed"])
                if candidate not in ELIGIBLE or domain not in DOMAINS:
                    continue
                if not bool(report.get("finalized")):
                    raise RuntimeError(f"artifact is not finalized: {name}")
                if int(final.get("roots", -1)) != 160:
                    raise RuntimeError(f"artifact does not contain 160-root final: {name}")
                if not bool(final.get("advantage_gate_pass")) or not bool(final.get("policy_gate_pass")):
                    raise RuntimeError(f"learning gate failed: {name}")
                out = cache / domain / str(seed) / candidate / "checkpoint.pt"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    out.write_bytes(inner.read(checkpoint_name))
                cells[(candidate, domain, seed)] = out
    expected_count = len(ELIGIBLE) * len(DOMAINS) * 3
    if len(cells) != expected_count:
        raise RuntimeError(f"expected {expected_count} eligible cells, found {len(cells)}")
    return cells


def _load_domain_seed(
    cells: dict[tuple[str, str, int], Path],
    *,
    repo_root: Path,
    domain: str,
    seed: int,
) -> dict[str, LoadedPolicy]:
    loaded: dict[str, LoadedPolicy] = {}
    for candidate in ELIGIBLE:
        bundle, progress, spec, extra = load_action_checkpoint(
            cells[(candidate, domain, seed)],
            repo_root=repo_root,
            device="cpu",
        )
        final = dict(extra.get("final_report") or {})
        if progress.phase != "post_policy_fit":
            raise RuntimeError(f"{candidate} {domain} {seed} is not a finalized checkpoint")
        if final.get("schema") != "SPINCORE_R7_5_ACTION_DOMAIN_FINAL_REPORT_V1":
            raise RuntimeError(f"{candidate} {domain} {seed} missing final report")
        loaded[candidate] = LoadedPolicy(
            candidate_id=candidate,
            domain=domain,
            training_seed=seed,
            bundle=bundle,
            spec=spec,
            wire=representation_wire(bundle.selected_representation),
        )
    return loaded


def _screen_one(
    solver: SolverLibrary,
    loaded: dict[str, LoadedPolicy],
    *,
    domain: str,
    seed: int,
    baseline_id: str,
    hands_per_scenario: int,
    progress_every: int,
) -> list[dict]:
    scenarios = action_scenario_cycle(domain)
    baseline = loaded[baseline_id]
    candidates = [c for c in ELIGIBLE if c != baseline_id]
    hand_scores = {candidate: [] for candidate in candidates}
    total_hands = int(hands_per_scenario) * len(scenarios)
    started = time.perf_counter()

    for hand_index in range(total_hands):
        episode = scenarios[hand_index % len(scenarios)]
        live = _live_seats(episode)
        deck_seed = _keyed_u64("deck", domain, seed, hand_index)
        base_assignment = {seat: baseline for seat in live}
        reference = _simulate(
            solver,
            episode,
            deck_seed=deck_seed,
            assignment=base_assignment,
            domain=domain,
            training_seed=seed,
            hand_index=hand_index,
        )
        for candidate_id in candidates:
            candidate = loaded[candidate_id]
            seat_scores = []
            for seat in live:
                assignment = {s: baseline for s in live}
                assignment[seat] = candidate
                test = _simulate(
                    solver,
                    episode,
                    deck_seed=deck_seed,
                    assignment=assignment,
                    domain=domain,
                    training_seed=seed,
                    hand_index=hand_index,
                )
                seat_scores.append(float(test[seat]) - float(reference[seat]))
            hand_scores[candidate_id].append(statistics.fmean(seat_scores))
        if progress_every and (hand_index + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "domain": domain,
                        "training_seed": seed,
                        "completed_hands": hand_index + 1,
                        "total_hands": total_hands,
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    rows = []
    for candidate_id in candidates:
        values = hand_scores[candidate_id]
        mean, se, low, high = _normal_ci(values)
        rows.append(
            {
                "candidate_id": candidate_id,
                "baseline_id": baseline_id,
                "domain": domain,
                "training_seed": int(seed),
                "hands": len(values),
                "seat_rotations_per_hand": len(_live_seats(scenarios[0])),
                "paired_mean_icm_delta_vs_baseline": mean,
                "standard_error": se,
                "normal_95_ci_low": low,
                "normal_95_ci_high": high,
                "sample_sd": statistics.stdev(values) if len(values) >= 2 else math.nan,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Lean decision-focused R7.5.4 action crossplay screen")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=ELIGIBLE, default="PF0_CONTROL_33_75_AI")
    parser.add_argument("--hands-per-scenario", type=int, default=20)
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=[1737995611, 645939859, 1311335590],
    )
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.hands_per_scenario <= 0:
        raise SystemExit("--hands-per-scenario must be positive")
    torch.set_num_threads(int(args.torch_threads))
    cells = _prepare_cells(args.bundle, args.cache)
    solver = SolverLibrary(args.solver)

    rows = []
    started = time.perf_counter()
    for domain in args.domains:
        for seed in args.training_seeds:
            loaded = _load_domain_seed(cells, repo_root=args.repo_root, domain=domain, seed=int(seed))
            rows.extend(
                _screen_one(
                    solver,
                    loaded,
                    domain=domain,
                    seed=int(seed),
                    baseline_id=args.baseline,
                    hands_per_scenario=int(args.hands_per_scenario),
                    progress_every=int(args.progress_every),
                )
            )
            del loaded

    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["candidate_id"], []).append(float(row["paired_mean_icm_delta_vs_baseline"]))
    aggregate = []
    for candidate_id, means in grouped.items():
        mean, se, low, high = _normal_ci(means)
        aggregate.append(
            {
                "candidate_id": candidate_id,
                "baseline_id": args.baseline,
                "domain_seed_cell_count": len(means),
                "mean_of_domain_seed_means": mean,
                "cell_mean_standard_error": se,
                "cell_mean_normal_95_ci_low": low,
                "cell_mean_normal_95_ci_high": high,
            }
        )
    aggregate.sort(key=lambda x: x["mean_of_domain_seed_means"], reverse=True)

    payload = {
        "schema": SCHEMA,
        "baseline_id": args.baseline,
        "hands_per_scenario": int(args.hands_per_scenario),
        "training_seeds": [int(x) for x in args.training_seeds],
        "domains": list(args.domains),
        "rows": rows,
        "aggregate": aggregate,
        "wall_seconds": time.perf_counter() - started,
        "interpretation": {
            "purpose": "cheap strategic screen only; not a final winner declaration",
            "positive_score": "candidate seat improves ICM delta versus the same seat in all-baseline reference play",
            "next_step": "eliminate only clear losers; escalate close survivors with more hands and additional baselines/pairwise crossplay",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output), "aggregate": aggregate}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
