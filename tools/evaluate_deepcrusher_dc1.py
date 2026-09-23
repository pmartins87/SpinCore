#!/usr/bin/env python3
from __future__ import annotations

"""DC1 paired SpinCore-vs-DeepCrusher hand-level development benchmark.

This runner intentionally remains DEVELOPMENT_ONLY until the real-OpenHoldem
parity fixture gate for DC0 is passed. It nevertheless exercises the complete
offline match path: empirical SpinGo sampler, paired deals, balanced HU/3H
lineups, exact actions, full DeepCrusher R8 oracle, chip accounting and
decision-level traces.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_benchmark import (  # noqa: E402
    DEEPC_RUSHER_POLICY_ID,
    SPINCORE_POLICY_ID,
    OfflineHeadToHeadEngine,
)
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler  # noqa: E402
from spincore.solver import Episode  # noqa: E402


_SOLVER = None
_SPIN_POLICY = None
_DC_POLICY = None
_ENGINE_SEED = None

LIB1 = ROOT / "fixtures" / "openppl_library_dc0" / "OpenPPL_Library_part1.ohf"
LIB2 = ROOT / "fixtures" / "openppl_library_dc0" / "OpenPPL_Library_part2.ohf"
DC_SOURCE = (
    ROOT
    / "fixtures"
    / "deepcrusher_r8_v22"
    / "DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_ASCII_20260914.txt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build" / "libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenarios", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--max-decisions", type=int, default=200)
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def _mix64(*values: int) -> int:
    x = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        y = int(value) & mask
        x ^= (
            y
            + 0x9E3779B97F4A7C15
            + ((x << 6) & mask)
            + (x >> 2)
        ) & mask
        x &= mask
    return x


def _init_worker(
    solver_path: str,
    checkpoint_path: str,
    root_path: str,
    seed: int,
) -> None:
    global _SOLVER, _SPIN_POLICY, _DC_POLICY, _ENGINE_SEED

    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["SPINCORE_TORCH_THREADS"] = "1"

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from spincore.deepcrusher_benchmark import SpinCoreCheckpointPolicy
    from spincore.deepcrusher_policy import DeepCrusherR8Policy
    from spincore.lean_functional_agent import LeanFunctionalAgent
    from spincore.solver import SolverLibrary

    _SOLVER = SolverLibrary(solver_path)
    if not _SOLVER.explicit_deal_available:
        raise RuntimeError(
            "DC1 requires solver explicit-deal snapshot ABI for exact suit semantics"
        )
    agent = LeanFunctionalAgent.from_checkpoint(checkpoint_path, seed=0)
    _SPIN_POLICY = SpinCoreCheckpointPolicy(agent)
    _DC_POLICY = DeepCrusherR8Policy.from_repository(root_path)
    _ENGINE_SEED = int(seed)


def _scenario_summary(observations) -> dict[str, Any]:
    sc_total = 0
    dc_total = 0
    sc_exposures = 0
    dc_exposures = 0
    decisions = 0

    for row in observations:
        decisions += int(row.decisions)
        if sum(int(x) for x in row.chip_delta) != 0:
            raise RuntimeError("non-zero-sum terminal row")
        for seat, policy in enumerate(row.lineup.seats):
            if policy == SPINCORE_POLICY_ID:
                sc_total += int(row.chip_delta[seat])
                sc_exposures += 1
            elif policy == DEEPC_RUSHER_POLICY_ID:
                dc_total += int(row.chip_delta[seat])
                dc_exposures += 1

    if sc_exposures != dc_exposures:
        raise RuntimeError(
            f"unbalanced policy exposures: SpinCore={sc_exposures} "
            f"DeepCrusher={dc_exposures}"
        )
    if sc_total + dc_total != 0:
        raise RuntimeError(
            f"policy aggregate not zero-sum: SpinCore={sc_total} DeepCrusher={dc_total}"
        )

    sc_ev = sc_total / float(sc_exposures)
    dc_ev = dc_total / float(dc_exposures)
    return {
        "games": len(observations),
        "decisions": decisions,
        "spincore_total_chips": sc_total,
        "deepcrusher_total_chips": dc_total,
        "policy_exposures_each": sc_exposures,
        "spincore_chips_per_exposure": sc_ev,
        "deepcrusher_chips_per_exposure": dc_ev,
        "paired_spincore_minus_deepcrusher_chips_per_exposure": sc_ev - dc_ev,
    }


def _worker(task: tuple[int, Episode, int, int]) -> dict[str, Any]:
    index, episode, deal_seed, max_decisions = task
    traces = []

    engine = OfflineHeadToHeadEngine(
        _SOLVER,
        spincore_policy=_SPIN_POLICY,
        deepcrusher_policy=_DC_POLICY,
        master_seed=int(_ENGINE_SEED),
        max_decisions=int(max_decisions),
        decision_sink=traces.append,
    )
    observations = engine.play_balanced_block(
        episode,
        deal_seed=int(deal_seed),
        scenario_index=int(index),
    )

    domain = "TRUE_HEADS_UP" if bool(episode.game_is_hu) else "THREE_HANDED"
    blind = f"{int(episode.small_blind)}/{int(episode.big_blind)}"
    summary = _scenario_summary(observations)
    summary.update(
        {
            "scenario": int(index),
            "domain": domain,
            "blind": blind,
            "dealer_id": int(episode.dealer_id),
            "deal_seed": int(deal_seed),
        }
    )
    return {
        "summary": summary,
        "traces": [asdict(trace) for trace in traces],
    }


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(statistics.fmean(values))
    sem = 0.0 if n == 1 else float(statistics.stdev(values) / math.sqrt(n))
    half = 1.96 * sem
    return {
        "n": n,
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margin = [
        float(row["paired_spincore_minus_deepcrusher_chips_per_exposure"])
        for row in rows
    ]
    sc = [float(row["spincore_chips_per_exposure"]) for row in rows]
    return {
        "scenario_clusters": len(rows),
        "balanced_games": sum(int(row["games"]) for row in rows),
        "decisions": sum(int(row["decisions"]) for row in rows),
        "spincore_chips_per_policy_seat_hand": _mean_ci(sc),
        "paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand": _mean_ci(margin),
    }


def _summaries(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_domain = {"ALL": _stats(rows)}
    for domain in ("THREE_HANDED", "TRUE_HEADS_UP"):
        by_domain[domain] = _stats([row for row in rows if row["domain"] == domain])

    by_blind: dict[str, Any] = {}
    keys = sorted({(str(row["domain"]), str(row["blind"])) for row in rows})
    for domain, blind in keys:
        subset = [
            row
            for row in rows
            if row["domain"] == domain and row["blind"] == blind
        ]
        by_blind[f"{domain}:{blind}"] = _stats(subset)
    return by_domain, by_blind


def main() -> int:
    args = parse_args()
    if args.scenarios <= 0:
        raise SystemExit("--scenarios must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_decisions <= 0:
        raise SystemExit("--max-decisions must be positive")

    required = (args.solver, args.checkpoint, DC_SOURCE, LIB1, LIB2)
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    # Multiprocessing workers must never fan out a multi-GB training checkpoint.
    if args.workers > 1 and args.checkpoint.stat().st_size > 512 * 1024 * 1024:
        raise SystemExit(
            "multiprocess DC1 requires a compact inference checkpoint; "
            f"refusing {args.checkpoint} ({args.checkpoint.stat().st_size} bytes)"
        )

    sampler = LegacyScenarioSampler(
        seed=int(args.seed) ^ 0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    tasks: list[tuple[int, Episode, int, int]] = []
    domain_counts = {"THREE_HANDED": 0, "TRUE_HEADS_UP": 0}
    blind_counts: dict[str, int] = {}
    for index in range(int(args.scenarios)):
        episode = sampler.sample_episode()
        domain = "TRUE_HEADS_UP" if bool(episode.game_is_hu) else "THREE_HANDED"
        blind = f"{int(episode.small_blind)}/{int(episode.big_blind)}"
        domain_counts[domain] += 1
        blind_counts[f"{domain}:{blind}"] = (
            blind_counts.get(f"{domain}:{blind}", 0) + 1
        )
        tasks.append(
            (
                int(index),
                episode,
                int(_mix64(args.seed, index, 0xD34A1)),
                int(args.max_decisions),
            )
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.traces.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    trace_count = 0
    policy_decision_counts = {SPINCORE_POLICY_ID: 0, DEEPC_RUSHER_POLICY_ID: 0}
    action_counts: dict[str, dict[str, int]] = {
        SPINCORE_POLICY_ID: {},
        DEEPC_RUSHER_POLICY_ID: {},
    }

    ctx = mp.get_context("spawn")
    worker_count = min(int(args.workers), int(args.scenarios))
    with args.traces.open("w", encoding="utf-8") as trace_stream:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                str(args.solver.resolve()),
                str(args.checkpoint.resolve()),
                str(ROOT.resolve()),
                int(args.seed),
            ),
        ) as pool:
            for chunk in pool.map(_worker, tasks, chunksize=1):
                rows.append(chunk["summary"])
                for trace in chunk["traces"]:
                    trace_stream.write(
                        json.dumps(trace, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    trace_count += 1
                    policy = str(trace["policy_id"])
                    policy_decision_counts[policy] += 1
                    action = str(trace["action_type"])
                    bucket = action_counts[policy]
                    bucket[action] = bucket.get(action, 0) + 1

    by_domain, by_blind = _summaries(rows)

    checkpoint_sha = _sha256(args.checkpoint)
    report = {
        "schema": "SPINCORE_DEEPCRUSHER_DC1_DEVELOPMENT_V1",
        "status": "PASS",
        "stage": "DC1_DEVELOPMENT_SMOKE",
        "canonical_quality_claim_authorized": False,
        "oracle_parity_status": "REAL_OPENHOLDEM_FIXTURES_PENDING",
        "warning": (
            "Mechanical/development benchmark only. Do not use this result as a "
            "canonical strength claim until DC0 real-OpenHoldem action+sizing "
            "parity fixtures pass across all streets."
        ),
        "spin_core": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
        },
        "deepcrusher": {
            "source": str(DC_SOURCE.resolve()),
            "source_sha256": _sha256(DC_SOURCE),
            "library_files": [str(LIB1.resolve()), str(LIB2.resolve())],
        },
        "spin_core_git_head": _git_head(),
        "seed": int(args.seed),
        "scenarios": int(args.scenarios),
        "workers": int(worker_count),
        "domain_counts": domain_counts,
        "blind_counts": dict(sorted(blind_counts.items())),
        "trace_file": str(args.traces.resolve()),
        "trace_decisions": int(trace_count),
        "policy_decision_counts": policy_decision_counts,
        "exact_action_type_counts": action_counts,
        "method": {
            "sampler": "legacy empirical full SpinGo 3H/HU blind-conditioned sampler",
            "deal_pairing": "same scenario/deal reused across balanced lineup block",
            "hu": "two games; policies swap live seats",
            "three_handed": "six AAB/ABB games; each policy has 9 seat exposures and each logical seat 3 times",
            "action_application": "exact Fold/Check/Call/BetTo/RaiseTo/AllIn without SpinCore-size quantization",
            "ci": "normal 95% CI over scenario-cluster values",
            "primary_scale_note": (
                "paired policy difference equals SpinCore per-exposure EV minus "
                "DeepCrusher per-exposure EV; because balanced blocks are zero-sum "
                "with equal exposures, it is exactly 2x SpinCore net EV."
            ),
        },
        "summary": by_domain,
        "by_blind": by_blind,
        "scenario_rows": rows,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    overall = by_domain["ALL"][
        "paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"
    ]
    print("=== SpinCore vs DeepCrusher DC1 development smoke ===")
    print(
        f"scenarios={args.scenarios} workers={worker_count} "
        f"seed={args.seed} traces={trace_count}"
    )
    print(
        "ALL paired SpinCore-DeepCrusher="
        f"{overall['mean']:+.3f} chips/policy-seat-hand "
        f"CI95=[{overall['ci95_low']:+.3f},{overall['ci95_high']:+.3f}]"
    )
    print(f"report={args.report.resolve()}")
    print(f"traces={args.traces.resolve()}")
    print("DC1_DEVELOPMENT_ONLY_NO_QUALITY_CLAIM")
    print("DEEPC_RUSHER_DC1_DEVELOPMENT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
