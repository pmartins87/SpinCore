#!/usr/bin/env python3
from __future__ import annotations

"""End-to-end DeepCrusher oracle smoke on the real SpinCore solver.

Uses a deterministic passive reference policy instead of a trained SpinCore
checkpoint so CI can exercise the complete R8 parse -> symbol projection ->
lifecycle -> action translation -> exact simulator apply path without bundling
large model artifacts. This is a runtime/mechanical smoke, never a strength
benchmark or OpenHoldem parity proof.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_benchmark import (  # noqa: E402
    ExternalExactAction,
    OfflineHeadToHeadEngine,
    SPINCORE_POLICY_ID,
)
from spincore.deepcrusher_policy import DeepCrusherR8Policy  # noqa: E402
from spincore.solver import Episode, SolverLibrary  # noqa: E402


class PassiveReferencePolicy:
    policy_id = SPINCORE_POLICY_ID

    def choose_exact(self, state, *, seat: int, rng: random.Random) -> ExternalExactAction:
        del seat, rng
        p = state.public_snapshot()
        if p.legal_call:
            return ExternalExactAction(2)
        if p.legal_check:
            return ExternalExactAction(1)
        if p.legal_fold:
            return ExternalExactAction(0)
        if p.legal_all_in:
            return ExternalExactAction(5)
        raise RuntimeError("passive reference has no legal action")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260923)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    solver = SolverLibrary(args.solver)
    if not solver.explicit_deal_available:
        raise SystemExit("solver missing explicit-deal snapshot ABI")

    deepcrusher = DeepCrusherR8Policy.from_repository(ROOT)
    passive = PassiveReferencePolicy()
    traces = []
    engine = OfflineHeadToHeadEngine(
        solver,
        spincore_policy=passive,
        deepcrusher_policy=deepcrusher,
        master_seed=int(args.seed),
        decision_sink=traces.append,
    )

    episodes = [
        Episode(
            total_chips=1500,
            game_is_hu=False,
            blind_index=0,
            small_blind=10,
            big_blind=20,
            stacks=(500, 500, 500),
            dealer_id=dealer,
        )
        for dealer in range(3)
    ] + [
        Episode(
            total_chips=1500,
            game_is_hu=True,
            blind_index=0,
            small_blind=10,
            big_blind=20,
            stacks=(750, 750, 0),
            dealer_id=dealer,
            dead_players=(2,),
        )
        for dealer in (0, 1)
    ]

    observations = []
    for index, episode in enumerate(episodes):
        observations.extend(
            engine.play_balanced_block(
                episode,
                deal_seed=int(args.seed + 1009 * index),
                scenario_index=index,
            )
        )

    if not observations:
        raise RuntimeError("no benchmark observations emitted")
    if any(sum(row.chip_delta) != 0 for row in observations):
        raise RuntimeError("non-zero-sum terminal row")

    dc_traces = [row for row in traces if row.policy_id == "DEEPCRUSHER"]
    if not dc_traces:
        raise RuntimeError("DeepCrusher emitted no decision traces")
    if any(row.policy_detail is None for row in dc_traces):
        raise RuntimeError("DeepCrusher trace missing OpenPPL provenance")

    streets = Counter(int(row.street) for row in dc_traces)
    actions = Counter(row.action_name for row in dc_traces)
    domains = Counter(row.domain for row in dc_traces)
    direct_kinds = Counter(
        str(row.policy_detail.get("openppl_result_kind"))
        for row in dc_traces
        if row.policy_detail is not None
    )

    payload = {
        "schema": "SPINCORE_DEEPCRUSHER_ORACLE_RUNTIME_SMOKE_V1",
        "status": "PASS",
        "scope": "MECHANICAL_RUNTIME_ONLY_NOT_OPENHOLDEM_PARITY_NOT_STRENGTH",
        "episodes": len(episodes),
        "balanced_games": len(observations),
        "decisions_total": len(traces),
        "deepcrusher_decisions": len(dc_traces),
        "deepcrusher_street_counts": dict(sorted(streets.items())),
        "deepcrusher_action_counts": dict(sorted(actions.items())),
        "deepcrusher_domain_counts": dict(sorted(domains.items())),
        "deepcrusher_openppl_result_kind_counts": dict(sorted(direct_kinds.items())),
        "all_terminal_rows_zero_sum": True,
        "all_deepcrusher_traces_have_provenance": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("DEEPC_RUSHER_ORACLE_RUNTIME_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
