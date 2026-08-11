from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import run_r7_3_partial_exact_ensemble_paired as base
from spincore.deep_cfr import uniform_policy
from spincore_nn.codec import collate_inputs, decode_spnniv1


REGRET_FLOOR_EPSILON = 0.05


def _epsilon_floor_policy(values: list[float], legal: tuple[int, ...], epsilon: float) -> tuple[float, ...]:
    """Scale-relative positive-regret floor from the certified sign-sensitivity probe."""
    scale = math.sqrt(sum(float(values[a]) ** 2 for a in legal) / max(len(legal), 1))
    floor = float(epsilon) * max(scale, 1e-8)
    weights = [0.0] * len(values)
    total = 0.0
    for action in legal:
        w = max(float(values[action]), 0.0) + floor
        weights[action] = w
        total += w
    if total <= 0.0:
        p = 1.0 / len(legal)
        return tuple(p if action in legal else 0.0 for action in range(len(values)))
    return tuple(w / total for w in weights)


class RegretFloorPolicyMixture:
    """Apply a scale-relative floor to each member's positive regret, then average policies.

    This directly regularizes the empirically fragile nonlinear regret-map boundary.
    It leaves the estimator, memory, neural fits, deal schedule, and primary RNG
    contract unchanged. It is an algorithmic diagnostic, not recovered production
    semantics.
    """

    def __init__(self, *, device: str):
        self.models: list[torch.nn.Module] = []
        self.device = device

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if not self.models:
            return uniform_policy(state, observation, legal)
        batch = collate_inputs([decode_spnniv1(observation)], device=self.device)
        policies = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                raw = [float(x) for x in model(batch)[0].detach().cpu().tolist()]
                policies.append(_epsilon_floor_policy(raw, legal, REGRET_FLOOR_EPSILON))
        return tuple(
            sum(float(policy[action]) for policy in policies) / float(len(policies))
            for action in range(6)
        )


def main() -> int:
    global REGRET_FLOOR_EPSILON
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--regret-floor-epsilon", type=float, default=0.05)
    known, remaining = probe.parse_known_args()
    if float(known.regret_floor_epsilon) < 0.0:
        raise SystemExit("regret-floor-epsilon must be nonnegative")
    REGRET_FLOOR_EPSILON = float(known.regret_floor_epsilon)

    import sys
    sys.argv = [sys.argv[0], "--out", str(known.out)] + remaining
    base.EnsembleAdvantagePolicy = RegretFloorPolicyMixture
    rc = int(base.main())

    payload = json.loads(known.out.read_text(encoding="utf-8"))
    if payload.get("runner_failed_before_report"):
        return rc
    payload["schema"] = "SPINCORE_R7_3_POLICY_MIXTURE_REGRET_FLOOR_V1"
    payload["ensemble_mapping"] = "SCALE_RELATIVE_REGRET_FLOOR_EACH_MEMBER_THEN_AVERAGE_POLICY"
    payload["regret_floor"] = {
        "epsilon": float(REGRET_FLOOR_EPSILON),
        "scale": "RMS_LEGAL_PREDICTED_ADVANTAGE_PER_MEMBER_PER_STATE",
        "formula": "weight_a=max(advantage_a,0)+epsilon*RMS(legal_advantages)",
        "source_evidence": "validation/R7_3_ADVANTAGE_FIT_SIGN_SENSITIVITY_256.json",
        "intent": "Regularize near-zero regret sign/support flips at the demonstrated nonlinear feedback boundary."
    }
    payload["production_policy_mapping_changed"] = False
    payload["theoretical_equivalence_claimed"] = False
    payload["promotion_note"] = (
        "Algorithmic diagnostic only. Each fitted Advantage member is mapped with the scale-relative "
        "positive-regret floor used in the prior same-memory sign-sensitivity probe, then member policies "
        "are averaged. Partial-exact estimator, member fitting, deal schedule, primary RNG stream and "
        "frozen gates remain unchanged. Any promotion requires explicit versioning and recertification."
    )
    payload["ready_for_tables"] = False
    known.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": payload["schema"],
        "iterations": payload["iterations"],
        "ensemble_size": payload["ensemble_size"],
        "regret_floor": payload["regret_floor"],
        "cross_seed": payload["cross_seed"],
        "per_seed_fit_pass": payload["per_seed_fit_pass"],
        "r7_3_pass": payload["r7_3_pass"],
    }, indent=2, sort_keys=True), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
