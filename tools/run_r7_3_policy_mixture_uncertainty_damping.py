from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_r7_3_partial_exact_ensemble_paired as base
from spincore.deep_cfr import regret_matching_policy, uniform_policy
from spincore_nn.codec import collate_inputs, decode_spnniv1


EPSILON_SCALE = 1.0
EPSILON_CAP = 0.50


class UncertaintyDampedPolicyMixture:
    """Damp only states where independently fitted regret policies disagree.

    Member disagreement is measured as the average total-variation distance from
    each member policy to the ensemble mean. Epsilon is scale*disagreement,
    capped globally. Stable states are left nearly untouched; high-uncertainty
    states are mixed toward exact legal-action uniform behavior.

    Runtime counters are diagnostics only. They do not sample RNG and do not
    alter the returned policy.
    """

    instances: list["UncertaintyDampedPolicyMixture"] = []

    def __init__(self, *, device: str):
        self.models: list[torch.nn.Module] = []
        self.device = device
        self.calls = 0
        self.epsilon_sum = 0.0
        self.epsilon_max = 0.0
        self.disagreement_sum = 0.0
        self.raw_epsilon_max = 0.0
        self.cap_hit_calls = 0
        self.epsilon_ge_010_calls = 0
        self.epsilon_ge_025_calls = 0
        type(self).instances.append(self)

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
                raw = model(batch)[0].detach().cpu().tolist()
                policies.append(regret_matching_policy(raw, legal))
        mean = [0.0] * 6
        for action in range(6):
            mean[action] = sum(float(p[action]) for p in policies) / float(len(policies))
        disagreement = sum(
            0.5 * sum(abs(float(p[a]) - float(mean[a])) for a in range(6))
            for p in policies
        ) / float(len(policies))
        raw_eps = float(EPSILON_SCALE) * float(disagreement)
        eps = min(float(EPSILON_CAP), raw_eps)
        uniform = uniform_policy(state, observation, legal)
        out = tuple((1.0 - eps) * float(mean[a]) + eps * float(uniform[a]) for a in range(6))

        self.calls += 1
        self.epsilon_sum += float(eps)
        self.epsilon_max = max(float(self.epsilon_max), float(eps))
        self.disagreement_sum += float(disagreement)
        self.raw_epsilon_max = max(float(self.raw_epsilon_max), float(raw_eps))
        if raw_eps >= float(EPSILON_CAP) - 1e-15:
            self.cap_hit_calls += 1
        if eps >= 0.10:
            self.epsilon_ge_010_calls += 1
        if eps >= 0.25:
            self.epsilon_ge_025_calls += 1
        return out


def _runtime_statistics(payload: dict) -> list[dict]:
    seeds = list(payload.get("algorithm_seeds") or [])
    rows = []
    for index, obj in enumerate(UncertaintyDampedPolicyMixture.instances):
        calls = int(obj.calls)
        rows.append(
            {
                "algorithm_seed": seeds[index] if index < len(seeds) else None,
                "fitted_behavior_calls": calls,
                "mean_epsilon": float(obj.epsilon_sum / calls) if calls else 0.0,
                "max_epsilon": float(obj.epsilon_max),
                "mean_disagreement": float(obj.disagreement_sum / calls) if calls else 0.0,
                "max_raw_epsilon_before_cap": float(obj.raw_epsilon_max),
                "cap_hit_calls": int(obj.cap_hit_calls),
                "cap_hit_fraction": float(obj.cap_hit_calls / calls) if calls else 0.0,
                "epsilon_ge_0_10_fraction": float(obj.epsilon_ge_010_calls / calls) if calls else 0.0,
                "epsilon_ge_0_25_fraction": float(obj.epsilon_ge_025_calls / calls) if calls else 0.0,
            }
        )
    return rows


def main() -> int:
    global EPSILON_SCALE, EPSILON_CAP
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--epsilon-scale", type=float, default=1.0)
    probe.add_argument("--epsilon-cap", type=float, default=0.50)
    known, remaining = probe.parse_known_args()
    if float(known.epsilon_scale) < 0.0:
        raise SystemExit("epsilon-scale must be nonnegative")
    if not (0.0 <= float(known.epsilon_cap) < 1.0):
        raise SystemExit("epsilon-cap must be in [0,1)")
    EPSILON_SCALE = float(known.epsilon_scale)
    EPSILON_CAP = float(known.epsilon_cap)
    UncertaintyDampedPolicyMixture.instances = []

    import sys
    sys.argv = [sys.argv[0], "--out", str(known.out)] + remaining
    base.EnsembleAdvantagePolicy = UncertaintyDampedPolicyMixture
    rc = int(base.main())

    payload = json.loads(known.out.read_text(encoding="utf-8"))
    if payload.get("runner_failed_before_report"):
        return rc
    payload["schema"] = "SPINCORE_R7_3_POLICY_MIXTURE_UNCERTAINTY_DAMPING_V1"
    payload["ensemble_mapping"] = "REGRET_POLICY_MEAN_WITH_ENSEMBLE_DISAGREEMENT_ADAPTIVE_UNIFORM_DAMPING"
    payload["uncertainty_damping"] = {
        "disagreement_metric": "MEAN_MEMBER_TV_TO_ENSEMBLE_MEAN",
        "epsilon_scale": float(EPSILON_SCALE),
        "epsilon_cap": float(EPSILON_CAP),
        "formula": "epsilon=min(cap,scale*mean_member_tv_to_mean)",
        "intent": "Damp only epistemically unstable states instead of globally flattening all behavior.",
    }
    payload["uncertainty_runtime_statistics"] = _runtime_statistics(payload)
    payload["uncertainty_runtime_statistics_note"] = (
        "Counters cover calls after an Advantage ensemble is fitted; initial zero-regret uniform bootstrap calls are excluded. "
        "Instrumentation consumes no RNG and does not alter policy outputs."
    )
    payload["production_policy_mapping_changed"] = False
    payload["theoretical_equivalence_claimed"] = False
    payload["promotion_note"] = (
        "Algorithmic diagnostic only. Partial-exact sampling, deck schedule, member fitting and primary "
        "RNG stream remain unchanged. Damping is state-adaptive and uses only disagreement among the "
        "already-fitted regret policies. It is not recovered Deep CFR semantics and requires "
        "versioning plus strategic/checkpoint recertification if ever promoted."
    )
    known.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": payload["schema"],
        "ensemble_size": payload["ensemble_size"],
        "iterations": payload["iterations"],
        "uncertainty_damping": payload["uncertainty_damping"],
        "uncertainty_runtime_statistics": payload["uncertainty_runtime_statistics"],
        "cross_seed": payload["cross_seed"],
        "per_seed_fit_pass": payload["per_seed_fit_pass"],
        "r7_3_pass": payload["r7_3_pass"],
    }, indent=2, sort_keys=True), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
