from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_r7_3_partial_exact_ensemble_paired as base
from spincore.deep_cfr import regret_matching_policy, uniform_policy
from spincore_nn.codec import collate_inputs, decode_spnniv1


CURRENT_WEIGHT = 0.50


class TemporalBlendPolicyMixture:
    """Blend the current regret-policy ensemble with the prior fitted ensemble.

    For the first feedback transition there is no prior fitted policy, so the
    reference side is the exact zero-regret uniform policy. On later transitions
    it is the previous iteration's size-N policy mixture. This explicitly damps
    abrupt fitted-policy changes while retaining regret-derived behavior.
    """

    def __init__(self, *, device: str):
        self.device = device
        self.current_models: list[torch.nn.Module] = []
        self.previous_models: list[torch.nn.Module] = []
        self.fit_generation = 0

    @property
    def ready(self) -> bool:
        return bool(self.current_models)

    @property
    def models(self):
        return self.current_models

    @models.setter
    def models(self, value):
        new_models = list(value)
        if not new_models:
            self.current_models = []
            self.previous_models = []
            self.fit_generation = 0
            return
        if self.current_models:
            self.previous_models = list(self.current_models)
        self.current_models = new_models
        self.fit_generation += 1

    def _mixture(self, models, observation: bytes, legal: tuple[int, ...]):
        batch = collate_inputs([decode_spnniv1(observation)], device=self.device)
        policies = []
        with torch.no_grad():
            for model in models:
                model.eval()
                raw = model(batch)[0].detach().cpu().tolist()
                policies.append(regret_matching_policy(raw, legal))
        out = [0.0] * 6
        for action in range(6):
            out[action] = sum(float(p[action]) for p in policies) / float(len(policies))
        return tuple(out)

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if not self.current_models:
            return uniform_policy(state, observation, legal)
        current = self._mixture(self.current_models, observation, legal)
        if self.previous_models:
            reference = self._mixture(self.previous_models, observation, legal)
        else:
            reference = uniform_policy(state, observation, legal)
        w = float(CURRENT_WEIGHT)
        return tuple(w * float(current[a]) + (1.0 - w) * float(reference[a]) for a in range(6))


def main() -> int:
    global CURRENT_WEIGHT
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--current-weight", type=float, default=0.50)
    known, remaining = probe.parse_known_args()
    if not (0.0 < float(known.current_weight) <= 1.0):
        raise SystemExit("current-weight must be in (0,1]")
    CURRENT_WEIGHT = float(known.current_weight)

    import sys
    sys.argv = [sys.argv[0], "--out", str(known.out)] + remaining
    base.EnsembleAdvantagePolicy = TemporalBlendPolicyMixture
    rc = int(base.main())

    payload = json.loads(known.out.read_text(encoding="utf-8"))
    if payload.get("runner_failed_before_report"):
        return rc
    payload["schema"] = "SPINCORE_R7_3_POLICY_MIXTURE_TEMPORAL_BLEND_V1"
    payload["ensemble_mapping"] = "TEMPORAL_BLEND_OF_CURRENT_AND_PREVIOUS_REGRET_POLICY_MIXTURES"
    payload["temporal_blend"] = {
        "current_policy_weight": float(CURRENT_WEIGHT),
        "previous_policy_weight": float(1.0 - CURRENT_WEIGHT),
        "first_feedback_reference": "EXACT_ZERO_REGRET_UNIFORM_POLICY",
        "later_feedback_reference": "PREVIOUS_ITERATION_POLICY_MIXTURE",
    }
    payload["production_policy_mapping_changed"] = False
    payload["theoretical_equivalence_claimed"] = False
    payload["promotion_note"] = (
        "Algorithmic diagnostic only. Partial-exact sampling, member fitting, deck schedule and the "
        "primary live RNG stream are unchanged. Only the behavior fed into the next CFR collection is "
        "temporally blended. This tests whether abrupt iteration-to-iteration regret-policy changes are "
        "the mechanism behind the observed five-iteration policy-mixture decay."
    )
    known.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": payload["schema"],
        "ensemble_size": payload["ensemble_size"],
        "iterations": payload["iterations"],
        "temporal_blend": payload["temporal_blend"],
        "cross_seed": payload["cross_seed"],
        "per_seed_fit_pass": payload["per_seed_fit_pass"],
        "r7_3_pass": payload["r7_3_pass"],
    }, indent=2, sort_keys=True), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
