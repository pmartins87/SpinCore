from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_r7_3_partial_exact_ensemble_paired as base
from run_r7_3_partial_exact_policy_mixture_paired import PolicyMixtureEnsembleAdvantagePolicy
from spincore.deep_cfr import uniform_policy


EPSILON0 = 0.30
DECAY = 0.50


class DecayingTremblePolicyMixture(PolicyMixtureEnsembleAdvantagePolicy):
    """Policy-mixture behavior with an explicit vanishing uniform tremble.

    After fit k, epsilon_k = epsilon0 * decay**(k-1). The first feedback
    transition therefore receives the strongest damping and the intervention
    rapidly vanishes. This is an algorithmic diagnostic, not recovered Deep CFR
    semantics and not a production promotion.
    """

    def __init__(self, *, device: str):
        self._models = []
        self.fit_generation = 0
        super().__init__(device=device)

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        self._models = list(value)
        if self._models:
            self.fit_generation += 1

    def epsilon(self) -> float:
        if self.fit_generation <= 0:
            return 1.0
        return float(EPSILON0) * (float(DECAY) ** float(self.fit_generation - 1))

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if not self._models:
            return uniform_policy(state, observation, legal)
        base_policy = super().__call__(state, observation, legal)
        uniform = uniform_policy(state, observation, legal)
        eps = self.epsilon()
        return tuple((1.0 - eps) * float(base_policy[a]) + eps * float(uniform[a]) for a in range(6))


def main() -> int:
    global EPSILON0, DECAY
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--epsilon0", type=float, default=0.30)
    probe.add_argument("--epsilon-decay", type=float, default=0.50)
    known, remaining = probe.parse_known_args()
    if not (0.0 <= float(known.epsilon0) < 1.0):
        raise SystemExit("epsilon0 must be in [0,1)")
    if not (0.0 <= float(known.epsilon_decay) <= 1.0):
        raise SystemExit("epsilon-decay must be in [0,1]")
    EPSILON0 = float(known.epsilon0)
    DECAY = float(known.epsilon_decay)

    # Remove wrapper-only arguments before delegating to the authoritative
    # paired runner's parser.
    import sys
    sys.argv = [sys.argv[0]] + remaining
    base.EnsembleAdvantagePolicy = DecayingTremblePolicyMixture
    rc = int(base.main())

    payload = json.loads(known.out.read_text(encoding="utf-8"))
    if payload.get("runner_failed_before_report"):
        return rc
    payload["schema"] = "SPINCORE_R7_3_POLICY_MIXTURE_DECAY_TREMBLE_V1"
    payload["ensemble_mapping"] = "REGRET_MATCH_EACH_MEMBER_THEN_AVERAGE_POLICY_THEN_UNIFORM_TREMBLE"
    payload["tremble_schedule"] = {
        "epsilon0_after_first_fit": float(EPSILON0),
        "geometric_decay_per_fit": float(DECAY),
        "epsilon_by_feedback_transition": [
            float(EPSILON0) * (float(DECAY) ** i)
            for i in range(max(int(payload.get("iterations", 1)) - 1, 0))
        ],
        "purpose": "Damp the first Advantage-fit feedback transition most strongly and rapidly vanish the intervention."
    }
    payload["production_policy_mapping_changed"] = False
    payload["theoretical_equivalence_claimed"] = False
    payload["promotion_note"] = (
        "Algorithmic diagnostic only. The underlying partial-exact estimator, member fits, deck schedule "
        "and primary RNG stream are unchanged. A uniform tremble is mixed into the size-4 regret-policy "
        "ensemble only when it drives subsequent CFR collection, with geometric decay after each fit. "
        "No production or acceptance semantics change without a separate versioning/strategic audit."
    )
    known.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": payload["schema"],
        "ensemble_size": payload["ensemble_size"],
        "iterations": payload["iterations"],
        "tremble_schedule": payload["tremble_schedule"],
        "cross_seed": payload["cross_seed"],
        "per_seed_fit_pass": payload["per_seed_fit_pass"],
        "r7_3_pass": payload["r7_3_pass"],
    }, indent=2, sort_keys=True), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
