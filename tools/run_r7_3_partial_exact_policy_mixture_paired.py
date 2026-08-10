from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_r7_3_partial_exact_ensemble_paired as base
from spincore.deep_cfr import regret_matching_policy, uniform_policy
from spincore_nn.codec import collate_inputs, decode_spnniv1


class PolicyMixtureEnsembleAdvantagePolicy:
    """Average member regret-matching policies, not raw Advantage predictions.

    Same-memory mapping evidence showed that the nonlinear hard-regret tail is
    materially smaller when each independently fitted member is mapped first
    and legal-action probabilities are then averaged.  This remains a
    diagnostic behavior mapping; production semantics are unchanged.
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
        policies: list[list[float] | tuple[float, ...]] = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                raw = model(batch)[0].detach().cpu().tolist()
                policies.append(regret_matching_policy(raw, legal))
        out = [0.0] * 6
        for action in range(6):
            out[action] = sum(float(p[action]) for p in policies) / float(len(policies))
        return tuple(out)


def main() -> int:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--ensemble-size", type=int, required=True)
    known, _ = probe.parse_known_args()
    if int(known.ensemble_size) < 2:
        raise SystemExit("policy-mixture candidate requires ensemble-size >= 2")

    # run_seed() resolves this global from the base module at execution time.
    # All collection, deck, primary-RNG and side-member training semantics stay
    # byte-for-byte in the paired base runner; only the ensemble behavior map is
    # replaced for this controlled candidate.
    base.EnsembleAdvantagePolicy = PolicyMixtureEnsembleAdvantagePolicy
    rc = int(base.main())

    payload = json.loads(known.out.read_text(encoding="utf-8"))
    if payload.get("runner_failed_before_report"):
        return rc
    payload["schema"] = "SPINCORE_R7_3_PARTIAL_EXACT_POLICY_MIXTURE_PAIRED_V1"
    payload["ensemble_mapping"] = "REGRET_MATCH_EACH_MEMBER_THEN_AVERAGE_POLICY"
    payload["raw_advantage_averaging_used"] = False
    payload["production_policy_mapping_changed"] = False
    payload["mapping_evidence_reference"] = {
        "file": "validation/R7_3_ADVANTAGE_ENSEMBLE_MAPPING_256.json",
        "same_memory_size4_raw_mean_tv": 0.11909729169343924,
        "same_memory_size4_raw_p95_tv": 0.4724242687225342,
        "same_memory_size4_policy_mixture_mean_tv": 0.11735808003423573,
        "same_memory_size4_policy_mixture_p95_tv": 0.36031225323677063,
        "policy_to_raw_p95_ratio": 0.7626878572751529,
    }
    payload["promotion_note"] = (
        "Experimental paired candidate only. It preserves the paired runner's "
        "authoritative deck schedule and primary coupled RNG stream, but changes "
        "the multi-model behavior mapping from mean(raw Advantage)->hard regret "
        "matching to hard regret matching per member->mean(policy). Production "
        "semantics remain unchanged pending physical evidence and recertification."
    )
    known.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": payload["schema"],
        "ensemble_size": payload["ensemble_size"],
        "ensemble_mapping": payload["ensemble_mapping"],
        "cross_seed": payload["cross_seed"],
        "per_seed_fit_pass": payload["per_seed_fit_pass"],
        "r7_3_pass": payload["r7_3_pass"],
    }, indent=2, sort_keys=True), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
