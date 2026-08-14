from __future__ import annotations

from dataclasses import asdict

import torch

import r7_5_fit_representation_candidate as base
from spincore_nn.codec import decode_spnniv1
from spincore_nn.codec_v2 import decode_spnniv2
from spincore_nn.hybrid_v3 import HybridNetworkConfigV3, build_hybrid_input, collate_hybrid_inputs
from spincore_nn.hybrid_v3_pruned import HybridCandidateNetV3


CANDIDATES = {
    "H0_FIXED_V1": "HYBRID_V3_H0",
    "H1_RELATIONAL_EXACT": "HYBRID_V3_H1",
    "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY": "HYBRID_V3_H2",
    "H3_HYBRID_EXACT_SEMANTIC": "HYBRID_V3_H3",
    "H4_HYBRID_CAPACITY": "HYBRID_V3_H4",
}

# These counts were mechanically verified by the pruned Phase-0 property
# preflight. H0 intentionally equals frozen V1 exactly.
EXPECTED_PARAMETER_COUNT = {
    "H0_FIXED_V1": 152438,
    "H1_RELATIONAL_EXACT": 160502,
    "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY": 163898,
    "H3_HYBRID_EXACT_SEMANTIC": 247590,
    "H4_HYBRID_CAPACITY": 362342,
}


class _ConfigReport:
    def __init__(self, candidate: str):
        cfg = HybridNetworkConfigV3()
        if candidate == "H4_HYBRID_CAPACITY":
            from dataclasses import replace
            cfg = replace(cfg, hidden=448, head_hidden=192)
        self.candidate = candidate
        self.cfg = cfg

    def to_dict(self):
        return {"candidate": self.candidate, **asdict(self.cfg)}


def _batch(samples, candidate: str, device: str):
    items = [
        build_hybrid_input(
            decode_spnniv1(sample.observation_v1),
            decode_spnniv2(sample.observation_v2),
        )
        for sample in samples
    ]
    batch = collate_hybrid_inputs(items, device=device)
    target = torch.tensor([sample.target for sample in samples], dtype=torch.float32, device=device)
    weights = torch.tensor([sample.weight for sample in samples], dtype=torch.float32, device=device)
    return batch, target, weights


def _models(candidate: str, device: str, fit_seed: int):
    torch.manual_seed(int(fit_seed))
    advantage = HybridCandidateNetV3(candidate).to(device)
    torch.manual_seed(int(fit_seed) ^ 0x5A17C0DE)
    policy = HybridCandidateNetV3(candidate).to(device)
    return _ConfigReport(candidate), advantage, policy


def main() -> int:
    # Reuse the frozen R7.5.3 benchmark implementation for loading, exact row
    # multiplicity/order, deterministic split, optimizer schedule, metrics,
    # sentinels, resource accounting and output schema.  Only candidate batch
    # construction and network factory are replaced here.
    base.SCHEMA = "SPINCORE_R7_5_3C_HYBRID_DIAGNOSTIC_FIT_V1"
    base.CANDIDATES = CANDIDATES
    base.EXPECTED_PARAMETER_COUNT = EXPECTED_PARAMETER_COUNT
    base._batch = _batch
    base._models = _models
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
