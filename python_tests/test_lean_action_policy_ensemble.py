from __future__ import annotations

import torch

from spincore.lean_action_policy import (
    LeanEnsembleActionAdvantagePolicy,
    LeanNeuralActionAdvantagePolicy,
)


class _Fixed(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, batch):
        n = next(iter(batch.values())).shape[0]
        return self.values.unsqueeze(0).repeat(n, 1)


def test_single_member_ensemble_matches_single_policy(monkeypatch):
    # Avoid depending on a real encoded observation; only the raw-output
    # aggregation semantics are under test here.
    import spincore.lean_action_policy as mod

    monkeypatch.setattr(
        mod,
        "collate_action_observations",
        lambda *args, **kwargs: {"x": torch.zeros((1, 1), dtype=torch.float32)},
    )
    model=_Fixed([0.0,1.0,0.0,2.0,0.0,0.0,0.0,0.0,0.0,3.0])
    legal=(0,1,3,9)
    single=LeanNeuralActionAdvantagePolicy(model)
    ensemble=LeanEnsembleActionAdvantagePolicy([model])
    assert ensemble(None,b"obs",legal)==single(None,b"obs",legal)


def test_two_member_ensemble_averages_raw_before_regret_matching(monkeypatch):
    import spincore.lean_action_policy as mod

    monkeypatch.setattr(
        mod,
        "collate_action_observations",
        lambda *args, **kwargs: {"x": torch.zeros((1, 1), dtype=torch.float32)},
    )
    a=_Fixed([0.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    b=_Fixed([0.0,0.0,0.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0])
    policy=LeanEnsembleActionAdvantagePolicy([a,b])
    out=policy(None,b"obs",(1,3))
    assert abs(out[1]-0.5)<1e-9
    assert abs(out[3]-0.5)<1e-9
