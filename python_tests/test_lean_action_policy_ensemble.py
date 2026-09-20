from __future__ import annotations

import unittest
from unittest.mock import patch

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


def _fake_collate(*args, **kwargs):
    return {"x": torch.zeros((1, 1), dtype=torch.float32)}


class LeanActionPolicyEnsembleTests(unittest.TestCase):
    def test_single_member_ensemble_matches_single_policy(self):
        model=_Fixed([0.0,1.0,0.0,2.0,0.0,0.0,0.0,0.0,0.0,3.0])
        legal=(0,1,3,9)
        with patch("spincore.lean_action_policy.collate_action_observations", _fake_collate):
            single=LeanNeuralActionAdvantagePolicy(model)
            ensemble=LeanEnsembleActionAdvantagePolicy([model])
            self.assertEqual(
                ensemble(None,b"obs",legal),
                single(None,b"obs",legal),
            )

    def test_two_member_ensemble_averages_raw_before_regret_matching(self):
        a=_Fixed([0.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        b=_Fixed([0.0,0.0,0.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0])
        with patch("spincore.lean_action_policy.collate_action_observations", _fake_collate):
            policy=LeanEnsembleActionAdvantagePolicy([a,b])
            out=policy(None,b"obs",(1,3))
            self.assertAlmostEqual(out[1],0.5,places=9)
            self.assertAlmostEqual(out[3],0.5,places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
