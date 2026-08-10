from pathlib import Path

import torch

from spincore.solver import Episode, SolverLibrary
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn import AdvantageNet, AveragePolicyNet, NetworkConfig

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"


def obs():
    library = SolverLibrary(LIB)
    state = library.create(Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,)), 2)
    raw = state.neural_bytes()
    state.close()
    return raw


def test_codec_shape_and_privacy():
    item = decode_spnniv1(obs())
    assert len(item.cards) == 7
    assert sum(card > 0 for card in item.cards) == 2
    assert len(item.numeric) == 16
    assert len(item.legal) == 6


def test_collate_and_network_shapes():
    item = decode_spnniv1(obs())
    batch = collate_inputs([item, item])
    cfg = NetworkConfig(card_emb=4, cat_emb=3, hidden=20, gru_hidden=8, head_hidden=12)
    advantage = AdvantageNet(cfg)
    policy = AveragePolicyNet(cfg)
    assert advantage(batch).shape == (2, 6)
    probs = policy.probabilities(batch)
    assert probs.shape == (2, 6)
    assert torch.allclose(probs.sum(1), torch.ones(2), atol=1e-6)


def test_default_network_matches_recovered_r4_scale():
    cfg = NetworkConfig()
    advantage = AdvantageNet(cfg)
    policy = AveragePolicyNet(cfg)
    advantage_params = sum(p.numel() for p in advantage.parameters())
    policy_params = sum(p.numel() for p in policy.parameters())
    assert advantage_params == 152438
    assert policy_params == 152438
