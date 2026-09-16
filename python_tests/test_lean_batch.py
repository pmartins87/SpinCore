from dataclasses import replace
import copy
import random
import struct

import pytest
import torch

from spincore.r7_5_action_cfr import ActionAdvantageSample
from spincore_nn.action_models import collate_action_observations, make_action_models
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step


def samples(n=37):
    rng = random.Random(20260916)
    result = []
    for i in range(n):
        raw = bytearray(126)
        raw[:8] = b"SPNNIV1\x00"
        raw[8:15] = bytes(rng.randrange(53) for _ in range(7))
        struct.pack_into('<16f', raw, 15, *[rng.uniform(-100, 100) for _ in range(16)])
        raw[79:87] = bytes(rng.randrange(32) for _ in range(8))
        raw[87:93] = bytes((0, 1, 0, 1, 0, 1))  # Must not become action mask.
        raw[93] = i % 33
        raw[94:126] = bytes(rng.randrange(64) for _ in range(32))
        legal = (1, 1, 0, 1, 0, 1, 0, 1, 1, 1)
        target = tuple(rng.uniform(-2, 2) if x else 0 for x in legal)
        result.append(ActionAdvantageSample(bytes(raw), legal, target, float(i+1), i+1))
    return result


def reference(s):
    batch = collate_action_observations('C0_V1_FROZEN_CONTROL', [x.observation for x in s], [x.legal for x in s])
    return batch, torch.tensor([x.target for x in s]), torch.tensor([x.weight for x in s])


@pytest.mark.parametrize('n', [1, 37, 1024])
def test_wire_tensor_parity(n):
    a, z = reference(samples(n)), vectorized_batch(samples(n))
    for key in a[0]:
        assert torch.equal(a[0][key], z[0][key]), key
    assert torch.equal(a[1], z[1])
    assert torch.equal(a[2], z[2])


@pytest.mark.parametrize('kind', ['advantage', 'strategy'])
def test_optimizer_parity(kind):
    torch.set_num_threads(1)
    s = samples()
    if kind == 'strategy':
        s = [replace(x, target=tuple(v / sum(x.legal) for v in x.legal)) for x in s]
    a, z = reference(s), vectorized_batch(s)
    _, model, _ = make_action_models('C0_V1_FROZEN_CONTROL', advantage_seed=31, policy_seed=32)
    other = copy.deepcopy(model)
    opt, opt2 = torch.optim.Adam(model.parameters()), torch.optim.Adam(other.parameters())
    for _ in range(3):
        assert train_step(model, opt, *a, kind) == train_step(other, opt2, *z, kind)
    for x, y in zip(model.parameters(), other.parameters()):
        assert torch.equal(x, y)
    for x, y in zip(opt.state.values(), opt2.state.values()):
        for k in x:
            assert torch.equal(x[k], y[k])


@pytest.mark.parametrize('bad', [b'', b'X'*126, b'SPNNIV1\x00'+b'\x00'*85+b'\x21'+b'\x00'*32])
def test_rejects_bad_payload(bad):
    with pytest.raises(ValueError):
        vectorized_batch([replace(samples(1)[0], observation=bad)])
