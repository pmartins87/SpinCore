"""Vectorized SPNNIV1 batching; no changes to reservoir or sampling semantics.

Legacy provenance: DeepSpin buffers.py/sample_batch and trainer.py used array
batches. Here we retain the LT1 wire/checkpoint and Python random.sample stream,
vectorizing only byte decoding and tensor construction. Full 32 history slots
are preserved (including padding); history_len does not trim the GRU input.
"""
from __future__ import annotations

import numpy as np
import torch


def vectorized_batch(samples, device="cpu"):
    if not samples:
        raise ValueError("empty vectorized batch")
    observations = [s.observation for s in samples]
    if any(len(x) != 126 or x[:8] != b"SPNNIV1\x00" for x in observations):
        raise ValueError("bad SPNNIV1 payload")
    if any(len(s.legal) != 10 or len(s.target) != 10 for s in samples):
        raise ValueError("expected ten action slots")
    raw = np.frombuffer(b"".join(observations), dtype=np.uint8).reshape(-1, 126)
    if np.any(raw[:, 93] > 32):
        raise ValueError("bad history length")

    def tensor(values, dtype):
        return torch.from_numpy(np.array(values, dtype=dtype, copy=True, order="C")).to(device)

    numeric = raw[:, 15:79].copy().view("<f4").reshape(-1, 16)
    batch = {
        "cards": tensor(raw[:, 8:15], np.int64),
        "numeric": tensor(numeric, np.float32),
        "categorical": tensor(raw[:, 79:87], np.int64),
        "legal": tensor([s.legal for s in samples], np.bool_),
        "history_len": tensor(raw[:, 93], np.int64),
        "history": tensor(raw[:, 94:126], np.int64),
    }
    return batch, tensor([s.target for s in samples], np.float32), tensor([s.weight for s in samples], np.float32)
