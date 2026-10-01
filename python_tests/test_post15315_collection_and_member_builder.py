from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_post15315_rebuild_member as builder
import post15315_collection as collection

PROTOCOL = ROOT / "contracts" / "post15315_multirebuild_preregistration.json"


class DummySampler:
    def __init__(self, *, seed, config):
        self.seed = int(seed)
        self.i = 0

    def sample_episode(self, *, force_domain):
        out = {"episode": self.i, "domain": force_domain}
        self.i += 1
        return out


class DummyInner:
    def __init__(self, raw):
        self.raw = raw
        self.done = False

    def neural_bytes_v2(self):
        b = bytearray(113)
        b[112] = 0
        return bytes(b)

    def universal_legal_actions(self, active):
        return (0, 1, 9)

    def neural_bytes(self):
        return f"obs:{self.raw['episode']}:{self.raw['seed']}".encode("ascii")


class DummyState:
    def __init__(self, raw):
        self.inner = DummyInner(raw)

    @property
    def terminal(self):
        return self.inner.done

    def close(self):
        pass


class DummySolver:
    def create(self, ep, seed):
        return {"episode": int(ep["episode"]), "seed": int(seed)}


def patch_collection_runtime(monkeypatch):
    monkeypatch.setattr(collection.distill, "LegacyScenarioSampler", DummySampler)
    monkeypatch.setattr(collection.distill, "LegacyScenarioConfig", lambda: object())
    monkeypatch.setattr(collection.distill, "LeanSolverState", DummyState)
    class DummyActionSpec:
        @staticmethod
        def active_mask(street):
            return 0x3FF

    monkeypatch.setattr(
        collection.distill, "FIRST_RELEASE_ACTION_SPEC", DummyActionSpec()
    )
    monkeypatch.setattr(
        collection.distill,
        "semantic_sigma",
        lambda models, obs, legal: tuple(
            0.5 if i in (1, 9) else 0.0 for i in range(10)
        ),
    )
    monkeypatch.setattr(
        collection.distill,
        "legal_mask",
        lambda legal: tuple(i in set(legal) for i in range(10)),
    )
    monkeypatch.setattr(
        collection.distill,
        "sample_action",
        lambda sigma, legal, rng: int(legal[0]),
    )
    monkeypatch.setattr(
        collection.distill,
        "apply_lean",
        lambda inner, active, action: setattr(inner, "done", True),
    )


def load_protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_post15315_collection_records_explicit_lineage_and_split(monkeypatch):
    patch_collection_runtime(monkeypatch)
    train, hold, stats = collection.collect_fresh_split(
        DummySolver(),
        models=[],
        episodes=5,
        master_seed=153151001,
        sample_iteration=15315,
    )
    assert len(train) == 4
    assert len(hold) == 1
    assert stats["train_episodes"] == 4
    assert stats["holdout_episodes"] == 1
    assert stats["sample_iteration"] == 15315
    assert {int(x.iteration) for x in train + hold} == {15315}


def test_post15315_collection_is_deterministic_for_same_seed(monkeypatch):
    patch_collection_runtime(monkeypatch)
    a, ah, _ = collection.collect_fresh_split(
        DummySolver(), [], 7, master_seed=153151004, sample_iteration=15315
    )
    b, bh, _ = collection.collect_fresh_split(
        DummySolver(), [], 7, master_seed=153151004, sample_iteration=15315
    )
    assert collection.sample_digest(a) == collection.sample_digest(b)
    assert collection.sample_digest(ah) == collection.sample_digest(bh)


def test_post15315_strong_collection_keeps_only_predicate_matches(monkeypatch):
    patch_collection_runtime(monkeypatch)
    monkeypatch.setattr(
        collection.cal,
        "strong",
        lambda sample: b"obs:2:" in bytes(sample.observation)
        or b"obs:3:" in bytes(sample.observation),
    )
    train, hold, stats = collection.collect_strong_split(
        DummySolver(), [], 5, master_seed=153153001, sample_iteration=15315
    )
    assert len(train) == 2
    assert len(hold) == 0
    assert stats["kept_train_samples"] == 2
    assert stats["kept_holdout_samples"] == 0
    assert {int(x.iteration) for x in train} == {15315}


def test_member_seed_lookup_and_construction_minima_are_preregistered():
    p = load_protocol()
    m1 = builder.member_spec(p, 1)
    m8 = builder.member_spec(p, 8)
    m12 = builder.member_spec(p, 12)
    assert m1["base_collection_seed"] == 153151001
    assert m8["specialist_fit_seed"] == 153156008
    assert m12["fullpool_fit_seed"] == 153155012

    avg = p["rebuild_plan"]["average_policy"]
    assert avg["min_ordinary_train_samples"] == 20000
    assert avg["min_ordinary_holdout_samples"] == 5000
    assert avg["min_novel_strong_states"] == 700


def test_member_builder_uses_explicit_15315_collection_and_per_member_fit_seeds():
    source = (ROOT / "tools" / "build_post15315_rebuild_member.py").read_text(
        encoding="utf-8"
    )
    assert "import post15315_collection as collect" in source
    assert "sample_iteration=FINAL_ITERATION" in source
    assert 'seeds["tail_fit_seed"]' in source
    assert 'seeds["fullpool_fit_seed"]' in source
    assert 'seeds["specialist_fit_seed"]' in source
    assert "distill.MASTER_SEED" not in source
    assert "strength_benchmark_performed" in source
