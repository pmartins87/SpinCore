import pytest

from spincore.strategic_sentinel import (
    IntegrityExpectation,
    PlausibilityRule,
    SentinelActionObservation,
    evaluate_strategic_sentinels,
)


MODEL = "a" * 64
OBS = "b" * 64
PROFILE = "spinprofile-v3:test"


def _observation(*, p0=0.25, p1=0.75, model=MODEL, sentinel_id="hu_extreme_aa"):
    return SentinelActionObservation(
        sentinel_id=sentinel_id,
        profile_id=PROFILE,
        domain="TRUE_HEADS_UP",
        model_sha256=model,
        observation_sha256=OBS,
        legal_actions=(0, 1),
        policy=(p0, p1, 0.0, 0.0, 0.0, 0.0),
    )


def test_action_fingerprint_is_exact_and_sensitive_to_policy_and_model():
    a = _observation()
    b = _observation()
    c = _observation(p0=0.25000000000000006, p1=0.7499999999999999)
    d = _observation(model="c" * 64)
    assert a.action_fingerprint == b.action_fingerprint
    assert a.action_fingerprint != c.action_fingerprint
    assert a.action_fingerprint != d.action_fingerprint


def test_observation_rejects_illegal_mass_and_bad_normalization():
    with pytest.raises(ValueError, match="illegal action"):
        SentinelActionObservation(
            "x", PROFILE, "TRUE_HEADS_UP", MODEL, OBS, (0, 1),
            (0.4, 0.5, 0.1, 0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="sum to one"):
        SentinelActionObservation(
            "x", PROFILE, "TRUE_HEADS_UP", MODEL, OBS, (0, 1),
            (0.4, 0.5, 0.0, 0.0, 0.0, 0.0),
        )


def test_integrity_only_never_passes_release_gate():
    obs = _observation()
    out = evaluate_strategic_sentinels(
        observations=[obs],
        required_sentinel_ids=[obs.sentinel_id],
        integrity_expectations=[IntegrityExpectation(obs.sentinel_id, obs.action_fingerprint)],
        plausibility_rules=[],
    )
    assert out["integrity"]["pass"] is True
    assert out["plausibility"]["complete"] is False
    assert out["plausibility"]["pass"] is False
    assert out["strategic_sentinel_gate_pass"] is False
    assert out["integrity_only_can_authorize_release"] is False
    assert out["ready_for_tables"] is False


def test_complete_integrity_and_plausibility_can_pass_sentinel_gate_only():
    obs = _observation()
    out = evaluate_strategic_sentinels(
        observations=[obs],
        required_sentinel_ids=[obs.sentinel_id],
        integrity_expectations=[IntegrityExpectation(obs.sentinel_id, obs.action_fingerprint)],
        plausibility_rules=[
            PlausibilityRule(
                sentinel_id=obs.sentinel_id,
                action=1,
                min_probability=0.70,
                rationale="precommitted example bound for test only",
            )
        ],
    )
    assert out["integrity"]["pass"] is True
    assert out["plausibility"]["pass"] is True
    assert out["strategic_sentinel_gate_pass"] is True
    assert out["ready_for_tables"] is False


def test_plausibility_violation_fails_closed():
    obs = _observation()
    out = evaluate_strategic_sentinels(
        observations=[obs],
        required_sentinel_ids=[obs.sentinel_id],
        integrity_expectations=[IntegrityExpectation(obs.sentinel_id, obs.action_fingerprint)],
        plausibility_rules=[
            PlausibilityRule(
                sentinel_id=obs.sentinel_id,
                action=1,
                min_probability=0.90,
                rationale="deliberately violated test bound",
            )
        ],
    )
    assert out["integrity"]["pass"] is True
    assert out["plausibility"]["pass"] is False
    assert out["strategic_sentinel_gate_pass"] is False


def test_changed_runtime_or_model_fingerprint_breaks_integrity():
    baseline = _observation()
    changed = _observation(model="c" * 64)
    out = evaluate_strategic_sentinels(
        observations=[changed],
        required_sentinel_ids=[changed.sentinel_id],
        integrity_expectations=[
            IntegrityExpectation(changed.sentinel_id, baseline.action_fingerprint)
        ],
        plausibility_rules=[
            PlausibilityRule(
                sentinel_id=changed.sentinel_id,
                action=1,
                min_probability=0.0,
                rationale="coverage only",
            )
        ],
    )
    assert out["integrity"]["pass"] is False
    assert out["strategic_sentinel_gate_pass"] is False


def test_missing_required_sentinel_fails_closed():
    obs = _observation()
    out = evaluate_strategic_sentinels(
        observations=[],
        required_sentinel_ids=[obs.sentinel_id],
        integrity_expectations=[IntegrityExpectation(obs.sentinel_id, obs.action_fingerprint)],
        plausibility_rules=[
            PlausibilityRule(
                sentinel_id=obs.sentinel_id,
                action=1,
                min_probability=0.0,
                rationale="coverage only",
            )
        ],
    )
    assert out["missing_observations"] == [obs.sentinel_id]
    assert out["integrity"]["pass"] is False
    assert out["plausibility"]["pass"] is False
    assert out["strategic_sentinel_gate_pass"] is False


def test_unknown_plausibility_sentinel_is_configuration_error():
    obs = _observation()
    with pytest.raises(ValueError, match="non-required sentinels"):
        evaluate_strategic_sentinels(
            observations=[obs],
            required_sentinel_ids=[obs.sentinel_id],
            integrity_expectations=[IntegrityExpectation(obs.sentinel_id, obs.action_fingerprint)],
            plausibility_rules=[
                PlausibilityRule(
                    sentinel_id="other",
                    action=1,
                    min_probability=0.0,
                    rationale="bad config",
                )
            ],
        )
