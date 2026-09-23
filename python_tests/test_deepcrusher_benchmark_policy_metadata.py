import spincore.lean_solver_actions as lean_solver_actions
from spincore.deepcrusher_benchmark import SpinCoreCheckpointPolicy


class _FakeAgent:
    @staticmethod
    def distribution(state):
        del state
        probs=[0.0]*10
        probs[0]=0.25
        probs[9]=0.75
        return (0x203,(0,9),tuple(probs))

    @staticmethod
    def domain_for_state(state):
        del state
        return "THREE_HANDED"


class _FixedRng:
    @staticmethod
    def random():
        return 0.80


def test_spincore_benchmark_policy_exposes_sampled_slot_metadata(monkeypatch):
    monkeypatch.setattr(
        lean_solver_actions,
        "resolve_lean_exact",
        lambda state, active_mask, slot: (5,0),
    )
    policy=SpinCoreCheckpointPolicy(_FakeAgent())
    action=policy.choose_exact(object(),seat=1,rng=_FixedRng())

    assert action.action_type==5
    assert action.amount_to==0
    detail=policy.decision_metadata(seat=1)
    assert detail is not None
    assert detail["selection"]=="SAMPLED_POLICY"
    assert detail["domain"]=="THREE_HANDED"
    assert detail["sample_u"]==0.80
    assert detail["selected_slot"]==9
    assert detail["selected_slot_name"]=="ALL_IN"
    assert detail["selected_probability"]==0.75
    assert detail["resolved_action_type"]==5
    assert detail["resolved_amount_to"]==0
    assert detail["legal_actions"]==[
        {"slot":0,"name":"FOLD","probability":0.25},
        {"slot":9,"name":"ALL_IN","probability":0.75},
    ]
