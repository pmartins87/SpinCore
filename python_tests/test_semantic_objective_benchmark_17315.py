from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))

import benchmark_semantic_objectives_17315 as b


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v=torch.nn.Parameter(torch.zeros(10))
    def forward(self,batch):
        n=batch["legal"].shape[0]
        return self.v.unsqueeze(0).expand(n,-1)


def test_balanced_objective_runs_and_is_finite():
    model=Tiny()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    batch={"legal":torch.tensor([[1,1,1,0,0,0,0,0,0,0]],dtype=torch.bool)}
    target=torch.tensor([[1.0,-1.0,-2.0,0,0,0,0,0,0,0]])
    weights=torch.tensor([1.0])
    loss=b.objective_step(
        model,opt,batch,target,weights,"C_POSNEG_BALANCED_MSE"
    )
    assert loss>0.0
    assert torch.isfinite(torch.tensor(loss))


def test_normalized_balanced_objective_runs():
    model=Tiny()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    batch={"legal":torch.tensor([[1,1,1,1,0,0,0,0,0,0]],dtype=torch.bool)}
    target=torch.tensor([[0.2,-4.0,-2.0,1.0,0,0,0,0,0,0]])
    weights=torch.tensor([2.0])
    loss=b.objective_step(
        model,opt,batch,target,weights,"D_NORMALIZED_POSNEG_BALANCED_MSE"
    )
    assert loss>0.0


def test_state_scale_normalization_is_policy_invariant_for_exact_target():
    legal=(0,1,2)
    raw=[0.2,-4.0,1.0]+[0.0]*7
    scaled=[x/4.0 for x in raw]
    p1=b.rm(raw,legal)
    p2=b.rm(scaled,legal)
    assert max(abs(float(x)-float(y)) for x,y in zip(p1,p2))<1e-12


def test_adjudicate_requires_all_frozen_checks():
    A={
        "holdout":{
            "weighted_mean_tv":0.5,
            "weighted_cross_entropy":5.0,
            "weighted_expected_stored_target_advantage":0.02,
            "weighted_target_argmax_agreement":0.6,
            "own_target_positive_fallback_count":100,
        },
        "fixed_primary":{"residual_ensemble_fallback_fraction":1.0},
    }
    good={
        "holdout":{
            "weighted_mean_tv":0.499,
            "weighted_cross_entropy":4.9,
            "weighted_expected_stored_target_advantage":0.0195,
            "weighted_target_argmax_agreement":0.6,
            "own_target_positive_fallback_count":50,
        },
        "fixed_primary":{"residual_ensemble_fallback_fraction":0.4},
    }
    bad={**good,"holdout":dict(good["holdout"])}
    bad["holdout"]["weighted_expected_stored_target_advantage"]=0.018
    results={
        "A_CURRENT_WEIGHTED_MSE_REFIT":A,
        "B_STATE_SCALE_NORMALIZED_MSE":good,
        "C_POSNEG_BALANCED_MSE":bad,
        "D_NORMALIZED_POSNEG_BALANCED_MSE":bad,
    }
    d=b.adjudicate(results)
    assert d["classification"]=="OBJECTIVE_CANDIDATE_IDENTIFIED"
    assert d["passing_candidates"]==["B_STATE_SCALE_NORMALIZED_MSE"]


def test_semantic_factory_resolves_and_is_seed_deterministic():
    from run_3h_v1_semantic_shadow_10105 import V1SemanticAdvantageNet
    a=b.make_semantic_model(17315)
    c=b.make_semantic_model(17315)
    assert isinstance(a,V1SemanticAdvantageNet)
    assert all(torch.equal(a.state_dict()[key],c.state_dict()[key]) for key in a.state_dict())
