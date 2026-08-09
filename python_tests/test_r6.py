from __future__ import annotations
import random
from pathlib import Path

from spincore.solver import Episode, SolverLibrary
from spincore.deep_cfr import ExternalSamplingCollector, chip_delta_utility, uniform_policy
from spincore_nn.reservoir import UniformReservoir


def _lib() -> SolverLibrary:
    root=Path(__file__).resolve().parents[1]
    candidates=[root/'build_r6'/'libspincore_solver_c.so', root/'build_r6'/'Release'/'spincore_solver_c.dll', root/'build_r6'/'spincore_solver_c.dll']
    for p in candidates:
        if p.exists(): return SolverLibrary(p)
    raise RuntimeError('R6 solver C ABI library not built')


def _hu_root(seed=123):
    lib=_lib()
    ep=Episode(1500,True,0,10,20,(0,750,750),1)
    return lib,lib.create(ep,seed)


def test_r6_external_sampling_advantage_targets_are_centered():
    _,root=_hu_root(777)
    adv=UniformReservoir(10000,1); strat=UniformReservoir(10000,2)
    c=ExternalSamplingCollector(policy=uniform_policy,terminal_utility=chip_delta_utility,rng=random.Random(3),advantage_memory=adv,strategy_memory=strat)
    try:
        result=c.collect_advantage(root,traverser=root.actor,iteration=1)
    finally: root.close()
    assert result.samples_added>0 and adv.items
    for s in adv.items:
        legal=[i for i,x in enumerate(s.legal) if x]
        assert legal
        assert abs(sum(s.target[a] for a in legal)/len(legal)) < 1e-6
        assert all(s.target[a]==0.0 for a,x in enumerate(s.legal) if not x)


def test_r6_own_reach_records_legal_normalized_policies():
    _,root=_hu_root(999)
    target=root.actor
    adv=UniformReservoir(10000,4); strat=UniformReservoir(10000,5)
    c=ExternalSamplingCollector(policy=uniform_policy,terminal_utility=chip_delta_utility,rng=random.Random(6),advantage_memory=adv,strategy_memory=strat)
    try: added=c.collect_strategy_own_reach(root,target_player=target,iteration=3)
    finally: root.close()
    assert added==len(strat.items) and added>0
    for s in strat.items:
        assert s.iteration==3 and s.weight==3.0
        assert abs(sum(s.target)-1.0)<1e-7
        for a,x in enumerate(s.legal):
            if not x: assert s.target[a]==0.0


def test_r6_collector_is_reproducible_from_same_seed():
    def once():
        _,root=_hu_root(2026)
        adv=UniformReservoir(10000,7); strat=UniformReservoir(10000,8)
        c=ExternalSamplingCollector(policy=uniform_policy,terminal_utility=chip_delta_utility,rng=random.Random(9),advantage_memory=adv,strategy_memory=strat)
        try: r=c.collect_advantage(root,traverser=root.actor,iteration=2)
        finally: root.close()
        return r,adv.items
    a=once(); b=once(); assert a==b


def test_r6_neural_domain_session_collects_and_trains():
    import torch
    from spincore.deep_cfr import DeepCFRDomainSession
    from spincore_nn import NetworkConfig,AdvantageNet,AveragePolicyNet,DomainBundle

    torch.manual_seed(123)
    cfg=NetworkConfig(card_emb=4,cat_emb=4,hidden=24,gru_hidden=12,head_hidden=16)
    adv=AdvantageNet(cfg); pol=AveragePolicyNet(cfg)
    bundle=DomainBundle(
        domain='HU',seed=123,config=cfg,advantage=adv,policy=pol,
        adv_opt=torch.optim.Adam(adv.parameters(),lr=1e-3),
        pol_opt=torch.optim.Adam(pol.parameters(),lr=1e-3),
        adv_mem=UniformReservoir(5000,11),pol_mem=UniformReservoir(5000,12),
        batch_rng=random.Random(13),counters={})
    lib=_lib(); ep=Episode(1500,True,0,10,20,(0,750,750),1)
    sess=DeepCFRDomainSession(solver_library=lib,bundle=bundle,terminal_utility=chip_delta_utility)
    stats=sess.collect_root(ep,iteration=1,deck_seed=444)
    assert stats['advantage_samples']>0 and stats['strategy_samples']>0
    la=sess.train_advantage(steps=1,batch_size=8)
    lp=sess.train_average_policy(steps=1,batch_size=8)
    assert len(la)==1 and len(lp)==1
    assert bundle.counters['roots']==1
    assert bundle.counters['adv_optimizer_steps']==1
    assert bundle.counters['policy_optimizer_steps']==1
