from __future__ import annotations
import random
from pathlib import Path
import torch
from spincore.solver import Episode,SolverLibrary
from spincore.deep_cfr import ExternalSamplingCollector,chip_delta_utility,icm_delta_utility,uniform_policy,regret_matching_policy,DeepCFRDomainSession
from spincore_nn import UniformReservoir,NetworkConfig,AdvantageNet,AveragePolicyNet,DomainBundle
ROOT=Path(__file__).resolve().parents[1];LIB=ROOT/'build'/'libspincore_solver_c.so'
def _root(seed=123):
    L=SolverLibrary(LIB);e=Episode(1500,True,0,10,20,(0,750,750),1,(0,));return L,e,L.create(e,seed)
def test_regret_matching_normalizes_legal():
    p=regret_matching_policy([-1,2,0,0,3,0],(1,4));assert abs(p[1]-.4)<1e-12 and abs(p[4]-.6)<1e-12 and sum(p)==1
def test_advantage_targets_centered():
    _,_,r=_root(777);adv=UniformReservoir(10000,1);pol=UniformReservoir(10000,2);c=ExternalSamplingCollector(policy=uniform_policy,terminal_utility=chip_delta_utility,rng=random.Random(3),advantage_memory=adv,strategy_memory=pol)
    x=c.collect_advantage(r,traverser=r.actor,iteration=1);r.close();assert x.samples_added and adv.items
    for s in adv.items:
        legal=[i for i,v in enumerate(s.legal) if v];assert abs(sum(s.target[i] for i in legal)/len(legal))<1e-7
        assert all(s.target[i]==0 for i,v in enumerate(s.legal) if not v)
def test_own_reach_records_normalized_policy():
    _,_,r=_root(888);adv=UniformReservoir(10000,1);pol=UniformReservoir(10000,2);c=ExternalSamplingCollector(policy=uniform_policy,terminal_utility=chip_delta_utility,rng=random.Random(3),advantage_memory=adv,strategy_memory=pol);n=c.collect_strategy_own_reach(r,target_player=r.actor,iteration=4);r.close();assert n==len(pol.items)>0
    for s in pol.items:assert abs(sum(s.target)-1)<1e-8 and s.weight==4
def test_collector_reproducible():
    def run():
        _,_,r=_root(999);a=UniformReservoir(10000,1);p=UniformReservoir(10000,2);c=ExternalSamplingCollector(policy=uniform_policy,terminal_utility=chip_delta_utility,rng=random.Random(3),advantage_memory=a,strategy_memory=p);x=c.collect_advantage(r,traverser=r.actor,iteration=2);r.close();return x,a.items
    assert run()==run()
def test_icm_utility_factory():
    _,_,r=_root(222)
    while not r.terminal:
        le=r.legal_actions();r.apply(1 if 1 in le else le[0])
    d=icm_delta_utility((.5,.3,.2))(r);assert abs(sum(d))<1e-10;r.close()
def test_neural_session_collects_and_trains():
    torch.manual_seed(1);cfg=NetworkConfig(card_emb=4,cat_emb=3,hidden=24,gru_hidden=8,head_hidden=12);adv=AdvantageNet(cfg);pol=AveragePolicyNet(cfg);bundle=DomainBundle('HU',1,cfg,adv,pol,torch.optim.Adam(adv.parameters(),lr=1e-3),torch.optim.Adam(pol.parameters(),lr=1e-3),UniformReservoir(5000,11),UniformReservoir(5000,12),random.Random(13),{})
    L,e,_=_root(1);sess=DeepCFRDomainSession(solver_library=L,bundle=bundle,terminal_utility=icm_delta_utility((.5,.3,.2)));st=sess.collect_root(e,iteration=1,deck_seed=444);assert st['advantage_samples']>0 and st['strategy_samples']>0;assert len(sess.train_advantage(steps=1,batch_size=8))==1;assert len(sess.train_average_policy(steps=1,batch_size=8))==1;assert bundle.counters['roots']==1
