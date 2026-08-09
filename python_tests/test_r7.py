from __future__ import annotations
import random,subprocess,sys
from pathlib import Path
import torch
from spincore.solver import Episode,SolverLibrary
from spincore.deep_cfr import ExternalSamplingCollector,uniform_policy,icm_delta_utility,DeepCFRDomainSession
from spincore.r7 import *
from spincore_nn import *
ROOT=Path(__file__).resolve().parents[1];LIB=ROOT/'build'/'libspincore_solver_c.so'
def bundle(seed=1):
    torch.manual_seed(seed);c=NetworkConfig(card_emb=3,cat_emb=2,hidden=16,gru_hidden=6,head_hidden=8);a=AdvantageNet(c);p=AveragePolicyNet(c);return DomainBundle('HU',seed,c,a,p,torch.optim.Adam(a.parameters(),lr=1e-3),torch.optim.Adam(p.parameters(),lr=1e-3),UniformReservoir(2000,seed+11),UniformReservoir(2000,seed+12),random.Random(seed+13),{})
def episode():return Episode(1500,True,0,10,20,(0,750,750),1,(0,))
def fill(b,roots=2):
    s=DeepCFRDomainSession(solver_library=SolverLibrary(LIB),bundle=b,terminal_utility=icm_delta_utility((.5,.3,.2)))
    for i in range(roots):s.collect_root(episode(),iteration=1,deck_seed=100+i)
    return s
def test_stratified_audit_spans_reservoir():
    x=stratified_audit_indices(10000,10,4);assert x==stratified_audit_indices(10000,10,4);assert min(x)<1000 and max(x)>=9000 and len(set(x))==10
def test_weighted_metrics_exact_zero():
    p=torch.tensor([[.2,.8],[.1,.9]]);t=p.clone();m=torch.ones_like(p,dtype=torch.bool);w=torch.tensor([1.,3.]);assert weighted_normalized_rmse(p,t,m,w)==0;assert weighted_mean_tv(p,t,w)==0
def test_checkpoint_roundtrip_preserves_state(tmp_path):
    b=bundle();fill(b,1);p=MidIterationProgress(iteration=2,phase='train',root_index=7);path=tmp_path/'x.pt';save_checkpoint(path,b,p,{'x':3});b2,p2,e=load_checkpoint(path);assert p2==p and e['x']==3 and b2.counters==b.counters and b2.adv_mem.items==b.adv_mem.items
    assert b2.batch_rng.random()==b.batch_rng.random()
def test_native_own_reach_matches_python_semantics():
    L=SolverLibrary(LIB);r1=L.create(episode(),333);r2=L.create(episode(),333);a1=UniformReservoir(10000,1);p1=UniformReservoir(10000,2);a2=UniformReservoir(10000,1);p2=UniformReservoir(10000,2);rng1=random.Random(9);rng2=random.Random(9);c=ExternalSamplingCollector(policy=uniform_policy,terminal_utility=icm_delta_utility((.5,.3,.2)),rng=rng1,advantage_memory=a1,strategy_memory=p1);n1=c.collect_strategy_own_reach(r1,target_player=r1.actor,iteration=3);n2=collect_strategy_own_reach_native(r2,target_player=r2.actor,iteration=3,policy=uniform_policy,rng=rng2,strategy_memory=p2);r1.close();r2.close();assert n1==n2 and p1.items==p2.items
def test_fit_audit_returns_finite_metrics():
    b=bundle();s=fill(b,2);s.train_advantage(steps=1,batch_size=8);s.train_average_policy(steps=1,batch_size=8);m=audit_model_fit(b,sample_size=16,seed=5);assert all(torch.isfinite(torch.tensor(list(m.values()))))
def test_cross_seed_tv_zero_for_same_model():
    b=bundle();fill(b,1);obs=[x.observation for x in b.pol_mem.items[:10]];m=cross_seed_policy_tv(b.policy,b.policy,obs);assert m['mean_tv']==0 and m['p95_tv']==0
def test_fresh_process_worker_updates_checkpoint(tmp_path):
    b=bundle();fill(b,1);path=tmp_path/'w.pt';save_checkpoint(path,b,MidIterationProgress(iteration=1));env={'PYTHONPATH':str(ROOT/'python')};import os;env={**os.environ,**env};subprocess.check_call([sys.executable,str(ROOT/'tools'/'r7_training_worker.py'),'--checkpoint',str(path),'--solver',str(LIB),'--kind','advantage','--steps','1','--batch-size','8','--payout','.5','.3','.2'],env=env);b2,p,e=load_checkpoint(path);assert b2.counters['adv_optimizer_steps']==1 and e['last_worker']['kind']=='advantage'
def test_continuous_equals_stop_restore_continue(tmp_path):
    def run(with_restore:bool):
        b=bundle(77);s=fill(b,1);s.train_advantage(steps=1,batch_size=8);s.train_average_policy(steps=1,batch_size=8)
        if with_restore:
            p=tmp_path/'resume.pt';save_checkpoint(p,b,MidIterationProgress(iteration=1,phase='collect',root_index=1));b,_,_=load_checkpoint(p);s=DeepCFRDomainSession(solver_library=SolverLibrary(LIB),bundle=b,terminal_utility=icm_delta_utility((.5,.3,.2)))
        s.collect_root(episode(),iteration=2,deck_seed=999);s.train_advantage(steps=1,batch_size=8);s.train_average_policy(steps=1,batch_size=8)
        return b
    a=run(False);b=run(True);assert a.counters==b.counters and a.adv_mem.items==b.adv_mem.items and a.pol_mem.items==b.pol_mem.items
    for x,y in zip(a.advantage.state_dict().values(),b.advantage.state_dict().values()):assert torch.equal(x,y)
    for x,y in zip(a.policy.state_dict().values(),b.policy.state_dict().values()):assert torch.equal(x,y)
