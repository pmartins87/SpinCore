from __future__ import annotations
from dataclasses import dataclass,asdict
import math,os,random,tempfile
from pathlib import Path
from typing import Iterable,Sequence
import torch
from spincore_nn import NetworkConfig,AdvantageNet,AveragePolicyNet,DomainBundle,UniformReservoir,StrategySample
from spincore_nn.codec import decode_spnniv1,collate_inputs
from .deep_cfr import _validate_policy,sample_action

FROZEN_GATES={
 'advantage_weighted_nrmse_max':0.75,
 'policy_weighted_mean_tv_max':0.12,
 'cross_seed_mean_tv_max':0.15,
 'cross_seed_p95_tv_max':0.35,
}

def stratified_audit_indices(n:int,k:int,seed:int)->list[int]:
    if n<0 or k<0:raise ValueError('negative size')
    if n==0 or k==0:return []
    k=min(n,k);rng=random.Random(seed);out=[]
    for j in range(k):
        lo=(j*n)//k;hi=((j+1)*n)//k
        out.append(rng.randrange(lo,max(lo+1,hi)))
    return out

def weighted_normalized_rmse(pred:torch.Tensor,target:torch.Tensor,legal:torch.Tensor,weights:torch.Tensor)->float:
    m=legal.float();w=(weights/weights.sum().clamp_min(1e-12)).unsqueeze(1);num=(((pred-target)**2)*m*w).sum();den=((target**2)*m*w).sum().clamp_min(1e-12);return float(torch.sqrt(num/den).cpu())

def weighted_mean_tv(pred:torch.Tensor,target:torch.Tensor,weights:torch.Tensor)->float:
    tv=.5*torch.abs(pred-target).sum(1);w=weights/weights.sum().clamp_min(1e-12);return float((tv*w).sum().cpu())

def evaluate_average_policy(model,observations:Sequence[bytes],device='cpu')->torch.Tensor:
    if not observations:return torch.empty((0,6),dtype=torch.float32)
    b=collate_inputs([decode_spnniv1(x) for x in observations],device=device);model.eval()
    with torch.no_grad():return model.probabilities(b).detach().cpu()

def audit_model_fit(bundle:DomainBundle,*,sample_size:int=512,seed:int=0,device='cpu')->dict[str,float]:
    ai=stratified_audit_indices(len(bundle.adv_mem.items),sample_size,seed);pi=stratified_audit_indices(len(bundle.pol_mem.items),sample_size,seed^0x9E3779B9)
    out={'advantage_weighted_nrmse':math.inf,'policy_weighted_mean_tv':math.inf}
    if ai:
        s=[bundle.adv_mem.items[i] for i in ai];b=collate_inputs([decode_spnniv1(x.observation) for x in s],device=device);t=torch.tensor([x.target for x in s],dtype=torch.float32,device=device);w=torch.tensor([x.weight for x in s],dtype=torch.float32,device=device);bundle.advantage.eval()
        with torch.no_grad():p=bundle.advantage(b)
        out['advantage_weighted_nrmse']=weighted_normalized_rmse(p,t,b['legal'],w)
    if pi:
        s=[bundle.pol_mem.items[i] for i in pi];b=collate_inputs([decode_spnniv1(x.observation) for x in s],device=device);t=torch.tensor([x.target for x in s],dtype=torch.float32,device=device);w=torch.tensor([x.weight for x in s],dtype=torch.float32,device=device);bundle.policy.eval()
        with torch.no_grad():p=bundle.policy.probabilities(b)
        out['policy_weighted_mean_tv']=weighted_mean_tv(p,t,w)
    return out

def cross_seed_policy_tv(model_a,model_b,observations:Sequence[bytes],device='cpu')->dict[str,float]:
    a=evaluate_average_policy(model_a,observations,device);b=evaluate_average_policy(model_b,observations,device)
    if len(a)==0:return {'mean_tv':math.inf,'p50_tv':math.inf,'p95_tv':math.inf,'max_tv':math.inf}
    tv=.5*torch.abs(a-b).sum(1);q=torch.quantile(tv,torch.tensor([.5,.95]));return {'mean_tv':float(tv.mean()),'p50_tv':float(q[0]),'p95_tv':float(q[1]),'max_tv':float(tv.max())}

@dataclass
class MidIterationProgress:
    iteration:int=0;phase:str='collect_advantage';root_index:int=0;traverser_index:int=0;adv_optimizer_step:int=0;policy_optimizer_step:int=0

def checkpoint_payload(bundle:DomainBundle,progress:MidIterationProgress,extra:dict|None=None)->dict:
    return {'schema':'SPINCORE_R7_CHECKPOINT_V2','domain':bundle.domain,'seed':bundle.seed,'config':bundle.config.to_dict(),'advantage':bundle.advantage.state_dict(),'policy':bundle.policy.state_dict(),'adv_opt':bundle.adv_opt.state_dict(),'pol_opt':bundle.pol_opt.state_dict(),'adv_mem':bundle.adv_mem.state_dict(),'pol_mem':bundle.pol_mem.state_dict(),'batch_rng':bundle.batch_rng.getstate(),'torch_rng':torch.get_rng_state(),'counters':dict(bundle.counters),'progress':asdict(progress),'extra':dict(extra or {})}

def save_checkpoint(path:str|Path,bundle:DomainBundle,progress:MidIterationProgress,extra:dict|None=None)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload=checkpoint_payload(bundle,progress,extra);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)

def load_checkpoint(path:str|Path,*,device='cpu')->tuple[DomainBundle,MidIterationProgress,dict]:
    p=torch.load(Path(path),map_location=device,weights_only=False)
    if p.get('schema')!='SPINCORE_R7_CHECKPOINT_V2':raise ValueError('wrong checkpoint schema')
    cfg=NetworkConfig(**p['config']);adv=AdvantageNet(cfg).to(device);pol=AveragePolicyNet(cfg).to(device);adv.load_state_dict(p['advantage']);pol.load_state_dict(p['policy']);ao=torch.optim.Adam(adv.parameters());po=torch.optim.Adam(pol.parameters());ao.load_state_dict(p['adv_opt']);po.load_state_dict(p['pol_opt']);bundle=DomainBundle(p['domain'],int(p['seed']),cfg,adv,pol,ao,po,UniformReservoir.from_state_dict(p['adv_mem']),UniformReservoir.from_state_dict(p['pol_mem']),random.Random(),dict(p['counters']));bundle.batch_rng.setstate(p['batch_rng']);torch.set_rng_state(p['torch_rng']);return bundle,MidIterationProgress(**p['progress']),dict(p.get('extra',{}))

def collect_strategy_own_reach_native(root,*,target_player:int,iteration:int,policy,rng:random.Random,strategy_memory,max_nodes:int=100000,max_depth:int=64)->int:
    if iteration<=0:raise ValueError('positive iteration required')
    def walk(state):
        total=0
        with state.frontier_until_actor(target_player,max_nodes=max_nodes,max_depth=max_depth) as f:
            clones=[f.clone_state(i) for i in range(len(f))]
        for s in clones:
            try:
                if s.terminal:continue
                legal=s.legal_actions();obs=s.neural_bytes();sig=_validate_policy(policy(s,obs,legal),legal);strategy_memory.add(StrategySample(obs,tuple(1 if a in legal else 0 for a in range(6)),tuple(sig),float(iteration),iteration));a=sample_action(sig,legal,rng);child=s.child(a)
                try:total+=1+walk(child)
                finally:child.close()
            finally:s.close()
        return total
    return walk(root)
