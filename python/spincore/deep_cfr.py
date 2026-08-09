from __future__ import annotations
from dataclasses import dataclass
import math,random
from typing import Callable,Protocol,Sequence
from spincore_nn.reservoir import AdvantageSample,StrategySample,UniformReservoir
from .solver import SolverState
NUM_ACTIONS=6;Policy=tuple[float,...];TerminalUtility=Callable[[SolverState],tuple[float,float,float]]
class PolicyProvider(Protocol):
    def __call__(self,state:SolverState,observation:bytes,legal:tuple[int,...])->Policy:...
def _validate_policy(policy:Sequence[float],legal:tuple[int,...])->Policy:
    if len(policy)!=6:raise ValueError('policy must have 6 actions')
    if not legal:raise ValueError('empty legal set')
    s=set(legal);o=[0.0]*6;z=0.0
    for a,x in enumerate(policy):
        p=float(x)
        if not math.isfinite(p) or p<0:raise ValueError('invalid probability')
        if a in s:o[a]=p;z+=p
        elif p!=0:raise ValueError('illegal action has mass')
    if z<=0:
        for a in legal:o[a]=1/len(legal)
    else:
        for a in legal:o[a]/=z
    return tuple(o)
def uniform_policy(_s,_o,legal):
    x=[0.0]*6
    for a in legal:x[a]=1/len(legal)
    return tuple(x)
def regret_matching_policy(advantages:Sequence[float],legal:tuple[int,...])->Policy:
    if len(advantages)!=6:raise ValueError('six advantages required')
    x=[0.0]*6;z=sum(max(0.0,float(advantages[a])) for a in legal)
    if z<=0:
        for a in legal:x[a]=1/len(legal)
    else:
        for a in legal:x[a]=max(0.0,float(advantages[a]))/z
    return tuple(x)
def sample_action(policy,legal,rng):
    p=_validate_policy(policy,legal);x=rng.random();acc=0.0
    for a in legal:
        acc+=p[a]
        if x<acc:return a
    return legal[-1]
@dataclass(frozen=True)
class TraversalResult:utility:float;nodes:int;samples_added:int
class ExternalSamplingCollector:
    def __init__(self,*,policy,terminal_utility,rng,advantage_memory,strategy_memory):self.policy=policy;self.terminal_utility=terminal_utility;self.rng=rng;self.advantage_memory=advantage_memory;self.strategy_memory=strategy_memory
    def _p(self,s,o,l):return _validate_policy(self.policy(s,o,l),l)
    def collect_advantage(self,root,*,traverser,iteration):
        if iteration<=0:raise ValueError('positive iteration required')
        u,n,a=self._adv(root,traverser,iteration);return TraversalResult(u,n,a)
    def _adv(self,s,tr,iteration):
        if s.terminal:return float(self.terminal_utility(s)[tr]),1,0
        actor=s.actor;legal=s.legal_actions();obs=s.neural_bytes();sig=self._p(s,obs,legal)
        if actor==tr:
            vals=[0.0]*6;n=1;added=0
            for a in legal:
                c=s.child(a)
                try:v,nn,aa=self._adv(c,tr,iteration)
                finally:c.close()
                vals[a]=v;n+=nn;added+=aa
            nv=sum(sig[a]*vals[a] for a in legal);target=[0.0]*6
            for a in legal:target[a]=vals[a]-nv
            self.advantage_memory.add(AdvantageSample(obs,tuple(1 if a in legal else 0 for a in range(6)),tuple(target),float(iteration),iteration));return nv,n,added+1
        a=sample_action(sig,legal,self.rng);c=s.child(a)
        try:v,n,added=self._adv(c,tr,iteration)
        finally:c.close()
        return v,n+1,added
    def collect_strategy_own_reach(self,root,*,target_player,iteration):
        if iteration<=0:raise ValueError('positive iteration required')
        return self._strategy(root,target_player,iteration)
    def _strategy(self,s,target,iteration):
        if s.terminal:return 0
        actor=s.actor;legal=s.legal_actions();obs=s.neural_bytes();sig=self._p(s,obs,legal)
        if actor==target:
            self.strategy_memory.add(StrategySample(obs,tuple(1 if a in legal else 0 for a in range(6)),tuple(sig),float(iteration),iteration));a=sample_action(sig,legal,self.rng);c=s.child(a)
            try:return 1+self._strategy(c,target,iteration)
            finally:c.close()
        total=0
        for a in legal:
            c=s.child(a)
            try:total+=self._strategy(c,target,iteration)
            finally:c.close()
        return total
def chip_delta_utility(s):return tuple(float(x) for x in s.terminal_chip_delta())
def icm_delta_utility(payout_by_place:Sequence[float])->TerminalUtility:
    p=tuple(float(x) for x in payout_by_place)
    if len(p)!=3:raise ValueError('three payouts required')
    return lambda s: s.terminal_icm_delta(p)
class NeuralAdvantagePolicy:
    def __init__(self,model,*,device='cpu'):self.model=model;self.device=device
    def __call__(self,_s,observation,legal):
        import torch
        from spincore_nn.codec import decode_spnniv1,collate_inputs
        b=collate_inputs([decode_spnniv1(observation)],device=self.device);self.model.eval()
        with torch.no_grad():raw=self.model(b)[0].cpu().tolist()
        return regret_matching_policy(raw,legal)
def _batch(samples,device):
    import torch
    from spincore_nn.codec import decode_spnniv1,collate_inputs
    b=collate_inputs([decode_spnniv1(x.observation) for x in samples],device=device);t=torch.tensor([x.target for x in samples],dtype=torch.float32,device=device);w=torch.tensor([x.weight for x in samples],dtype=torch.float32,device=device);return b,t,w
class DeepCFRDomainSession:
    def __init__(self,*,solver_library,bundle,terminal_utility,device='cpu'):
        self.solver_library=solver_library;self.bundle=bundle;self.terminal_utility=terminal_utility;self.device=device;self.behavior=NeuralAdvantagePolicy(bundle.advantage,device=device);self.collector=ExternalSamplingCollector(policy=self.behavior,terminal_utility=terminal_utility,rng=bundle.batch_rng,advantage_memory=bundle.adv_mem,strategy_memory=bundle.pol_mem)
        for k in ('iteration','roots','advantage_samples','strategy_samples','nodes','adv_optimizer_steps','policy_optimizer_steps'):bundle.counters.setdefault(k,0)
    def collect_root(self,episode,*,iteration,deck_seed=None):
        if deck_seed is None:deck_seed=self.bundle.batch_rng.getrandbits(64)
        live=[i for i,x in enumerate(episode.stacks) if x>0];nodes=aa=ss=0
        for p in live:
            r=self.solver_library.create(episode,deck_seed)
            try:x=self.collector.collect_advantage(r,traverser=p,iteration=iteration)
            finally:r.close()
            nodes+=x.nodes;aa+=x.samples_added
        for p in live:
            r=self.solver_library.create(episode,deck_seed)
            try:ss+=self.collector.collect_strategy_own_reach(r,target_player=p,iteration=iteration)
            finally:r.close()
        c=self.bundle.counters;c['iteration']=max(c['iteration'],iteration);c['roots']+=1;c['nodes']+=nodes;c['advantage_samples']+=aa;c['strategy_samples']+=ss;return {'nodes':nodes,'advantage_samples':aa,'strategy_samples':ss}
    def _train(self,memory,model,opt,kind,steps,batch_size):
        from spincore_nn.training import train_step
        if steps and not memory.items:raise ValueError('empty memory')
        loss=[]
        for _ in range(steps):
            s=memory.sample(min(batch_size,len(memory.items)),self.bundle.batch_rng);b,t,w=_batch(s,self.device);loss.append(train_step(model,opt,b,t,w,kind))
        return loss
    def train_advantage(self,*,steps,batch_size):
        x=self._train(self.bundle.adv_mem,self.bundle.advantage,self.bundle.adv_opt,'advantage',steps,batch_size);self.bundle.counters['adv_optimizer_steps']+=len(x);return x
    def train_average_policy(self,*,steps,batch_size):
        x=self._train(self.bundle.pol_mem,self.bundle.policy,self.bundle.pol_opt,'strategy',steps,batch_size);self.bundle.counters['policy_optimizer_steps']+=len(x);return x
