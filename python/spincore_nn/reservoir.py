from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Generic,TypeVar
T=TypeVar('T')
@dataclass(frozen=True)
class AdvantageSample:
    observation:bytes; legal:tuple[int,...]; target:tuple[float,...]; weight:float; iteration:int
@dataclass(frozen=True)
class StrategySample:
    observation:bytes; legal:tuple[int,...]; target:tuple[float,...]; weight:float; iteration:int
class UniformReservoir(Generic[T]):
    def __init__(self,capacity:int,seed:int):
        if capacity<=0:raise ValueError('capacity must be positive')
        self.capacity=int(capacity);self.items:list[T]=[];self.seen=0;self.rng=random.Random(seed)
    def add(self,item:T):
        self.seen+=1
        if len(self.items)<self.capacity:self.items.append(item);return
        j=self.rng.randrange(self.seen)
        if j<self.capacity:self.items[j]=item
    def sample(self,n:int,rng:random.Random|None=None)->list[T]:
        if n<0 or n>len(self.items):raise ValueError('bad sample size')
        return (rng or self.rng).sample(self.items,n)
    def state_dict(self):return {'capacity':self.capacity,'items':list(self.items),'seen':self.seen,'rng_state':self.rng.getstate()}
    @classmethod
    def from_state_dict(cls,state):
        obj=cls(int(state['capacity']),0);obj.items=list(state['items']);obj.seen=int(state['seen']);obj.rng.setstate(state['rng_state']);return obj
