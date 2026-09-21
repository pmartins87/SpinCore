from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable,Generic,TypeVar

T=TypeVar("T")


@dataclass(frozen=True)
class AdvantageSample:
    observation:bytes
    legal:tuple[int,...]
    target:tuple[float,...]
    weight:float
    iteration:int


@dataclass(frozen=True)
class StrategySample:
    observation:bytes
    legal:tuple[int,...]
    target:tuple[float,...]
    weight:float
    iteration:int


class UniformReservoir(Generic[T]):
    """Algorithm-R reservoir with an optional observational write hook.

    The hook is deliberately excluded from state_dict/checkpoints.  It is used
    by LT3 only to keep a compact mmap mirror synchronized with the
    authoritative Python reservoir.  Installing a hook must not consume RNG or
    alter the replacement decision.
    """

    def __init__(self,capacity:int,seed:int):
        if capacity<=0:
            raise ValueError("capacity must be positive")
        self.capacity=int(capacity)
        self.items:list[T]=[]
        self.seen=0
        self.rng=random.Random(seed)
        self._write_observer:Callable[[int,T],None]|None=None

    def set_write_observer(
        self,
        observer:Callable[[int,T],None]|None,
    )->None:
        self._write_observer=observer

    def add(self,item:T):
        self.seen+=1
        write_index=None
        if len(self.items)<self.capacity:
            self.items.append(item)
            write_index=len(self.items)-1
        else:
            j=self.rng.randrange(self.seen)
            if j<self.capacity:
                self.items[j]=item
                write_index=int(j)

        if write_index is not None and self._write_observer is not None:
            self._write_observer(int(write_index),item)

    def sample(self,n:int,rng:random.Random|None=None)->list[T]:
        if n<0 or n>len(self.items):
            raise ValueError("bad sample size")
        return (rng or self.rng).sample(self.items,n)

    def state_dict(self):
        return {
            "capacity":self.capacity,
            "items":list(self.items),
            "seen":self.seen,
            "rng_state":self.rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls,state):
        obj=cls(int(state["capacity"]),0)
        obj.items=list(state["items"])
        obj.seen=int(state["seen"])
        obj.rng.setstate(state["rng_state"])
        return obj
