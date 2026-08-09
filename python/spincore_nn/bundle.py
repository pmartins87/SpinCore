from __future__ import annotations
from dataclasses import dataclass
import random
from .reservoir import UniformReservoir
@dataclass
class DomainBundle:
    domain:str;seed:int;config:object;advantage:object;policy:object;adv_opt:object;pol_opt:object;adv_mem:UniformReservoir;pol_mem:UniformReservoir;batch_rng:random.Random;counters:dict
