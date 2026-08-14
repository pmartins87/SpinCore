from __future__ import annotations
from dataclasses import dataclass
import ctypes as C
import math
from pathlib import Path
from typing import Iterator, Sequence

@dataclass(frozen=True)
class Episode:
    total_chips:int; game_is_hu:bool; blind_index:int; small_blind:int; big_blind:int
    stacks:tuple[int,int,int]; dealer_id:int; dead_players:tuple[int,...]=()

class _ScenarioV2(C.Structure):
    _fields_=[('total_chips',C.c_int32),('game_is_hu',C.c_int32),('blind_index',C.c_int32),('small_blind',C.c_int32),('big_blind',C.c_int32),('stack_0',C.c_int32),('stack_1',C.c_int32),('stack_2',C.c_int32),('dead_player_0',C.c_int32),('dead_player_1',C.c_int32),('dead_player_count',C.c_int32),('dealer_id',C.c_int32)]

def _dead(e:Episode)->tuple[int,...]:
    if len(e.stacks)!=3: raise ValueError('exactly three seat stacks required')
    d=tuple(int(x) for x in e.dead_players)
    if not d and e.game_is_hu: d=tuple(i for i,x in enumerate(e.stacks) if x<=0)
    if len(d)>2 or len(set(d))!=len(d) or any(x not in (0,1,2) for x in d): raise ValueError('invalid dead_players')
    return d

class SolverLibrary:
    def __init__(self,path:str|Path):
        self.path=Path(path); self.lib=C.CDLL(str(self.path)); L=self.lib
        L.spincore_solver_c_abi_version.argtypes=[];L.spincore_solver_c_abi_version.restype=C.c_int32
        L.spincore_solver_last_error.argtypes=[];L.spincore_solver_last_error.restype=C.c_char_p
        if L.spincore_solver_c_abi_version()!=2: raise RuntimeError('SPINCORE_SOLVER_C_ABI_V2 required')
        L.spincore_solver_state_create_v2.argtypes=[C.POINTER(_ScenarioV2),C.c_uint64];L.spincore_solver_state_create_v2.restype=C.c_void_p
        L.spincore_solver_state_clone.argtypes=[C.c_void_p];L.spincore_solver_state_clone.restype=C.c_void_p
        L.spincore_solver_state_destroy.argtypes=[C.c_void_p];L.spincore_solver_state_destroy.restype=None
        L.spincore_solver_state_terminal.argtypes=[C.c_void_p];L.spincore_solver_state_terminal.restype=C.c_int32
        L.spincore_solver_state_actor.argtypes=[C.c_void_p];L.spincore_solver_state_actor.restype=C.c_int32
        L.spincore_solver_state_domain.argtypes=[C.c_void_p];L.spincore_solver_state_domain.restype=C.c_int32
        L.spincore_solver_state_legal_mask.argtypes=[C.c_void_p];L.spincore_solver_state_legal_mask.restype=C.c_uint32
        L.spincore_solver_state_apply_abstract.argtypes=[C.c_void_p,C.c_int32];L.spincore_solver_state_apply_abstract.restype=C.c_int32
        L.spincore_solver_state_neural_input.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_size_t];L.spincore_solver_state_neural_input.restype=C.c_size_t
        L.spincore_solver_state_neural_input_v2.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_size_t];L.spincore_solver_state_neural_input_v2.restype=C.c_size_t
        L.spincore_solver_state_terminal_chip_delta.argtypes=[C.c_void_p,C.POINTER(C.c_int32)];L.spincore_solver_state_terminal_chip_delta.restype=C.c_int32
        L.spincore_solver_state_terminal_icm_delta.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.POINTER(C.c_double)];L.spincore_solver_state_terminal_icm_delta.restype=C.c_int32
        L.spincore_solver_frontier_create_until_actor.argtypes=[C.c_void_p,C.c_int32,C.c_size_t,C.c_size_t];L.spincore_solver_frontier_create_until_actor.restype=C.c_void_p
        L.spincore_solver_frontier_destroy.argtypes=[C.c_void_p];L.spincore_solver_frontier_destroy.restype=None
        L.spincore_solver_frontier_size.argtypes=[C.c_void_p];L.spincore_solver_frontier_size.restype=C.c_size_t
        L.spincore_solver_frontier_nodes_visited.argtypes=[C.c_void_p];L.spincore_solver_frontier_nodes_visited.restype=C.c_size_t
        L.spincore_solver_frontier_max_depth_reached.argtypes=[C.c_void_p];L.spincore_solver_frontier_max_depth_reached.restype=C.c_size_t
        L.spincore_solver_frontier_is_terminal.argtypes=[C.c_void_p,C.c_size_t];L.spincore_solver_frontier_is_terminal.restype=C.c_int32
        L.spincore_solver_frontier_clone_state.argtypes=[C.c_void_p,C.c_size_t];L.spincore_solver_frontier_clone_state.restype=C.c_void_p
    def error(self)->str:return (self.lib.spincore_solver_last_error() or b'').decode('utf-8','replace')
    def create(self,e:Episode,seed:int)->'SolverState':
        d=_dead(e); x=_ScenarioV2(e.total_chips,int(e.game_is_hu),e.blind_index,e.small_blind,e.big_blind,*e.stacks,d[0] if len(d)>0 else -1,d[1] if len(d)>1 else -1,len(d),e.dealer_id)
        p=self.lib.spincore_solver_state_create_v2(C.byref(x),C.c_uint64(seed));
        if not p: raise RuntimeError(self.error() or 'state creation failed')
        return SolverState(self,p)

class SolverState:
    def __init__(self,owner:SolverLibrary,ptr:int|C.c_void_p):self.owner=owner;self.ptr=ptr if isinstance(ptr,C.c_void_p) else C.c_void_p(ptr)
    def _p(self):
        if not self.ptr or not self.ptr.value:raise RuntimeError('solver state is closed')
        return self.ptr
    def close(self):
        if self.ptr and self.ptr.value:self.owner.lib.spincore_solver_state_destroy(self.ptr);self.ptr=C.c_void_p()
    def __enter__(self):self._p();return self
    def __exit__(self,*_):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
    def clone(self):
        p=self.owner.lib.spincore_solver_state_clone(self._p());
        if not p:raise RuntimeError(self.owner.error() or 'clone failed')
        return SolverState(self.owner,p)
    @property
    def terminal(self):return bool(self.owner.lib.spincore_solver_state_terminal(self._p()))
    @property
    def actor(self):return int(self.owner.lib.spincore_solver_state_actor(self._p()))
    @property
    def domain(self):return int(self.owner.lib.spincore_solver_state_domain(self._p()))
    def legal_actions(self):
        m=int(self.owner.lib.spincore_solver_state_legal_mask(self._p()));return tuple(i for i in range(6) if m&(1<<i))
    def apply(self,a:int):
        if self.owner.lib.spincore_solver_state_apply_abstract(self._p(),int(a))!=0:raise RuntimeError(self.owner.error() or 'apply failed')
        return self
    def child(self,a:int):
        c=self.clone()
        try:return c.apply(a)
        except Exception:c.close();raise
    def _neural_payload(self,fn_name:str)->bytes:
        p=self._p();fn=getattr(self.owner.lib,fn_name);n=int(fn(p,None,0));
        if n<=0:raise RuntimeError(self.owner.error() or f'no {fn_name} payload')
        b=(C.c_uint8*n)();got=int(fn(p,b,n));
        if got!=n:raise RuntimeError(self.owner.error() or f'{fn_name} size mismatch')
        return bytes(b)
    def neural_bytes(self)->bytes:return self._neural_payload('spincore_solver_state_neural_input')
    def neural_bytes_v2(self)->bytes:return self._neural_payload('spincore_solver_state_neural_input_v2')
    def terminal_chip_delta(self):
        o=(C.c_int32*3)();
        if self.owner.lib.spincore_solver_state_terminal_chip_delta(self._p(),o)!=0:raise RuntimeError(self.owner.error())
        return tuple(int(x) for x in o)
    def terminal_icm_delta(self,payout_by_place:Sequence[float]):
        if len(payout_by_place)!=3:raise ValueError('three payouts required')
        p=tuple(float(x) for x in payout_by_place)
        if any(not math.isfinite(x) for x in p):raise ValueError('nonfinite payout')
        a=(C.c_double*3)(*p);o=(C.c_double*3)()
        if self.owner.lib.spincore_solver_state_terminal_icm_delta(self._p(),a,o)!=0:raise RuntimeError(self.owner.error())
        return tuple(float(x) for x in o)
    def frontier_until_actor(self,target_actor:int,*,max_nodes:int=100000,max_depth:int=64):
        p=self.owner.lib.spincore_solver_frontier_create_until_actor(self._p(),target_actor,max_nodes,max_depth)
        if not p:raise RuntimeError(self.owner.error() or 'frontier failed')
        return SolverFrontier(self.owner,p)

class SolverFrontier:
    def __init__(self,owner,ptr):self.owner=owner;self.ptr=C.c_void_p(ptr)
    def _p(self):
        if not self.ptr or not self.ptr.value:raise RuntimeError('frontier closed')
        return self.ptr
    def close(self):
        if self.ptr and self.ptr.value:self.owner.lib.spincore_solver_frontier_destroy(self.ptr);self.ptr=C.c_void_p()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
    def __len__(self):return int(self.owner.lib.spincore_solver_frontier_size(self._p()))
    @property
    def nodes_visited(self):return int(self.owner.lib.spincore_solver_frontier_nodes_visited(self._p()))
    @property
    def max_depth_reached(self):return int(self.owner.lib.spincore_solver_frontier_max_depth_reached(self._p()))
    def is_terminal(self,i):
        if i<0 or i>=len(self):raise IndexError(i)
        r=int(self.owner.lib.spincore_solver_frontier_is_terminal(self._p(),i));
        if r not in (0,1):raise RuntimeError(self.owner.error())
        return bool(r)
    def clone_state(self,i):
        if i<0 or i>=len(self):raise IndexError(i)
        p=self.owner.lib.spincore_solver_frontier_clone_state(self._p(),i)
        if not p:raise RuntimeError(self.owner.error())
        return SolverState(self.owner,p)
    def cloned_states(self)->Iterator[SolverState]:
        for i in range(len(self)):yield self.clone_state(i)
