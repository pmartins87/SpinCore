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

@dataclass(frozen=True,order=True)
class ResolvedExactAction:
    action_type:int
    amount_to:int
    def __post_init__(self):
        if self.action_type<0 or self.action_type>5:raise ValueError('exact action type must be 0..5')
        if self.amount_to<0:raise ValueError('exact action amount_to must be nonnegative')

@dataclass(frozen=True)
class DealSnapshot:
    holes:tuple[tuple[int,int],tuple[int,int],tuple[int,int]]
    board:tuple[int,int,int,int,int]
    visible_board_count:int
    def __post_init__(self):
        if len(self.holes)!=3 or any(len(row)!=2 for row in self.holes):raise ValueError('deal snapshot requires three two-card hole rows')
        if len(self.board)!=5:raise ValueError('deal snapshot requires five board cards')
        if self.visible_board_count<0 or self.visible_board_count>5:raise ValueError('invalid visible board count')

class _ScenarioV2(C.Structure):
    _fields_=[('total_chips',C.c_int32),('game_is_hu',C.c_int32),('blind_index',C.c_int32),('small_blind',C.c_int32),('big_blind',C.c_int32),('stack_0',C.c_int32),('stack_1',C.c_int32),('stack_2',C.c_int32),('dead_player_0',C.c_int32),('dead_player_1',C.c_int32),('dead_player_count',C.c_int32),('dealer_id',C.c_int32)]

class _DealV1(C.Structure):
    _fields_=[('hole_0_0',C.c_int32),('hole_0_1',C.c_int32),('hole_1_0',C.c_int32),('hole_1_1',C.c_int32),('hole_2_0',C.c_int32),('hole_2_1',C.c_int32),('board_0',C.c_int32),('board_1',C.c_int32),('board_2',C.c_int32),('board_3',C.c_int32),('board_4',C.c_int32)]

def _dead(e:Episode)->tuple[int,...]:
    if len(e.stacks)!=3: raise ValueError('exactly three seat stacks required')
    d=tuple(int(x) for x in e.dead_players)
    if not d and e.game_is_hu: d=tuple(i for i,x in enumerate(e.stacks) if x<=0)
    if len(d)>2 or len(set(d))!=len(d) or any(x not in (0,1,2) for x in d): raise ValueError('invalid dead_players')
    return d

def _scenario(e:Episode)->_ScenarioV2:
    d=_dead(e)
    return _ScenarioV2(e.total_chips,int(e.game_is_hu),e.blind_index,e.small_blind,e.big_blind,*e.stacks,d[0] if len(d)>0 else -1,d[1] if len(d)>1 else -1,len(d),e.dealer_id)

def _deal(e:Episode,holes:Sequence[Sequence[int]],board:Sequence[int])->_DealV1:
    if len(holes)!=3 or any(len(row)!=2 for row in holes):raise ValueError('explicit deal requires three two-card hole rows')
    if len(board)!=5:raise ValueError('explicit deal requires five board cards')
    dead=set(_dead(e)); flat=[]
    for seat,row in enumerate(holes):
        for raw in row:
            value=int(raw)
            if seat in dead:
                if value!=-1:raise ValueError('dead-seat explicit hole id must be -1')
            elif value<0 or value>=52:raise ValueError('live-seat explicit hole id outside 0..51')
            if value>=0:flat.append(value)
    board_ids=tuple(int(x) for x in board)
    if any(x<0 or x>=52 for x in board_ids):raise ValueError('explicit board id outside 0..51')
    flat.extend(board_ids)
    if len(flat)!=len(set(flat)):raise ValueError('explicit deal contains duplicate card ids')
    vals=[int(holes[s][r]) for s in range(3) for r in range(2)]+list(board_ids)
    return _DealV1(*vals)

class SolverLibrary:
    def __init__(self,path:str|Path):
        self.path=Path(path); self.lib=C.CDLL(str(self.path)); L=self.lib
        L.spincore_solver_c_abi_version.argtypes=[];L.spincore_solver_c_abi_version.restype=C.c_int32
        L.spincore_solver_last_error.argtypes=[];L.spincore_solver_last_error.restype=C.c_char_p
        if L.spincore_solver_c_abi_version()!=2: raise RuntimeError('SPINCORE_SOLVER_C_ABI_V2 required')
        L.spincore_solver_state_create_v2.argtypes=[C.POINTER(_ScenarioV2),C.c_uint64];L.spincore_solver_state_create_v2.restype=C.c_void_p
        create_deal=getattr(L,'spincore_solver_state_create_v2_deal',None);snapshot=getattr(L,'spincore_solver_state_deal_snapshot_v1',None)
        if (create_deal is None)!=(snapshot is None):raise RuntimeError('incomplete explicit-deal diagnostic solver extension')
        self.explicit_deal_available=create_deal is not None
        if self.explicit_deal_available:
            create_deal.argtypes=[C.POINTER(_ScenarioV2),C.POINTER(_DealV1)];create_deal.restype=C.c_void_p
            snapshot.argtypes=[C.c_void_p,C.POINTER(_DealV1),C.POINTER(C.c_int32)];snapshot.restype=C.c_int32
        L.spincore_solver_state_clone.argtypes=[C.c_void_p];L.spincore_solver_state_clone.restype=C.c_void_p
        L.spincore_solver_state_destroy.argtypes=[C.c_void_p];L.spincore_solver_state_destroy.restype=None
        L.spincore_solver_state_terminal.argtypes=[C.c_void_p];L.spincore_solver_state_terminal.restype=C.c_int32
        L.spincore_solver_state_actor.argtypes=[C.c_void_p];L.spincore_solver_state_actor.restype=C.c_int32
        L.spincore_solver_state_domain.argtypes=[C.c_void_p];L.spincore_solver_state_domain.restype=C.c_int32
        L.spincore_solver_state_legal_mask.argtypes=[C.c_void_p];L.spincore_solver_state_legal_mask.restype=C.c_uint32
        L.spincore_solver_state_apply_abstract.argtypes=[C.c_void_p,C.c_int32];L.spincore_solver_state_apply_abstract.restype=C.c_int32
        L.spincore_solver_state_universal_legal_mask.argtypes=[C.c_void_p,C.c_uint32];L.spincore_solver_state_universal_legal_mask.restype=C.c_uint32
        L.spincore_solver_state_apply_universal.argtypes=[C.c_void_p,C.c_uint32,C.c_int32];L.spincore_solver_state_apply_universal.restype=C.c_int32
        L.spincore_solver_state_resolve_universal_exact.argtypes=[C.c_void_p,C.c_uint32,C.c_int32,C.POINTER(C.c_int32),C.POINTER(C.c_int32)];L.spincore_solver_state_resolve_universal_exact.restype=C.c_int32
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
        x=_scenario(e);p=self.lib.spincore_solver_state_create_v2(C.byref(x),C.c_uint64(seed));
        if not p: raise RuntimeError(self.error() or 'state creation failed')
        return SolverState(self,p)
    def create_with_deal(self,e:Episode,holes:Sequence[Sequence[int]],board:Sequence[int])->'SolverState':
        if not self.explicit_deal_available:raise RuntimeError('solver library does not expose explicit-deal diagnostic API')
        x=_scenario(e);d=_deal(e,holes,board);p=self.lib.spincore_solver_state_create_v2_deal(C.byref(x),C.byref(d))
        if not p:raise RuntimeError(self.error() or 'explicit-deal state creation failed')
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
    def deal_snapshot(self)->DealSnapshot:
        if not self.owner.explicit_deal_available:raise RuntimeError('solver library does not expose explicit-deal diagnostic API')
        d=_DealV1();visible=C.c_int32()
        if self.owner.lib.spincore_solver_state_deal_snapshot_v1(self._p(),C.byref(d),C.byref(visible))!=0:raise RuntimeError(self.owner.error() or 'deal snapshot failed')
        holes=((int(d.hole_0_0),int(d.hole_0_1)),(int(d.hole_1_0),int(d.hole_1_1)),(int(d.hole_2_0),int(d.hole_2_1)))
        board=(int(d.board_0),int(d.board_1),int(d.board_2),int(d.board_3),int(d.board_4))
        return DealSnapshot(holes,board,int(visible.value))
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
    @staticmethod
    def _validated_universal_mask(active_mask:int)->int:
        mask=int(active_mask)
        if mask<0 or mask>0x3ff:raise ValueError('universal active mask must use only slots 0..9')
        return mask
    def universal_legal_actions(self,active_mask:int):
        mask=self._validated_universal_mask(active_mask)
        m=int(self.owner.lib.spincore_solver_state_universal_legal_mask(self._p(),C.c_uint32(mask)))
        return tuple(i for i in range(10) if m&(1<<i))
    def resolve_universal_exact(self,active_mask:int,a:int)->ResolvedExactAction:
        mask=self._validated_universal_mask(active_mask);action=int(a)
        if action<0 or action>9:raise ValueError('bad universal action')
        out_type=C.c_int32();out_amount=C.c_int32()
        rc=self.owner.lib.spincore_solver_state_resolve_universal_exact(self._p(),C.c_uint32(mask),action,C.byref(out_type),C.byref(out_amount))
        if rc!=0:raise RuntimeError(self.owner.error() or 'universal exact resolution failed')
        return ResolvedExactAction(int(out_type.value),int(out_amount.value))
    def universal_resolved_actions(self,active_mask:int)->tuple[tuple[int,ResolvedExactAction],...]:
        mask=self._validated_universal_mask(active_mask)
        return tuple((slot,self.resolve_universal_exact(mask,slot)) for slot in self.universal_legal_actions(mask))
    def apply_universal(self,active_mask:int,a:int):
        mask=self._validated_universal_mask(active_mask);action=int(a)
        if action<0 or action>9:raise ValueError('bad universal action')
        if self.owner.lib.spincore_solver_state_apply_universal(self._p(),C.c_uint32(mask),action)!=0:raise RuntimeError(self.owner.error() or 'universal apply failed')
        return self
    def child_universal(self,active_mask:int,a:int):
        c=self.clone()
        try:return c.apply_universal(active_mask,a)
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
