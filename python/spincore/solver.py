from __future__ import annotations
from dataclasses import dataclass
import ctypes as C
from pathlib import Path

@dataclass(frozen=True)
class Episode:
    total_chips:int; game_is_hu:bool; blind_index:int; small_blind:int; big_blind:int
    stacks:tuple[int,int,int]; dealer_id:int

class _Episode(C.Structure):
    _fields_=[('total_chips',C.c_int32),('game_is_hu',C.c_int32),('blind_index',C.c_int32),
              ('small_blind',C.c_int32),('big_blind',C.c_int32),('stacks',C.c_int32*3),('dealer_id',C.c_int32)]

class SolverLibrary:
    def __init__(self,path:str|Path):
        self.lib=C.CDLL(str(path)); L=self.lib
        L.spincore_solver_abi_version.restype=C.c_int
        if L.spincore_solver_abi_version()!=2: raise RuntimeError('SPINCORE_SOLVER_C_ABI_V2 required')
        L.spincore_state_create.argtypes=[C.POINTER(_Episode),C.c_uint64]; L.spincore_state_create.restype=C.c_void_p
        L.spincore_state_clone.argtypes=[C.c_void_p]; L.spincore_state_clone.restype=C.c_void_p
        L.spincore_state_destroy.argtypes=[C.c_void_p]
        L.spincore_state_terminal.argtypes=[C.c_void_p]; L.spincore_state_terminal.restype=C.c_int
        L.spincore_state_actor.argtypes=[C.c_void_p]; L.spincore_state_actor.restype=C.c_int32
        L.spincore_state_domain.argtypes=[C.c_void_p]; L.spincore_state_domain.restype=C.c_int
        L.spincore_state_legal_mask.argtypes=[C.c_void_p]; L.spincore_state_legal_mask.restype=C.c_uint8
        L.spincore_state_apply.argtypes=[C.c_void_p,C.c_uint8]; L.spincore_state_apply.restype=C.c_int
        L.spincore_state_neural_size.argtypes=[C.c_void_p]; L.spincore_state_neural_size.restype=C.c_size_t
        L.spincore_state_neural_copy.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_size_t]; L.spincore_state_neural_copy.restype=C.c_int
        L.spincore_state_terminal_chip_delta.argtypes=[C.c_void_p,C.POINTER(C.c_int32)]; L.spincore_state_terminal_chip_delta.restype=C.c_int
        L.spincore_last_error.restype=C.c_char_p
    def error(self): return (self.lib.spincore_last_error() or b'').decode('utf-8','replace')
    def create(self,e:Episode,seed:int)->'SolverState':
        x=_Episode(e.total_chips,int(e.game_is_hu),e.blind_index,e.small_blind,e.big_blind,(C.c_int32*3)(*e.stacks),e.dealer_id)
        p=self.lib.spincore_state_create(C.byref(x),C.c_uint64(seed))
        if not p: raise RuntimeError(self.error())
        return SolverState(self,p)

class SolverState:
    def __init__(self,owner:SolverLibrary,ptr:int): self.owner=owner; self.ptr=C.c_void_p(ptr)
    def close(self):
        if self.ptr:
            self.owner.lib.spincore_state_destroy(self.ptr); self.ptr=C.c_void_p()
    def __del__(self):
        try:self.close()
        except Exception:pass
    def clone(self):
        p=self.owner.lib.spincore_state_clone(self.ptr)
        if not p: raise RuntimeError(self.owner.error())
        return SolverState(self.owner,p)
    @property
    def terminal(self): return bool(self.owner.lib.spincore_state_terminal(self.ptr))
    @property
    def actor(self): return int(self.owner.lib.spincore_state_actor(self.ptr))
    @property
    def domain(self): return int(self.owner.lib.spincore_state_domain(self.ptr))
    def legal_actions(self):
        m=int(self.owner.lib.spincore_state_legal_mask(self.ptr)); return tuple(i for i in range(6) if m&(1<<i))
    def apply(self,a:int):
        if self.owner.lib.spincore_state_apply(self.ptr,a)!=0: raise RuntimeError(self.owner.error())
        return self
    def child(self,a:int): return self.clone().apply(a)
    def neural_bytes(self)->bytes:
        n=int(self.owner.lib.spincore_state_neural_size(self.ptr))
        if n<=0: raise RuntimeError(self.owner.error() or 'terminal state has no neural input')
        buf=(C.c_uint8*n)(); got=self.owner.lib.spincore_state_neural_copy(self.ptr,buf,n)
        if got!=n: raise RuntimeError(self.owner.error())
        return bytes(buf)
    def terminal_chip_delta(self):
        out=(C.c_int32*3)()
        if self.owner.lib.spincore_state_terminal_chip_delta(self.ptr,out)!=0: raise RuntimeError(self.owner.error())
        return tuple(int(x) for x in out)
