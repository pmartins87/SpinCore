from __future__ import annotations
from dataclasses import dataclass
import struct
import torch
@dataclass(frozen=True)
class DecodedInput:
    cards:tuple[int,...];numeric:tuple[float,...];categorical:tuple[int,...];legal:tuple[int,...];history:tuple[int,...];history_len:int
SIZE=126
def decode_spnniv1(b:bytes)->DecodedInput:
    if len(b)!=SIZE or b[:8]!=b'SPNNIV1\x00':raise ValueError('bad SPNNIV1 payload')
    p=8;cards=tuple(b[p:p+7]);p+=7;numeric=struct.unpack_from('<16f',b,p);p+=64;cat=tuple(b[p:p+8]);p+=8;legal=tuple(b[p:p+6]);p+=6;hlen=b[p];p+=1;hist=tuple(b[p:p+32]);
    if hlen>32:raise ValueError('bad history length')
    return DecodedInput(cards,tuple(float(x) for x in numeric),cat,legal,hist,int(hlen))
def collate_inputs(items:list[DecodedInput],device='cpu'):
    return {'cards':torch.tensor([x.cards for x in items],dtype=torch.long,device=device),'numeric':torch.tensor([x.numeric for x in items],dtype=torch.float32,device=device),'categorical':torch.tensor([x.categorical for x in items],dtype=torch.long,device=device),'legal':torch.tensor([x.legal for x in items],dtype=torch.bool,device=device),'history':torch.tensor([x.history for x in items],dtype=torch.long,device=device),'history_len':torch.tensor([x.history_len for x in items],dtype=torch.long,device=device)}
