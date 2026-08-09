from __future__ import annotations
from dataclasses import dataclass,asdict
import torch
from torch import nn
@dataclass(frozen=True)
class NetworkConfig:
    card_emb:int=8;cat_emb:int=4;hidden:int=96;gru_hidden:int=32;head_hidden:int=64
    def to_dict(self):return asdict(self)
class _Net(nn.Module):
    def __init__(self,cfg:NetworkConfig):
        super().__init__();self.cfg=cfg
        self.card_emb=nn.Embedding(53,cfg.card_emb,padding_idx=0);self.cat_emb=nn.Embedding(32,cfg.cat_emb);self.hist_emb=nn.Embedding(64,cfg.cat_emb,padding_idx=0);self.gru=nn.GRU(cfg.cat_emb,cfg.gru_hidden,batch_first=True)
        dim=7*cfg.card_emb+8*cfg.cat_emb+16+cfg.gru_hidden
        self.body=nn.Sequential(nn.Linear(dim,cfg.hidden),nn.ReLU(),nn.Linear(cfg.hidden,cfg.head_hidden),nn.ReLU());self.head=nn.Linear(cfg.head_hidden,6)
    def forward(self,b):
        ce=self.card_emb(b['cards']).flatten(1);ca=self.cat_emb(b['categorical'].clamp(0,31)).flatten(1);he=self.hist_emb(b['history'].clamp(0,63));_,h=self.gru(he);x=torch.cat([ce,ca,b['numeric'],h[-1]],dim=1);return self.head(self.body(x))
class AdvantageNet(_Net):pass
class AveragePolicyNet(_Net):
    def probabilities(self,b):
        logits=self.forward(b);mask=b['legal'];logits=logits.masked_fill(~mask,-1e9);return torch.softmax(logits,dim=-1)
