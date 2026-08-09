from pathlib import Path
import torch
from spincore.solver import Episode,SolverLibrary
from spincore_nn.codec import decode_spnniv1,collate_inputs
from spincore_nn import NetworkConfig,AdvantageNet,AveragePolicyNet
ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT/'build'/'libspincore_solver_c.so'
def obs():
    L=SolverLibrary(LIB);s=L.create(Episode(1500,True,0,10,20,(0,750,750),1,(0,)),2);b=s.neural_bytes();s.close();return b
def test_codec_shape_and_privacy():
    x=decode_spnniv1(obs());assert len(x.cards)==7 and sum(c>0 for c in x.cards)==2 and len(x.numeric)==16 and len(x.legal)==6
def test_collate_and_network_shapes():
    x=decode_spnniv1(obs());b=collate_inputs([x,x]);cfg=NetworkConfig(card_emb=4,cat_emb=3,hidden=20,gru_hidden=8,head_hidden=12);a=AdvantageNet(cfg);p=AveragePolicyNet(cfg);assert a(b).shape==(2,6);q=p.probabilities(b);assert q.shape==(2,6);assert torch.allclose(q.sum(1),torch.ones(2),atol=1e-6)
