from __future__ import annotations
import torch
def train_step(model,optimizer,batch,target,weights,kind:str)->float:
    model.train();optimizer.zero_grad(set_to_none=True);out=model(batch);w=weights/weights.mean().clamp_min(1e-12)
    if kind=='advantage':
        mask=batch['legal'].float();per=(((out-target)**2)*mask).sum(1)/mask.sum(1).clamp_min(1.0)
    elif kind=='strategy':
        logits=out.masked_fill(~batch['legal'],-1e9);logp=torch.log_softmax(logits,dim=-1);per=-(target*logp).sum(1)
    else:raise ValueError(kind)
    loss=(per*w).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.0);optimizer.step();return float(loss.detach().cpu())
