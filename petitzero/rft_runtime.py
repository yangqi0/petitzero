"""Positive selection and real-slot control, shared by toy tests and M2.
Author: Yang Qi. No model loading or data IO.
"""
import torch
from .rft import accepted_completion_count


def nominal_lr(slot):
    assert 1 <= slot <= 256
    return 1e-5 * min(slot / 8, 1)


def select_responses(records):
    rewards=torch.tensor([r['score']['reward'] for r in records],dtype=torch.float32)
    count=accepted_completion_count(rewards)
    indices=[i for i,r in enumerate(records) if r['score']['reward']==1]
    assert len(indices)==count
    return rewards,indices # Stable original order; duplicate texts are retained.


def run_optimizer_slot(optimizer,parameters,rewards,slot,backward):
    """A zero-positive slot returns BEFORE any optimizer or backward operation.

    backward(N+) accumulates sixteen microbatch contributions with a common count.
    The caller owns nominal and actual counters; successful return means one step.
    """
    accepted=accepted_completion_count(rewards)
    info={'nominal_slot':slot,'learning_rate':nominal_lr(slot),'accepted_count':accepted,'skipped':accepted==0}
    if accepted==0:
        return info
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in parameters}
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:group['lr']=nominal_lr(slot)
    metrics=backward(accepted)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    assert torch.isfinite(norm)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in parameters)
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
    return info|metrics|{'gradient_norm_before_clip':norm.item()}
