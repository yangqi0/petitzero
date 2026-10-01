"""Pure tensor reference for PetitZero M3; no model loading or optimizer.

Compact [B,T] tensors contain action positions only. A mask is True for each real
sampled action, including an actual EOS, and False for right padding. Policy logp
and V(s_t) both use the prefix BEFORE action t. The caller freezes rollout targets.
"""
from __future__ import annotations
import math
import torch


def _mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[1] == 0:
        raise ValueError('Need bool [B,T] compact action mask')
    if mask.shape[0] == 0 or not bool(mask[:, 0].all()):
        raise ValueError('Each trajectory must contain a real first action')
    if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError('Action mask must be prefix contiguous before right padding')
    return mask.sum(1)


def _matrix(x: torch.Tensor, mask: torch.Tensor, name: str, detached: bool = False):
    if x.shape != mask.shape or x.device != mask.device or not x.is_floating_point():
        raise ValueError(f'{name}: require floating matrix aligned with action mask')
    if detached and x.requires_grad:
        raise ValueError(f'{name} must be detached rollout data')
    if not bool(torch.isfinite(x[mask]).all()):
        raise ValueError(f'{name}: nonfinite real action')


def sampled_rewards(old_logp: torch.Tensor, ref_logp: torch.Tensor,
                    task_rewards: torch.Tensor, mask: torch.Tensor,
                    beta: float = 0.02) -> torch.Tensor:
    """Detached k1 reward shaping; no hidden clamp or second KL loss.

    r_t=-beta*(old_logp-ref_logp); add binary task reward at the final action.
    Individual sampled log-ratios and shaped rewards can have either sign.
    """
    lengths = _mask(mask)
    _matrix(old_logp, mask, 'old_logp', True); _matrix(ref_logp, mask, 'ref_logp', True)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('Nonnegative finite beta required')
    if task_rewards.shape != (mask.shape[0],) or task_rewards.device != mask.device or task_rewards.requires_grad:
        raise ValueError('Need detached per-trajectory task rewards on same device')
    if not bool(((task_rewards == 0) | (task_rewards == 1)).all()):
        raise ValueError('Task rewards must be binary')
    with torch.no_grad():
        r = -beta * (torch.where(mask, old_logp.float(), 0.) - torch.where(mask, ref_logp.float(), 0.))
        r[torch.arange(mask.shape[0], device=mask.device), lengths - 1] += task_rewards.float()
    return r


def gae_targets(rewards: torch.Tensor, old_values: torch.Tensor,
                mask: torch.Tensor, terminated: torch.Tensor,
                bootstrap: torch.Tensor, *, gamma: float = 1.0,
                lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute detached GAE and lambda-return from ONE frozen rollout snapshot.

    terminated is one Boolean per trajectory. EOS and the declared finite-horizon
    task budget are terminal in M3 (bootstrap=0). A genuine unfinished fragment
    would use terminated=False and V(s_after_last); M3 does not collect fragments.
    No leakage from the next prompt, right padding, or future answer into V(s_t).
    """
    lengths = _mask(mask)
    _matrix(rewards, mask, 'rewards', True); _matrix(old_values, mask, 'old_values', True)
    B,T = mask.shape
    if terminated.shape != (B,) or terminated.dtype != torch.bool or terminated.device != mask.device:
        raise ValueError('Need per-trajectory bool termination flags')
    if bootstrap.shape != (B,) or bootstrap.device != mask.device or bootstrap.requires_grad:
        raise ValueError('Need detached per-trajectory bootstrap values')
    if not bool(torch.isfinite(bootstrap).all()) or bool((bootstrap[terminated] != 0).any()):
        raise ValueError('Terminal bootstrap must be zero; other bootstrap must be finite')
    if not (0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError('gamma/lam must lie in [0,1]')
    with torch.no_grad():
        rv = torch.where(mask, rewards.float(), 0.)
        vv = torch.where(mask, old_values.float(), 0.)
        advantage = torch.zeros_like(vv)
        for i in range(B):
            L=int(lengths[i]); carry=torch.zeros((),device=vv.device)
            for t in range(L-1,-1,-1):
                nxt = bootstrap[i].float() if t == L-1 else vv[i,t+1]
                delta = rv[i,t] + gamma*nxt - vv[i,t]
                carry = delta + gamma*lam*carry
                advantage[i,t] = carry
        returns = torch.where(mask, advantage + vv, 0.)
    return advantage, returns


def whiten_advantages(raw: torch.Tensor, mask: torch.Tensor,
                      eps: float = 1e-8) -> tuple[torch.Tensor, dict]:
    """ONE global valid-action mean/population variance, computed before microbatches.

    Never normalize per prompt or per microbatch. Raw advantages and unnormalized
    returns remain saved; only the policy's advantages are whitened.
    """
    _mask(mask); _matrix(raw,mask,'raw advantages',True)
    if not math.isfinite(eps) or eps <= 0: raise ValueError('Positive epsilon required')
    with torch.no_grad():
        a=raw.float()[mask];mean=a.mean();var=((a-mean)**2).mean()
        out=torch.where(mask,(torch.where(mask,raw.float(),mean)-mean)*torch.rsqrt(var+eps),0.)
    return out, {'mean':float(mean),'population_variance':float(var),'epsilon':eps,'actions':int(mask.sum())}


def _denom(mask, total_actions):
    _mask(mask)
    if type(total_actions) is not int or total_actions < int(mask.sum()):
        raise ValueError('Use full-rollout valid-action denominator, including this microbatch')


def policy_loss(current_logp: torch.Tensor, old_logp: torch.Tensor,
                advantages: torch.Tensor, mask: torch.Tensor, *,
                total_actions: int, clip: float = 0.2) -> tuple[torch.Tensor, dict]:
    """Microbatch contribution to a global VALID-TOKEN-MEAN PPO clipped loss."""
    _denom(mask,total_actions)
    _matrix(current_logp,mask,'current');_matrix(old_logp,mask,'old',True);_matrix(advantages,mask,'advantages',True)
    if not 0 < clip < 1: raise ValueError('clip must lie in (0,1)')
    logratio=torch.where(mask,current_logp.float(),0.)-torch.where(mask,old_logp.float(),0.)
    ratio=logratio.exp()
    if not bool(torch.isfinite(ratio[mask]).all()):raise ValueError('Nonfinite probability ratio; no silent clamp')
    a=torch.where(mask,advantages.float(),0.)
    unclipped=ratio*a;clipped=ratio.clamp(1-clip,1+clip)*a
    point=-torch.minimum(unclipped,clipped)
    loss=torch.where(mask,point,0.).sum()/total_actions
    return loss, {'loss_numerator':torch.where(mask,point,0.).sum().detach(),
                  'active_clip_count':((unclipped>clipped)&mask).sum().detach(),
                  'outside_count':(((ratio<1-clip)|(ratio>1+clip))&mask).sum().detach(),
                  'ratio_sum':ratio[mask].sum().detach(),'actions':mask.sum().detach()}


def value_loss(current_values: torch.Tensor, old_values: torch.Tensor,
               returns: torch.Tensor, mask: torch.Tensor, *, total_actions: int,
               clip: float = 0.2) -> tuple[torch.Tensor, dict]:
    """Independent critic objective: 0.5 * mean(max(raw error², clipped error²)).

    Its optimizer is separate from the actor's. No actor gradient from this loss.
    """
    _denom(mask,total_actions)
    _matrix(current_values,mask,'current values');_matrix(old_values,mask,'old values',True);_matrix(returns,mask,'returns',True)
    if not math.isfinite(clip) or clip<=0:raise ValueError('Positive value clip required')
    cur=torch.where(mask,current_values.float(),0.)
    old=torch.where(mask,old_values.float(),0.);target=torch.where(mask,returns.float(),0.)
    clipped=old+(cur-old).clamp(-clip,clip)
    raw=(cur-target)**2;limited=(clipped-target)**2
    point=0.5*torch.maximum(raw,limited)
    loss=torch.where(mask,point,0.).sum()/total_actions
    return loss, {'loss_numerator':torch.where(mask,point,0.).sum().detach(),
                  'unclipped_squared_error_sum':raw[mask].sum().detach(),
                  'value_clip_active_count':((limited>raw)&mask).sum().detach(),'actions':mask.sum().detach()}


def prefix_values(hidden: torch.Tensor, action_logit_positions: torch.Tensor,
                  action_mask: torch.Tensor, head: torch.nn.Module) -> torch.Tensor:
    """Select final-layer state representations BEFORE each sampled action.

    action_logit_positions comes from the existing C-1+j policy-score positions.
    This helper cannot verify a caller supplied correct offsets; integration must.
    """
    _mask(action_mask)
    if hidden.ndim!=3 or action_logit_positions.shape!=action_mask.shape or action_logit_positions.dtype!=torch.long:
        raise ValueError('Need [B,S,H] hidden states and [B,T] long logit positions')
    if hidden.shape[0]!=action_mask.shape[0] or len({hidden.device,action_mask.device,action_logit_positions.device})!=1:
        raise ValueError('Batch/device mismatch')
    p=action_logit_positions
    if bool(((p[action_mask]<0)|(p[action_mask]>=hidden.shape[1])).any()):raise ValueError('State position outside hidden sequence')
    p=torch.where(action_mask,p,0)
    selected=hidden.gather(1,p.unsqueeze(-1).expand(-1,-1,hidden.shape[-1]))
    # Disable an outer autocast for the deliberately FP32 scalar head.
    with torch.autocast(device_type=hidden.device.type,enabled=False):
        out=head(selected.float()).squeeze(-1)
    if out.shape!=action_mask.shape or not bool(torch.isfinite(out[action_mask]).all()):raise ValueError('Invalid scalar value head output')
    return torch.where(action_mask,out,0.)
