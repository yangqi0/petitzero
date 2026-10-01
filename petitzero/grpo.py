"""PetitZero: explicit outcome-GRPO tensor primitives.

Pure functions, no model loading, no file writes, no training entry point.
This reference uses population group std and per-completion token means.
It is not a claim to reproduce current TRL defaults or an entire paper's recipe.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class LossSpec:
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    advantage_epsilon: float = 1e-4


def group_advantages(rewards: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    """[prompts,G] -> detached [prompts,G], population std; constants exactly0."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or rewards.requires_grad:
        raise ValueError('Require detached [prompts,G>=2] rewards')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Positive finite epsilon required')
    r = rewards.float()
    if not bool(torch.isfinite(r).all()) or not bool(((r == 0) | (r == 1)).all()):
        raise ValueError('This task uses finite binary rewards only')
    centered = r - r.mean(dim=1, keepdim=True)
    std = centered.square().mean(dim=1, keepdim=True).sqrt()
    return torch.where(std > 0, centered / (std + epsilon), torch.zeros_like(centered)).detach()


def action_labels(prompt_ids: Sequence[int], output_ids: Sequence[int],
                  eos_ids: Sequence[int] = (151645, 151643)) -> tuple[list[int], list[int]]:
    """Retain generated IDs, including actual terminator; never retokenize/append EOS.

    A generated151643 is a real sampled EOS action, unlike a padded151643 position.
    Full forward includes the last action; shift below leaves its own final logit unscored.
    """
    if not prompt_ids or not output_ids:
        raise ValueError('Nonempty prompt and generated action sequence required')
    if any(type(x) is not int or x < 0 for x in list(prompt_ids)+list(output_ids)):
        raise ValueError('IDs must be nonnegative Python integers')
    if any(x in eos_ids for x in output_ids[:-1]):
        raise ValueError('Actions cannot continue after the first native EOS')
    ids = list(prompt_ids)+list(output_ids)
    return ids, [-100]*len(prompt_ids)+list(output_ids)


def selected_action_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal shifted action log-probabilities in FP32, plus a position mask.

    The returned rectangular tensor is [B,T-1], with unscored entries0; the mask,
    NOT token identity, decides whether a generated EOS is an action or padding.
    Select positions BEFORE promoting full-vocabulary logits to float32.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError('Expected matching [B,T,V] logits/[B,T] labels')
    targets = labels[:,1:]
    mask = targets.ne(-100)
    if not bool(mask.any(dim=1).all()):
        raise ValueError('Every completion needs at least one scored action')
    z = logits[:,:-1,:][mask].float()
    y = targets[mask]
    if not bool(torch.isfinite(z).all()) or bool(((y<0)|(y>=z.shape[-1])).any()):
        raise ValueError('Nonfinite logits or invalid action ID')
    terms = F.log_softmax(z,dim=-1).gather(1,y[:,None]).squeeze(1)
    result = terms.new_zeros(mask.shape).masked_scatter(mask,terms)
    return result, mask


def grpo_loss(current: torch.Tensor, old: torch.Tensor, reference: torch.Tensor,
              advantages: torch.Tensor, mask: torch.Tensor, *,
              total_completions: int | None = None, spec: LossSpec = LossSpec()
              ) -> tuple[torch.Tensor,dict[str,torch.Tensor]]:
    """A microbatch contribution to sum_i mean_t(loss_it) / GLOBAL completion count.

    old/reference and advantages must already be detached snapshots. No KL added
    into advantages. The sampled k3 term is a differentiable regularizer on these
    sampled prefixes, NOT an exact full-vocabulary or current-policy KL measurement.
    """
    if current.ndim!=2 or old.shape!=current.shape or reference.shape!=current.shape or mask.shape!=current.shape:
        raise ValueError('Log-probabilities and mask must share [completions,positions]')
    if mask.dtype!=torch.bool or advantages.shape!=(current.shape[0],):
        raise ValueError('Boolean position mask and one advantage per completion required')
    if old.requires_grad or reference.requires_grad or advantages.requires_grad:
        raise ValueError('Only current-policy log-probabilities may have gradients')
    if len({x.device for x in (current,old,reference,advantages,mask)})!=1:
        raise ValueError('All inputs must share one device')
    if not (0<spec.clip_epsilon<1) or not (math.isfinite(spec.kl_beta) and spec.kl_beta>=0):
        raise ValueError('Invalid clip or KL coefficient')
    n=current.shape[0]; denom=n if total_completions is None else total_completions
    if type(denom) is not int or denom<n or n==0:
        raise ValueError('Use the fixed global completion count, not token or microbatch count')
    lengths=mask.sum(dim=1)
    if not bool((lengths>0).all()) or not bool(torch.isfinite(advantages).all()):
        raise ValueError('Invalid length or advantage')
    for x in (current,old,reference):
        if not bool(torch.isfinite(x[mask]).all()):raise ValueError('Nonfinite selected log-probability')
    # Clean masked slots BEFORE exp; masked NaN/Inf must not pollute derivatives.
    cur=torch.where(mask,current.float(),0.0)
    old=torch.where(mask,old.float(),0.0)
    ref=torch.where(mask,reference.float(),0.0)
    ratio=(cur-old).exp()
    adv=advantages.float()[:,None]
    plain=ratio*adv
    clipped=ratio.clamp(1-spec.clip_epsilon,1+spec.clip_epsilon)*adv
    surrogate=torch.minimum(plain,clipped)
    delta=ref-cur
    k3=torch.expm1(delta)-delta
    if not bool(torch.isfinite(ratio[mask]).all()) or not bool(torch.isfinite(k3[mask]).all()):
        raise FloatingPointError('Overflow; do not hide with an extra unrecorded clamp')
    pg_rows=(-surrogate*mask).sum(dim=1)/lengths
    kl_rows=(k3*mask).sum(dim=1)/lengths
    loss=(pg_rows+spec.kl_beta*kl_rows).sum()/denom
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
    # Outside clip interval != actively clipped surrogate: include sign.
    active=((advantages[:,None]>0)&(ratio>1+spec.clip_epsilon)) | ((advantages[:,None]<0)&(ratio<1-spec.clip_epsilon))
    metrics={'policy_loss_contribution':pg_rows.detach().sum()/denom,
             'sampled_k3_contribution':kl_rows.detach().sum()/denom,
             'ratio_selected':ratio.detach()[mask],
             'active_clip_count':active[mask].sum().detach(),
             'ratio_outside_interval_count':((ratio<1-spec.clip_epsilon)|(ratio>1+spec.clip_epsilon))[mask].sum().detach(),
             'action_count':mask.sum().detach(),
             'sequence_lengths':lengths.detach()}
    return loss,metrics
