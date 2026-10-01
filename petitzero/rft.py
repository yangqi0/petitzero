"""Pure positive-only CE reference for PetitZero's proposed online-RFT control.

No model loading, dataset IO, optimizers or automatic training. The caller preserves
all rollout records and passes the number of correct answers in the WHOLE batch.
"""
from __future__ import annotations
import torch


def accepted_completion_count(rewards: torch.Tensor) -> int:
    if rewards.ndim != 1 or rewards.requires_grad or rewards.numel() == 0:
        raise ValueError('Require nonempty detached [completions] binary rewards')
    if not bool(torch.isfinite(rewards).all()) or not bool(((rewards == 0) | (rewards == 1)).all()):
        raise ValueError('Strict rewards must be finite and binary')
    return int((rewards == 1).sum().item())


def positive_ce_contribution(current: torch.Tensor, rewards: torch.Tensor,
                             mask: torch.Tensor, *, total_accepted: int
                             ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum_i r_i mean_t[-logp_it] / accepted_count_of_entire_rollout_batch.

    All-zero microbatches are permitted when the FULL batch has positives. A full
    batch with zero positives must be skipped by the trainer before this call;
    in particular it must NOT execute AdamW.step with zero gradients/momentum.
    """
    if current.ndim != 2 or mask.shape != current.shape or mask.dtype != torch.bool:
        raise ValueError('Require [B,T] logprobs and boolean action mask')
    if rewards.shape != (current.shape[0],) or len({x.device for x in (current, rewards, mask)}) != 1:
        raise ValueError('Require one reward per completion on the same device')
    npos = accepted_completion_count(rewards)
    if type(total_accepted) is not int or total_accepted <= 0 or total_accepted < npos:
        raise ValueError('Use positive count of full batch; zero requires trainer skip')
    lengths = mask.sum(1)
    if not bool((lengths > 0).all()) or not bool(torch.isfinite(current[mask]).all()):
        raise ValueError('Each answer needs finite scored actions, including actual EOS')
    safe = torch.where(mask, current.float(), 0.0)
    sequence_nll = -safe.sum(1) / lengths
    positive = rewards.float().detach()
    numerator = (sequence_nll * positive).sum()
    loss = numerator / total_accepted
    return loss, {'positive_sequence_nll_sum': numerator.detach(),
                  'accepted_in_microbatch': positive.sum().detach(),
                  'all_action_tokens': lengths.sum().detach(),
                  'accepted_action_tokens': (lengths * positive).sum().detach()}
