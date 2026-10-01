"""Explicit dispatch to unchanged mathematical recipe helpers. Author: Yang Qi."""
from __future__ import annotations

import torch
from petitzero.grpo import group_advantages, grpo_loss
from petitzero.ppo import policy_loss, value_loss
from petitzero.ppo_runtime import rollout_targets, stored, diagnostics
from petitzero.rft import positive_ce_contribution
from petitzero.rft_runtime import select_responses

METHOD_RECIPES = {
    'grpo': {'roles': ['actor', 'reference'], 'denominator': '32 responses; mean tokens within each response',
             'group_size': 4, 'population_std_epsilon': 1e-4, 'policy_clip': .2, 'k3_loss_beta': .02},
    'rft': {'roles': ['actor'], 'denominator': 'whole-rollout positive response count; mean tokens per response',
            'duplicates_retained': True, 'zero_positive': 'skip before zero_grad/backward/step', 'reference': None},
    'ppo': {'roles': ['actor', 'reference', 'critic'], 'denominator': 'whole-rollout valid action count',
            'policy_clip': .2, 'value_clip': .2, 'value_factor': .5, 'gamma': 1., 'lambda': .95,
            'reference_reward_beta': .02, 'whitening_population_epsilon': 1e-8, 'critic_head': 'zero FP32 Linear(1536,1,bias=True)'},
}


METHOD_RECIPES['ppo_zero'] = {'roles':['actor','reference'],'denominator':'whole-rollout valid action count','policy_clip':.2,'gamma':1.,'lambda':.95,'reference_reward_beta':.02,'whitening_population_epsilon':1e-8,'value_mode':'fixed_zero_FP32','critic_head':None,'critic_optimizer':None}


def roles_for(method: str) -> tuple[str, ...]:
    if method not in METHOD_RECIPES:
        raise ValueError(f'Unknown method: {method}')
    return tuple(METHOD_RECIPES[method]['roles'])


def prepare_targets(method: str, records: list[dict]) -> dict:
    """Attach immutable recipe-specific targets once, before either pass."""
    rewards, positives = select_responses(records)
    total_actions = sum(len(record['output_ids']) for record in records)
    info = {'D': total_actions, 'N': len(records), 'N_positive': len(positives),
            'positive_indices': positives, 'recipe': METHOD_RECIPES[method]}
    if method == 'grpo':
        advantages = group_advantages(rewards.reshape(-1, 4)).reshape(-1).tolist()
        for record, advantage in zip(records, advantages):
            record['group_advantage'] = advantage
    elif method in ['ppo','ppo_zero']:
        if method == 'ppo_zero':
            for record in records:
                record['old_values']=torch.zeros(len(record['output_ids']),dtype=torch.float32).tolist()
                record['value_mode']='fixed_zero_FP32'
                record['value_dtype']='torch.float32'
        targets, mask, whitening = rollout_targets(records)
        info['whitening'] = whitening
        old_values = stored(records, 'old_values', mask)
        info['critic_prefit_token_weighted'] = {
            name: diagnostics(old_values, targets[key], mask)
            for name, key in [('lambda_target', 'returns'), ('monte_carlo_shaped_return', 'mc_returns')]
        }
        if method == 'ppo_zero':
            info['critic_prefit_token_weighted']=None
            info['value_mode']='fixed_zero_FP32; critic fit N/A'
        for index, record in enumerate(records):
            for name, matrix in targets.items():
                record[name] = matrix[index, mask[index]].tolist()
    elif method != 'rft':
        raise ValueError(method)
    for record in records:
        record['full_rollout_D'] = total_actions
        record['full_rollout_N_positive'] = len(positives)
        record['full_rollout_N'] = len(records)
    return info


def objective(method: str, role: str, current: torch.Tensor, mask: torch.Tensor,
              records: list[dict], totals: dict) -> tuple[torch.Tensor, dict]:
    """Each microbatch retains its method's FULL-rollout denominator."""
    if role == 'critic':
        assert method == 'ppo'
        return value_loss(current, stored(records, 'old_values', mask), stored(records, 'returns', mask),
                          mask, total_actions=totals['D'], clip=.2)
    if method in ['ppo','ppo_zero']:
        return policy_loss(current, stored(records, 'old_action_logps', mask),
                           stored(records, 'advantages', mask), mask, total_actions=totals['D'], clip=.2)
    if method == 'grpo':
        advantages = torch.tensor([record['group_advantage'] for record in records], device=current.device)
        return grpo_loss(current, stored(records, 'old_action_logps', mask),
                         stored(records, 'reference_action_logps', mask), advantages, mask,
                         total_completions=totals['N'])
    if method == 'rft':
        rewards = torch.tensor([record['score']['reward'] for record in records], device=current.device)
        return positive_ce_contribution(current, rewards, mask, total_accepted=totals['N_positive'])
    raise ValueError(method)
