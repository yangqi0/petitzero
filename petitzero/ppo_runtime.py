"""Narrow stored-trajectory and paired-optimizer helpers for M3. Author: Yang Qi."""
import torch
from .grpo_runtime import action_batch
from .ppo import sampled_rewards,gae_targets,whiten_advantages


def compact_batch(records,device='cpu'):
    batch=action_batch(records,device);width=max(len(r['output_ids']) for r in records)
    positions=[];mask=[]
    for r in records:
        c=len(r['prompt_ids']);n=len(r['output_ids'])
        positions.append(list(range(c-1,c+n-1))+[0]*(width-n))
        mask.append([True]*n+[False]*(width-n))
    return batch,torch.tensor(positions,device=device,dtype=torch.long),torch.tensor(mask,device=device,dtype=torch.bool)


def stored(records,key,mask):
    out=torch.zeros(mask.shape,device=mask.device,dtype=torch.float32)
    for i,r in enumerate(records):
        n=len(r['output_ids']);assert n==int(mask[i].sum())==len(r[key])
        out[i,:n]=torch.tensor(r[key],device=mask.device,dtype=torch.float32)
    assert not out.requires_grad and not out.is_inference()
    return out


def terminal_kind(output_ids):
    if not output_ids:raise ValueError('An incomplete fragment is not an episode')
    if any(a in [151645,151643] for a in output_ids[:-1]):raise ValueError('Action after terminal EOS')
    if output_ids[-1] in [151645,151643]:return 'NATIVE_EOS_TERMINAL'
    if len(output_ids)==128:return 'TASK_HORIZON_TERMINAL'
    raise ValueError('Unfinished fragment cannot be trained as a terminal episode')


def rollout_targets(records):
    _,_,mask=compact_batch(records)
    for r in records:assert r['terminal_kind']==terminal_kind(r['output_ids'])
    old=stored(records,'old_action_logps',mask);ref=stored(records,'reference_action_logps',mask);v=stored(records,'old_values',mask)
    rewards=torch.tensor([r['score']['reward'] for r in records],dtype=torch.float32)
    shaped=sampled_rewards(old,ref,rewards,mask,beta=.02)
    terminal=torch.ones(len(records),dtype=torch.bool);bootstrap=torch.zeros(len(records))
    raw,returns=gae_targets(shaped,v,mask,terminal,bootstrap,gamma=1.,lam=.95)
    advantage,white=whiten_advantages(raw,mask,eps=1e-8)
    _,mc=gae_targets(shaped,v,mask,terminal,bootstrap,gamma=1.,lam=1.)
    delta=torch.zeros_like(v)
    for i,r in enumerate(records):
        n=len(r['output_ids']);nextvalues=torch.cat([v[i,1:n],torch.zeros(1)])
        delta[i,:n]=shaped[i,:n]+nextvalues-v[i,:n]
    return {'k1':old-ref,'shaped_rewards':shaped,'td_deltas':delta,'raw_advantages':raw,'advantages':advantage,'returns':returns,'mc_returns':mc},mask,white


def diagnostics(values,targets,mask):
    v=values[mask].float();y=targets[mask].float();error=v-y;var=y.var(unbiased=False)
    return {'mse':float(error.square().mean()),'target_population_variance':float(var),'explained_variance':float(1-error.var(unbiased=False)/var) if var>0 else None,'value_mean':float(v.mean()),'value_min':float(v.min()),'value_max':float(v.max()),'target_mean':float(y.mean()),'target_min':float(y.min()),'target_max':float(y.max()),'actions':int(mask.sum())}


def paired_step(actor_optimizer,critic_optimizer,actor_parameters,critic_parameters,round_index,actor_backward,critic_backward):
    assert 1<=round_index<=256
    assert {id(p) for p in actor_parameters}.isdisjoint({id(p) for p in critic_parameters})
    for opt,params,peak in [(actor_optimizer,actor_parameters,1e-5),(critic_optimizer,critic_parameters,1e-4)]:
        assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in params}
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=peak*min(round_index/8,1)
    actor_metrics=actor_backward()
    assert all(p.grad is None for p in critic_parameters),'Actor loss reached critic'
    critic_metrics=critic_backward()
    norms=[]
    for opt,params in [(actor_optimizer,actor_parameters),(critic_optimizer,critic_parameters)]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True);norms.append(float(norm))
    actor_optimizer.step();critic_optimizer.step()
    for opt,params in [(actor_optimizer,actor_parameters),(critic_optimizer,critic_parameters)]:
        assert all(torch.isfinite(p).all() for p in params)
        for s in opt.state.values():
            assert s['step'].item()==round_index
            assert all(s[k].dtype==torch.float32 and torch.isfinite(s[k]).all() for k in ['exp_avg','exp_avg_sq'])
    return {'actor':actor_metrics,'critic':critic_metrics,'actor_gradient_norm':norms[0],'critic_gradient_norm':norms[1],'actor_lr':actor_optimizer.param_groups[0]['lr'],'critic_lr':critic_optimizer.param_groups[0]['lr']}
