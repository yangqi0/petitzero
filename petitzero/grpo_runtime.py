"""Exact stored-action batching and offline accounting. Author: Yang Qi."""
from collections import Counter
import statistics
import torch
from .grpo import action_labels


def action_batch(records,device='cpu'):
    examples=[action_labels(r['prompt_ids'],r['output_ids']) for r in records]
    width=max(len(ids) for ids,_ in examples)
    ids=[];labels=[];attention=[]
    for x,y in examples:
        n=len(x);ids.append(x+[151643]*(width-n));labels.append(y+[-100]*(width-n));attention.append([1]*n+[0]*(width-n))
    return {k:torch.tensor(v,device=device,dtype=torch.long) for k,v in [('input_ids',ids),('labels',labels),('attention_mask',attention)]}


def cached_logps(records,key,mask):
    out=torch.zeros(mask.shape,device=mask.device,dtype=torch.float32)
    for i,r in enumerate(records):
        values=r[key]
        assert len(values)==len(r['output_ids'])==int(mask[i].sum())
        out[i,mask[i]]=torch.tensor(values,device=mask.device,dtype=torch.float32)
    assert not out.requires_grad and not out.is_inference()
    return out


def reward_summary(records):
    """Four samples/group, with fixed scheduling strata, including constants."""
    result={}
    filters={'all':lambda r:True,
             'add_sub_solvable':lambda r:r['band']=='add_sub_solvable',
             'requires_mul_or_div':lambda r:r['band']=='requires_mul_or_div',
             'sft_exposed':lambda r:r['sft_exposed'],
             'sft_unexposed':lambda r:not r['sft_exposed']}
    for band in ['add_sub_solvable','requires_mul_or_div']:
        for exposed in [True,False]:
            filters[f'{band}/sft_exposed={exposed}']=lambda r,b=band,e=exposed:r['band']==b and r['sft_exposed']==e
    for name,predicate in filters.items():
        use=[r for r in records if predicate(r)]
        if not use:continue
        groups={i:[r for r in use if r['id']==i] for i in sorted({r['id'] for r in use})}
        assert all(len(g)==4 for g in groups.values())
        counts=[sum(r['score']['correct'] for r in g) for g in groups.values()]
        unique=[len({r['scoring_text'] for r in g}) for g in groups.values()]
        result[name]={'prompts':len(groups),'answers':len(use),'successes':sum(counts),'success_rate':sum(counts)/len(use),'any_of_four':sum(c>0 for c in counts)/len(groups),'allzero':sum(c==0 for c in counts),'allone':sum(c==4 for c in counts),'mixed':sum(0<c<4 for c in counts),'format_rate':sum(r['score']['format_valid'] for r in use)/len(use),'number_use_rate':sum(r['score']['numbers_valid'] for r in use)/len(use),'mean_unique_texts_per_four':statistics.mean(unique),'duplicate_texts_within_groups':sum(4-u for u in unique),'unique_texts_total':len({r['scoring_text'] for r in use}),'length_min':min(r['generated_tokens'] for r in use),'length_mean':statistics.mean(r['generated_tokens'] for r in use),'length_max':max(r['generated_tokens'] for r in use),'caps':sum(r['stop_reason']=='length_cap' for r in use),'first_failure_counts':dict(Counter(r['score']['reason'] for r in use))}
    return result
