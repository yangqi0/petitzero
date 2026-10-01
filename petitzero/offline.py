"""Portable analysis of previously scored sufficient statistics; no rescoring."""
import csv
import json
import re
from collections import defaultdict, Counter
from statistics import mean, stdev
from pathlib import Path
from .metrics import KS, pass_k

BANDS = ('all', 'add_sub_solvable', 'requires_mul_or_div')
SEEDS = (17, 29, 43)
POPULATIONS = {'HISTORICAL_FINAL256_OBSERVED':256, 'CONFIRM128_RESERVED':128}
METRICS = ('greedy',) + tuple(f'pass@{k}' for k in KS)
CAVEAT = 'Learned PPO43 recovered after failure; original slot192 commit identity absent; historical roundtrip and uninterrupted equivalence unproven. Pair43 and learned/paired summaries inherit this limitation.'

def load_counts(path):
    rows = list(csv.DictReader(Path(path).read_text().splitlines()))
    seen = set()
    for r in rows:
        if r['policy'] != 'SFT' and not re.fullmatch(r'(GRPO|RFT|PPO|PPO_ZERO)(17|29|43)', r['policy']):
            raise ValueError('Unknown fixed endpoint')
        for key in ('greedy_correct','c','n'): r[key] = int(r[key])
        if r['population'] not in POPULATIONS or r['original_band'] not in BANDS[1:] or r['n'] != 32 or not 0 <= r['c'] <= 32 or r['greedy_correct'] not in (0,1):
            raise ValueError('Invalid sufficient statistic')
        if r['seed'] != ('' if r['policy']=='SFT' else r['policy'][-2:]): raise ValueError('Seed mismatch')
        key = (r['policy'],r['population'],r['question_id'])
        if key in seen: raise ValueError('Duplicate question')
        seen.add(key)
    expected={'SFT'}|{m+str(s) for m in ('GRPO','RFT','PPO','PPO_ZERO') for s in SEEDS}
    if {r['policy'] for r in rows} != expected: raise ValueError('Missing policy')
    for pop,n in POPULATIONS.items():
        reference={(r['question_id'],r['original_band']) for r in rows if r['policy']=='SFT' and r['population']==pop}
        if len(reference)!=n:raise ValueError('Missing population rows')
        if Counter(b for _,b in reference)!=Counter({b:n//2 for b in BANDS[1:]}):raise ValueError('Band counts')
        for policy in expected:
            if {(r['question_id'],r['original_band']) for r in rows if r['policy']==policy and r['population']==pop}!=reference:raise ValueError('Unmatched questions')
    return rows

def analyze(rows):
    groups=defaultdict(list)
    for r in rows:
        for band in ('all',r['original_band']):groups[r['policy'],r['population'],band].append(r)
    points=[]; hist=[]; lookup={}
    for (policy,pop,band),rr in sorted(groups.items()):
        for metric in METRICS:
            vals=[r['greedy_correct'] if metric=='greedy' else pass_k(r['c'],int(metric[5:])) for r in rr]
            value=100*mean(vals);lookup[policy,pop,band,metric]=value
            points.append(dict(policy=policy,population=pop,band=band,metric=metric,questions=len(rr),percent=value))
        bins=Counter(r['c'] for r in rr)
        hist.extend(dict(policy=policy,population=pop,band=band,c=c,questions=bins[c]) for c in range(33))
    summary=[];paired=[];secondary=[]
    for pop in POPULATIONS:
        for band in BANDS:
            for metric in METRICS:
                for method in ('GRPO','RFT','PPO','PPO_ZERO'):
                    v=[lookup[method+str(s),pop,band,metric] for s in SEEDS]
                    summary.append(dict(method=method,population=pop,band=band,metric=metric,mean_percent=mean(v),sample_SD_pp=stdev(v),seed_points=v))
                a=[lookup['PPO'+str(s),pop,band,metric] for s in SEEDS]
                b=[lookup['PPO_ZERO'+str(s),pop,band,metric] for s in SEEDS]
                d=[y-x for x,y in zip(a,b)]
                paired.append(dict(population=pop,band=band,metric=metric,mean_difference_pp=mean(d),sample_SD_difference_pp=stdev(d),seed_differences=d))
                secondary.append(dict(population=pop,band=band,metric=metric,matched_seeds=[17,29],learned_mean_percent=mean(a[:2]),learned_SD_pp=stdev(a[:2]),zero_mean_percent=mean(b[:2]),zero_SD_pp=stdev(b[:2]),paired_mean_difference_pp=mean(d[:2]),paired_SD_difference_pp=stdev(d[:2])))
    return dict(schema='PZ_M12_OFFLINE_V1',caveat=CAVEAT,seed_points=points,method_summaries=summary,primary_zero_minus_learned=paired,secondary_matched17_29=secondary,histograms=hist)

def write_results(data,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'statistics.json').write_text(json.dumps(data,indent=2)+'\n')
    for name in ('seed_points','method_summaries','primary_zero_minus_learned','secondary_matched17_29','histograms'):
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[name][0]));w.writeheader();w.writerows(data[name])

def figures(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    lookup={(r['policy'],r['population'],r['band'],r['metric']):r['percent'] for r in data['seed_points']}
    colors={'GRPO':'C0','RFT':'C1','PPO':'C2','PPO_ZERO':'C3'};markers={17:'o',29:'s',43:'^'}
    # Adapted from M10 make_figures: same pools, bands, seed markers and ddof=1.
    for pop in POPULATIONS:
        fig,axes=plt.subplots(1,3,figsize=(15,5.2),dpi=120)
        for ax,band in zip(axes,BANDS):
            ax.plot(KS,[lookup['SFT',pop,band,f'pass@{k}'] for k in KS],'k--',marker='x')
            for method,color in colors.items():
                values=[[lookup[method+str(s),pop,band,f'pass@{k}'] for s in SEEDS] for k in KS]
                ax.errorbar(KS,[mean(v) for v in values],yerr=[stdev(v) for v in values],color=color,capsize=2,lw=1.3)
                for s,shift in zip(SEEDS,(.96,1.,1.04)):
                    ax.scatter([k*shift for k in KS],[lookup[method+str(s),pop,band,f'pass@{k}'] for k in KS],color=color,marker=markers[s],s=15)
            ax.set_xscale('log',base=2);ax.set_xticks(KS,[str(k) for k in KS]);ax.set_ylim(-3,103)
            ax.set(title={'all':'All questions','add_sub_solvable':'Add/sub solvable','requires_mul_or_div':'Requires mul/div'}[band],xlabel='k (same 32-response pool)',ylabel='Question-average pass@k (%)');ax.grid(alpha=.15)
        handles=[Line2D([],[],color=c,label=m) for m,c in colors.items()]+[Line2D([],[],color='black',ls='--',label='SFT (one pool)')]+[Line2D([],[],color='black',ls='',marker=markers[s],label=f'seed {s}') for s in SEEDS]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.06),ncol=8,fontsize=8)
        fig.suptitle('CONFIRM128: first evaluated in M10, now observed' if pop.startswith('CONFIRM') else 'Historical FINAL256: previously observed',fontsize=13)
        fig.text(.5,.018,'Means ± sample SD (3 seeds); includes recovered learned PPO43. M11 zero protocol set after both populations were observed.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.16,1,.94));fig.savefig(out/(pop+'.png'));plt.close(fig)
