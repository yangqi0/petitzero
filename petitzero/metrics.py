"""Prespecified question-level statistics; no model imports. Author: Yang Qi."""
from math import comb
from statistics import mean,stdev
KS = (1, 2, 4, 8, 16, 32)

def pass_k(c,k,n=32):
 assert type(c)is int and type(k)is int and 0<=c<=n and 1<=k<=n
 return 1.-comb(n-c,k)/comb(n,k) if n-c>=k else 1.

def summarize_question(records):
 greedy=[r for r in records if r['mode']=='greedy'];sample=[r for r in records if r['mode']=='sample']
 assert len(greedy)==1 and len(sample)==32 and sorted(r['sample_index'] for r in sample)==list(range(32))
 c=sum(bool(r['score']['correct']) for r in sample)
 return {'greedy':int(greedy[0]['score']['correct']),'c':c,'n':32,'any32':int(c>0),**{f'pass@{k}':pass_k(c,k) for k in KS}}
def seed_summary(points):
 assert len(points)==3
 return {'mean':mean(points),'sample_sd_ddof1':stdev(points),'seed_points':points}
