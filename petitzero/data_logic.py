"""Extracted/adapted M1A numerical-multiset partition logic; no held-out payloads."""
from itertools import combinations_with_replacement
from .calibration import digest, build_calibration
from .countdown import reachable_targets, check_expression
SEED=20260929

def partition_groups(excluded):
    old_groups={tuple(sorted(x)) for x in excluded}
    if len(old_groups)!=64:raise ValueError('Provide the 64 distinct historical calibration groups')
    triples=[t for t in combinations_with_replacement(range(1,21),3) if t not in old_groups]
    triples.sort(key=lambda t:digest([SEED,'m1-group-split',list(t)]))
    if len(triples)!=1476:raise ValueError('Calibration groups outside task universe')
    return {'train':triples[:1024],'dev':triples[1024:1088],'final':triples[1088:1344],'unused':triples[1344:]}

def target_for(split,index,triple):
    """Construction primitive; does not serialize any witness or held-out bank."""
    full=reachable_targets(triple);simple=reachable_targets(triple,'+-')
    band=('add_sub_solvable','requires_mul_or_div')[index%2]
    candidates=sorted(simple if index%2==0 else set(full)-set(simple))
    if not candidates:raise ValueError('Fixed split/band has no eligible target')
    target=min(candidates,key=lambda v:digest([SEED,'m1-target',split,list(triple),v]))
    return {'numbers':list(triple),'target':target,'band':band}

def cold_start_subset(train):
    warm=[]
    for band in ('add_sub_solvable','requires_mul_or_div'):
        subset=sorted((r for r in train if r['band']==band),key=lambda r:digest([SEED,'cold-subset',r['number_group']]))
        warm.extend(subset[:256])
    if len(warm)!=512:raise ValueError('Need 256 TRAIN rows per band')
    return sorted(warm,key=lambda r:digest([SEED,'cold-display',r['id']]))

def synthetic_fixture():
    """Hand-authored algebra, not a model response or a held-out observation."""
    return {'kind':'synthetic_only','numbers':[2,3,4],'target':14,'expression':'2*(3+4)','correct':check_expression('2*(3+4)',[2,3,4],14).correct}
