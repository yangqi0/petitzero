"""Append-only work identity and atomic completed-boundary checkpoints; no torch."""
import json,os,time,fcntl
from contextlib import contextmanager
from pathlib import Path
from .protocol import digest,file_hash,read,require,safe_file,CHECKPOINT,SCHEMA

class AmbiguousWork(RuntimeError):pass

def snapshot(value):
    """Detach nested data using the exact JSON representation persisted by append."""
    return json.loads(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False))

def sync_dir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    with tmp.open('x') as f:json.dump(value,f,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);sync_dir(p.parent)
def append(p,value):
    with Path(p).open('a') as f:f.write(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    sync_dir(Path(p).parent)

@contextmanager
def locked(root):
    with (Path(root)/'execution.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

class Journal:
    def __init__(self,root,run,requests,reconcile=False):
        self.path=Path(root)/'journal.jsonl';self.run=run;self.requests=requests
        raw=self.path.read_bytes() if self.path.exists() else b''
        if raw and not raw.endswith(b'\n'):raise AmbiguousWork('Torn journal tail; no replay')
        try:self.events=[json.loads(x) for x in raw.splitlines()]
        except (ValueError,UnicodeDecodeError) as e:raise AmbiguousWork('Malformed journal; no replay') from e
        previous=None
        for i,e in enumerate(self.events):
            require(e['seq']==i and e['run_id']==run['run_id'] and e['previous']==previous,'Journal chain/run mismatch')
            previous=digest(e)
        self.scan(reconcile)
    def add(self,event,**values):
        e=snapshot({'seq':len(self.events),'run_id':self.run['run_id'],'previous':digest(self.events[-1]) if self.events else None,'event':event,'time_ns':time.time_ns(),**values})
        append(self.path,e);self.events.append(e)
    def scan(self,reconcile=False):
        self.records=[];self.updates=[];pending=None;record=None;updating=None;last_slot=0;checkpoint_pending=None
        for e in self.events:
            kind=e['event']
            if kind=='request_start':
                if pending is not None or updating is not None:raise AmbiguousWork('Overlapping work')
                require(e['request']==self.requests[len(self.records)],'Request identity/cursor mismatch')
                pending=e['request'];record=None
            elif kind=='response':
                require(pending is not None and record is None and e['request']==pending,'Unbound response')
                require(e['record']['request']==pending,'Response/request mismatch');record=e['record']
            elif kind=='request_commit':
                require(pending is not None and record is not None and e['request']==pending and e['response_hash']==digest(record),'Invalid response commit')
                self.records.append(snapshot(record));pending=None;record=None
            elif kind=='update_start':
                if pending is not None or updating is not None:raise AmbiguousWork('Overlapping update')
                require(e['slot']==last_slot+1,'Update cursor mismatch');updating=e
            elif kind=='update_commit':
                require(updating is not None and e['slot']==updating['slot'],'Unbound update commit')
                self.updates.append(e);last_slot=e['slot'];updating=None
            elif kind=='checkpoint_start':
                if checkpoint_pending is not None:raise AmbiguousWork('Incomplete checkpoint publication')
                checkpoint_pending=e['name']
            elif kind=='checkpoint_commit':
                require(checkpoint_pending==e['name'],'Unbound checkpoint commit');checkpoint_pending=None
        if updating is not None:raise AmbiguousWork('Partially committed update; stop without rollback/replay')
        if checkpoint_pending is not None:raise AmbiguousWork('Incomplete checkpoint publication; preserve files, do not replay')
        if pending is not None:
            if record is None:raise AmbiguousWork('Started request has no durable complete response; no replay')
            if not reconcile:raise AmbiguousWork('Complete response awaits model-free reconciliation')
            self.add('request_commit',request=pending,response_hash=digest(record),reconciled_without_model=True)
            self.records.append(snapshot(record))
    def response(self,request,record):
        require(record['request']==request,'Backend returned wrong request')
        self.add('response',request=request,record=record)
        self.add('request_commit',request=request,response_hash=digest(record));self.records.append(snapshot(record))
    def counters(self):
        from collections import Counter
        c=Counter(e['event'] for e in self.events)
        return {'generation_attempts':c['request_start'],'committed_responses':c['request_commit'],'update_attempts':c['update_start'],'committed_slots':c['update_commit'],'errors':c['error'],'model_constructions':c['model_construction_start'],'tokenizer_constructions':c['tokenizer_construction_start'],'optimizer_constructions':c['optimizer_construction_start'],'optimizer_step_attempts':c['optimizer_step_start'],'optimizer_steps':c['optimizer_step_commit'],'model_forward_entries':c['model_forward_start'],'checkpoint_layer_recomputations':c['checkpoint_layer_recomputation'],'value_head_constructions':c['value_head_construction_start'],'generated_actions':sum(len(r.get('output_ids',[])) for r in self.records),'actor_optimizer_steps':sum(e['event']=='optimizer_step_commit' and e.get('role')=='actor' for e in self.events),'critic_optimizer_steps':sum(e['event']=='optimizer_step_commit' and e.get('role')=='critic' for e in self.events)}

def publish(root,run,journal,backend,completed):
    root=Path(root);name=f'rollout{completed:04d}';folder=root/'checkpoints';folder.mkdir(exist_ok=True);sync_dir(root)
    stage=folder/(name+'.staging');final=folder/name
    require(not stage.exists() and not final.exists(),'Checkpoint destination already exists')
    journal.add('checkpoint_start',name=name);stage.mkdir();sync_dir(folder)
    metadata=backend.export(stage)
    identity={'schema':CHECKPOINT,'run_id':run['run_id'],'spec_hash':run['spec_hash'],'roles':run['spec']['roles'],'completed_rollouts':completed,'nominal_slot':2*completed,'requests_committed':len(journal.records),'next_request':journal.requests[len(journal.records)] if len(journal.records)<len(journal.requests) else None,'actual_steps':dict(backend.steps),'backend':metadata,'journal_prefix_length':len(journal.events),'journal_prefix_sha256':digest(journal.events),'files':{str(p.relative_to(stage)):file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}}
    require(identity['requests_committed']==32*completed and len(journal.updates)==2*completed,'Checkpoint is not a completed rollout boundary')
    require(set(identity['actual_steps'])==({'actor','critic'} if 'critic' in identity['roles'] else {'actor'}),'Wrong step roles')
    require(set(metadata['trainable_roles'])==set(identity['actual_steps']),'Wrong serialized role set')
    atomic(stage/'identity.json',identity)
    for p in stage.rglob('*'):
        require(not p.is_symlink(),'No checkpoint symlinks')
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    for p in sorted([p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):sync_dir(p)
    sync_dir(stage);os.rename(stage,final);sync_dir(folder)
    journal.add('checkpoint_commit',name=name,identity_sha256=file_hash(final/'identity.json'))
    return final

def validate_checkpoint(root,checkpoint,run,journal,payload_kind="NATIVE_FM_V1"):
    root=Path(root);checkpoint=Path(checkpoint)
    require(checkpoint.parent==root/'checkpoints','Resume uses a committed checkpoint in its original run directory')
    if list((root/'checkpoints').glob('*.staging')):raise AmbiguousWork('Staged checkpoint retained; no replay')
    identity=read(safe_file(checkpoint,'identity.json'))
    require(identity['schema']==CHECKPOINT,'Only new portable checkpoint schema supported')
    require(identity['run_id']==run['run_id'] and identity['spec_hash']==run['spec_hash'] and identity['roles']==run['spec']['roles'],'Checkpoint source/data/config/role/run identity mismatch')
    commits=[e for e in journal.events if e['event']=='checkpoint_commit']
    require(commits and commits[-1]['name']==checkpoint.name and commits[-1]['identity_sha256']==file_hash(checkpoint/'identity.json'),'Only the latest journal-committed checkpoint may resume')
    files=identity['files'];require(isinstance(files,dict) and files,'Empty checkpoint')
    require(identity['backend'].get('kind')==payload_kind,'Checkpoint payload boundary kind mismatch')
    if payload_kind=='NATIVE_FM_V1':
        required={'state.pt','actor/adapter_config.json','actor/adapter_model.safetensors'}
        if 'critic' in identity['roles']:required|={'head.pt','critic/adapter_config.json','critic/adapter_model.safetensors'}
        require(required<=set(files),'Missing native checkpoint role/state payload')
        require(('head.pt' in files)==('critic' in identity['roles']),'Unexpected/missing critic head')
        require(not any(name.startswith(('reference/','critic/')) for name in files) if 'critic' not in identity['roles'] else not any(name.startswith('reference/') for name in files),'Unexpected checkpoint model role')
    require(set(identity['backend']['trainable_roles'])==set(identity['actual_steps']),'Serialized role mismatch')
    actual={str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()}
    require(actual==set(files)|{'identity.json'},'Checkpoint payload file set mismatch')
    for name,h in files.items():require(file_hash(safe_file(checkpoint,name))==h,'Checkpoint payload hash mismatch: '+name)
    require(set(identity['actual_steps'])==({'actor','critic'} if 'critic' in identity['roles'] else {'actor'}),'Checkpoint role mismatch')
    n=identity['journal_prefix_length'];require(digest(journal.events[:n])==identity['journal_prefix_sha256'],'Checkpoint journal binding mismatch')
    require(len(journal.records)==identity['requests_committed']==32*identity['completed_rollouts'] and len(journal.updates)==identity['nominal_slot']==2*identity['completed_rollouts'],'Unsaved completed work beyond checkpoint; no rollback/replay')
    wanted=journal.requests[len(journal.records)] if len(journal.records)<len(journal.requests) else None
    require(wanted==identity['next_request'],'Next request mismatch')
    require(identity['actual_steps']==(journal.updates[-1]['actual_steps'] if journal.updates else {r:0 for r in identity['actual_steps']}),'Checkpoint/journal actual-step mismatch')
    # Any started real work after the committed boundary is unsafe, even with an error record.
    tail=journal.events[commits[-1]['seq']+1:]
    if any(e['event'] in ('request_start','update_start','optimizer_step_start') for e in tail):raise AmbiguousWork('Work after checkpoint; no replay')
    return identity
