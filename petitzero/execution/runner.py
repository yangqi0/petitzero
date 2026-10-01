"""Real operational dispatch with an injectable non-model boundary for CPU tests."""
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
from petitzero.metrics import pass_k
from .protocol import preflight,manifest,read,digest,require,SCHEMA
from .durable import Journal,atomic,locked,publish,validate_checkpoint,sync_dir

def summarize_evaluation(records,samples):
    """Single NEW_EXECUTION actor only; independent of historical 13-policy tables."""
    groups=defaultdict(list)
    for r in records:
        for band in ('all',r['band']):groups[band].append(r)
    result={}
    for band,rr in groups.items():
        questions=defaultdict(list)
        for r in rr:questions[r['request']['id']].append(r)
        counts=[];greedy=[]
        for rows in questions.values():
            g=[r for r in rows if r['request']['mode']=='greedy'];s=[r for r in rows if r['request']['mode']=='sample']
            require(len(g)==1 and len(s)==samples,'Incomplete single-actor evaluation question')
            require(sorted(r['request']['sample_index'] for r in s)==list(range(samples)),'Evaluation sample indices')
            greedy.append(int(g[0]['score']['correct']));counts.append(sum(int(r['score']['correct']) for r in s))
        result[band]={'questions':len(questions),'greedy':mean(greedy),'samples_per_question':samples,'histogram':{str(c):counts.count(c) for c in range(samples+1)},'pass_k':{str(k):mean(pass_k(c,k,samples) for c in counts) for k in (1,2,4,8,16,32) if k<=samples}}
    return {'namespace':'NEW_EXECUTION','not_historical_study':True,'bands':result}

def execute(args,backend_factory=None):
    context=preflight(args) # Always before importing the model backend or reserving output.
    root=context['output'];resume=context['checkpoint'];continuing=resume is not None or getattr(args,'continue_evaluation',False)
    if continuing:
        run=read(root/'manifest.json')
        require(run['schema']==SCHEMA and run['spec_hash']==digest(run['spec'])==context['spec_hash'],'Source/data/assets/config/roles/budget changed; cannot resume')
        require(read(root/'requests.json')==context['requests'],'Request schedule changed')
    else:
        root.mkdir(parents=True,exist_ok=False);sync_dir(root.parent);run=manifest(context)
        atomic(root/'manifest.json',run);atomic(root/'requests.json',context['requests'])
    with locked(root):
        journal=Journal(root,run,context['requests'],reconcile=continuing)
        identity=validate_checkpoint(root,resume,run,journal,payload_kind=getattr(backend_factory,'checkpoint_kind','NATIVE_FM_V1')) if resume else None
        if identity and args.stop_after is not None:
            require(args.stop_after>identity['completed_rollouts'],'stop-after must advance beyond the committed boundary')
        if not continuing:require(not journal.events,'New run has existing work')
        if args.operation=='evaluate' and len(journal.records)==len(context['requests']):
            atomic(root/'evaluation_summary.json',summarize_evaluation(journal.records,args.samples));return {'status':'COMPLETE_NO_NEW_MODEL_WORK','run_id':run['run_id']}
        if identity and identity['completed_rollouts']==context['spec']['rollouts']:
            return {'status':'COMPLETE_NO_NEW_MODEL_WORK','run_id':run['run_id']}
        backend=None
        try:
            journal.add('attempt_start',resume_checkpoint=resume.name if resume else None)
            if backend_factory is None:
                from .native import NativeBackend
                backend_factory=NativeBackend
            backend=backend_factory(context,run,journal,resume,identity)
            rows={r['id']:r for r in context['rows']}
            if args.operation=='evaluate':
                for request in context['requests'][len(journal.records):]:
                    journal.add('request_start',request=request) # Durable before the only generation boundary.
                    record=backend.generate(rows[request['id']],request)
                    journal.response(request,record)
                summary=summarize_evaluation(journal.records,args.samples);atomic(root/'evaluation_summary.json',summary)
            else:
                completed=identity['completed_rollouts'] if identity else 0
                if identity is None:publish(root,run,journal,backend,0)
                end=args.stop_after if args.stop_after is not None else args.rollouts
                require(end>completed,'stop-after must advance beyond the committed boundary')
                for b in range(completed+1,end+1):
                    records=[]
                    for request in context['requests'][(b-1)*32:b*32]:
                        journal.add('request_start',request=request)
                        record=backend.generate(rows[request['id']],request);journal.response(request,record);records.append(record)
                    records,totals=backend.prepare(records,b)
                    for pass_index in (1,2):
                        slot=2*(b-1)+pass_index
                        journal.add('update_start',slot=slot,rollout=b,pass_index=pass_index)
                        skipped=args.method=='rft' and totals['N_positive']==0
                        before_steps=dict(backend.steps)
                        if skipped:
                            # Do not call an optimizer-facing backend boundary on empty-positive RFT.
                            backend.skip(slot)
                        else:backend.update(records,totals,pass_index)
                        require(backend.steps=={role:n+(0 if skipped else 1) for role,n in before_steps.items()},'Actual optimizer-step accounting mismatch')
                        journal.add('update_commit',slot=slot,rollout=b,skipped=skipped,actual_steps=dict(backend.steps))
                        journal.updates.append(journal.events[-1])
                    publish(root,run,journal,backend,b)
                if end<args.rollouts:
                    journal.add('clean_pause',completed_rollouts=end)
                    return {'status':'PAUSED_AT_COMMITTED_BOUNDARY','checkpoint':str(root/'checkpoints'/f'rollout{end:04d}'),'run_id':run['run_id']}
            journal.add('complete');return {'status':'COMPLETE_NEW_EXECUTION','run_id':run['run_id']}
        except BaseException as exc:
            journal.add('error',error_type=type(exc).__name__,message=str(exc));raise
        finally:
            atomic(root/'counters.json',journal.counters())
            if backend is not None:backend.close()
