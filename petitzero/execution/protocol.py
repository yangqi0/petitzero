"""Local asset/schema validation and NEW_EXECUTION schedules. Standard library only."""
import hashlib,json,os,uuid
from pathlib import Path
from datetime import datetime,timezone
from petitzero.countdown import validate_problem,make_messages

SCHEMA='PZ_PORTABLE_RUN_V1'
CHECKPOINT='PZ_PORTABLE_CHECKPOINT_V1'
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
SFT_WEIGHT='bf7e362147437e90c97f0e4675cbb724f89323fc02b7722a501fd975ec1049b7'
SFT_CONFIG='4b299f5a0985c94eb4322a69a4f49089ad641c4bcb8b946915de02f43bba1e9a'
BASE_WEIGHT='dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
BASE_METADATA={'config.json': '98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670', 'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583', 'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539', 'generation_config.json': 'e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6', 'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'}
METHODS=('grpo','rft','ppo','ppo_zero')
BANDS=('add_sub_solvable','requires_mul_or_div')
FIXED={'profile':'F-M','max_new_tokens':128,'temperature':1.,'top_p':1.,'top_k':0,'rollout_questions':8,'samples_per_question':4,'microbatch':2,'passes':2,'actor_lr':1e-5,'critic_lr':1e-4,'warmup_slots':8,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,'max_grad_norm':1.,'alignment_atol':2e-4,'alignment_rtol':1e-5}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(condition,message):
    if not condition:raise ValueError(message)
def roles(method,evaluation=False):
    require(method in METHODS,'Unknown method')
    return ['actor'] if evaluation or method=='rft' else ['actor','reference']+(['critic'] if method=='ppo' else [])
def safe_file(root,name):
    rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe asset/payload filename')
    p=Path(root)/rel;require(p.is_file() and not p.is_symlink(),'Missing regular local file: '+str(p));return p

def config(path):
    c=read(path);require(set(c)=={'schema','base_revision','recipe','base_weight_hashes'},'Unexpected/missing config keys')
    require(c['schema']==SCHEMA and c['base_revision']==REVISION,'Unsupported schema/base revision')
    require(c['recipe']==FIXED,'Scientific recipe differs from recorded F-M recipe')
    require(isinstance(c['base_weight_hashes'],dict) and c['base_weight_hashes'],'Supply expected local base weight hashes')
    require(all(isinstance(v,str) and len(v)==64 and all(ch in '0123456789abcdef' for ch in v) for v in c['base_weight_hashes'].values()),'Invalid expected weight hash')
    return c

def adapter(path,shared=False):
    root=Path(path);c=safe_file(root,'adapter_config.json');w=safe_file(root,'adapter_model.safetensors');cfg=read(c)
    expected_modules={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    require(cfg.get('peft_type')=='LORA' and cfg.get('r')==16 and cfg.get('lora_alpha')==32 and cfg.get('lora_dropout')==0 and cfg.get('bias')=='none' and set(cfg.get('target_modules',[]))==expected_modules,'Adapter must use the recorded LoRA structure')
    require(cfg.get('task_type')=='CAUSAL_LM' and not cfg.get('use_dora',False) and not cfg.get('use_rslora',False),'Unsupported adapter mode')
    hashes={p.name:file_hash(p) for p in (c,w)}
    if shared:require(hashes=={'adapter_config.json':SFT_CONFIG,'adapter_model.safetensors':SFT_WEIGHT},'Shared M1A adapter/config hash mismatch; base-only or another SFT is not allowed')
    return hashes

def assets(base,sft,actor,cfg,evaluation):
    root=Path(base);names=['config.json','tokenizer_config.json','tokenizer.json','generation_config.json','vocab.json','merges.txt']
    paths=[safe_file(root,n) for n in names]
    for path in paths:require(file_hash(path)==BASE_METADATA[path.name],'Original base/tokenizer metadata hash mismatch: '+path.name)
    bc=read(paths[0]);tc=read(paths[1])
    require(bc.get('model_type')=='qwen2' and bc.get('hidden_size')==1536 and bc.get('attention_dropout',0)==0,'Wrong base config/profile')
    require(bc.get('torch_dtype')=='bfloat16','Base config must declare stored BF16 values')
    require('chat_template' in tc,'Original tokenizer chat template missing')
    index=root/'model.safetensors.index.json'
    if index.exists():
        ix=read(index);require(isinstance(ix.get('weight_map'),dict) and ix['weight_map'],'Malformed safetensors shard index')
        weight_names=sorted(set(ix['weight_map'].values()));paths.append(safe_file(root,index.name))
    else:weight_names=['model.safetensors']
    require(set(weight_names)==set(cfg['base_weight_hashes']),'Expected base weight hashes must cover the complete local layout')
    for n in weight_names:
        p=safe_file(root,n);require(p.suffix=='.safetensors','Only safetensors base weights supported');require(file_hash(p)==cfg['base_weight_hashes'][n],'Base weight hash mismatch: '+n);paths.append(p)
    result={'base':{str(p.relative_to(root)):file_hash(p) for p in paths}}
    if evaluation:result['actor']=adapter(actor or sft,shared=not bool(actor))
    else:result['sft']=adapter(sft,shared=True)
    return result

def prompts(path,operation):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    require(rows and len(rows)<=100000,'Empty/oversized prompt-only dataset')
    ids=set();groups=set()
    allowed={'id','numbers','target','band','sft_exposed','namespace'}
    for r in rows:
        require(isinstance(r,dict) and set(r)==allowed,'Prompt-only schema requires exactly id/numbers/target/band/sft_exposed/namespace; no witnesses/answers/messages')
        require(isinstance(r['id'],str) and r['id'] and len(r['id'])<=128 and r['id'] not in ids,'Invalid/duplicate question ID');ids.add(r['id'])
        validate_problem(r['numbers'],r['target'])
        require(r['band'] in BANDS and type(r['sft_exposed']) is bool,'Invalid band/exposure metadata')
        require(r['namespace'] in ('NEW_EXECUTION','SYNTHETIC_ENGINEERING'),'No historical benchmark namespace')
        if operation=='smoke':require(r['namespace']=='SYNTHETIC_ENGINEERING','Smoke requires synthetic engineering-only input')
        group=tuple(sorted(r['numbers']));require(group not in groups,'Duplicate numerical multiset in prompt input');groups.add(group)
        r['messages']=make_messages(r['numbers'],r['target']);r['number_group']=digest(sorted(r['numbers']))
    return rows

def seed32(value):return int(digest(value)[:16],16)%(2**31-1)
def requests(rows,seed,operation,rollouts,samples):
    out=[]
    if operation=='evaluate':
        for r in sorted(rows,key=lambda r:r['id']):
            for mode in ['greedy','sample']:
                for i in range(1 if mode=='greedy' else samples):
                    out.append({'index':len(out),'id':r['id'],'mode':mode,'sample_index':i,'seed':seed32(['PZ-NEW','evaluate',seed,r['id'],mode,i]),'rollout':0})
    else:
        pools={}
        for b in BANDS:
            for seen in (False,True):
                pool=[r for r in rows if r['band']==b and r['sft_exposed'] is seen]
                require(len(pool)>=2*rollouts,'Each band/exposure stratum needs two distinct prompts per rollout')
                pools[b,seen]=sorted(pool,key=lambda r:(digest(['PZ-V2','order',seed,r['id']]),r['id']))
        for b in range(rollouts):
            group=[r for pool in pools.values() for r in pool[2*b:2*b+2]]
            group.sort(key=lambda r:(digest(['PZ-V2','within-batch',seed,b+1,r['id']]),r['id']))
            for r in group:
                for i in range(4):out.append({'index':len(out),'id':r['id'],'mode':'sample','sample_index':i,'seed':seed32(['PZ-V2','train-sample',seed,r['id'],i]),'rollout':b+1})
    return out

def source_hashes():
    root=Path(__file__).resolve().parents[1]
    files=list(root.rglob('*.py'))+[root.parent/'pz.py']
    return {str(p.relative_to(root.parent)):file_hash(p) for p in sorted(files)}

def preflight(args):
    """No model/torch import. Complete local validation before backend creation."""
    require(args.execute,'Execution requires explicit --execute; use plan for intentions')
    operation=args.operation;require(operation in ('train','smoke','evaluate'),'Unsupported operation')
    require(type(args.seed)is int and 0<=args.seed<2**31-1,'Seed out of range')
    require(type(args.rollouts)is int and 1<=args.rollouts<=128,'rollouts must be 1..128')
    require(type(args.samples)is int and 0<=args.samples<=32,'samples must be 0..32')
    if operation=='smoke':require(args.rollouts<=2,'Smoke capped at two rollouts of one selected method')
    require(args.method in METHODS,'Select exactly one method')
    require(not getattr(args,'actor',None) or operation=='evaluate','--actor is evaluation-only; training always starts from the verified shared M1A')
    cfg=config(args.config);rows=prompts(args.data,operation)
    req=requests(rows,args.seed,operation,args.rollouts,args.samples)
    require(args.max_responses==len(req),'Explicit response cap must equal the complete requested workload')
    wanted_updates=0 if operation=='evaluate' else 2*args.rollouts
    require(args.max_updates==wanted_updates,'Explicit actor nominal-slot cap mismatch')
    require(args.stop_after is None or operation!='evaluate' and 1<=args.stop_after<=args.rollouts,'stop-after must be a completed rollout within this training budget')
    out=Path(args.output).resolve();checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else None
    continuing_eval=getattr(args,'continue_evaluation',False)
    require(not continuing_eval or operation=='evaluate' and checkpoint is None and out.is_dir(),'continue-evaluation requires its original existing evaluation directory')
    require(checkpoint is not None or continuing_eval or not out.exists(),'Output exists: choose a dedicated new directory, or use resume with its committed checkpoint')
    require(checkpoint is None or operation!='evaluate','Evaluation continuation uses --continue-evaluation, not training checkpoints')
    require(checkpoint is None or out.is_dir(),'Resume requires the original existing run directory')
    ancestor=out if out.exists() else out.parent
    while not ancestor.exists():ancestor=ancestor.parent
    require(ancestor.is_dir() and os.access(ancestor,os.W_OK),'Output parent is not writable')
    paths={k:str(Path(getattr(args,k)).resolve()) if getattr(args,k,None) else None for k in ('base','sft','actor','data','config')}
    require(paths['sft'] is not None or operation=='evaluate' and paths['actor'] is not None,'Explicit shared SFT required for training, actor/SFT for evaluation')
    for key in ('base','sft','actor'):
        if paths[key]:
            asset_root=Path(paths[key]);require(out!=asset_root and asset_root not in out.parents and out not in asset_root.parents,'Output must be dedicated and disjoint from model/adapter directories')
    identities=assets(paths['base'],paths['sft'],paths['actor'],cfg,operation=='evaluate')
    spec={'schema':SCHEMA,'namespace':'ENGINEERING' if operation=='smoke' else 'NEW_EXECUTION','operation':operation,'method':args.method,'roles':roles(args.method,operation=='evaluate'),'seed':args.seed,'config_sha256':file_hash(args.config),'recipe':cfg['recipe'],'base_revision':REVISION,'assets':identities,'data_sha256':file_hash(args.data),'requests_sha256':digest(req),'sources':source_hashes(),'caps':{'responses':len(req),'actor_slots':wanted_updates,'critic_steps':wanted_updates if args.method=='ppo' and operation!='evaluate' else 0},'rollouts':0 if operation=='evaluate' else args.rollouts,'samples':args.samples if operation=='evaluate' else 4}
    return {'spec':spec,'spec_hash':digest(spec),'paths':paths,'rows':rows,'requests':req,'output':out,'checkpoint':checkpoint,'stop_after':args.stop_after}

def manifest(context):
    return {'schema':SCHEMA,'run_id':str(uuid.uuid4()),'created_utc':datetime.now(timezone.utc).isoformat(),'spec':context['spec'],'spec_hash':context['spec_hash'],'input_paths':context['paths'],'historical_replay':False}
