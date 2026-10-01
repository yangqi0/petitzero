"""Fresh execution intentions only. No historical lock authorization or GPU runner."""
from pathlib import Path
import json,hashlib,uuid
from datetime import datetime,timezone
METHODS=('grpo','rft','ppo','ppo_zero')
BLOCKER='M12C_LEARNED_PPO_SMOKE_RESUME_VALIDATED_ONLY: GRPO/RFT/ppo_zero and standalone evaluation native paths, other devices/shapes and full replay remain unvalidated; uninterrupted equivalence is not established. Plan only; no execution.'

def make_plan(args):
    config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text())
    if config.get('schema')!='PZ_NEW_EXECUTION_INTENT_V1':raise ValueError('Unsupported new-execution config')
    paths={k:str(Path(getattr(args,k)).resolve()) if getattr(args,k,None) else None for k in ('base','sft','data','output','resume','actor')}
    if args.operation=='evaluate' and paths['actor'] is None:paths['actor']=paths['sft']
    if args.operation=='smoke' and (args.responses!=32 or args.updates!=2):raise ValueError('Proposed smoke is exactly one rollout: 32 responses, two nominal actor slots')
    if args.operation=='smoke' and args.resume:raise ValueError('Initial smoke plan cannot also request resume')
    roles=['actor'] if args.operation=='evaluate' or args.method=='rft' else ['actor','reference']+(['critic'] if args.method=='ppo' else [])
    return {'schema':config['schema'],'intent_id':str(uuid.uuid4()),'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PLAN_ONLY_NO_EXECUTION','operation':args.operation,'method':args.method,'seed':args.seed,'paths':paths,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'roles':roles,'proposed_smoke_budget':{'responses':32,'actor_nominal_slots':2,'critic_steps':2 if args.method=='ppo' else 0,'evaluation_responses':0,'engineering_only':True} if args.operation=='smoke' else None,'runtime_status':BLOCKER,'historical_identity_reused':False}

def validate_assets(plan):
    errors=[]
    for key,files in [('base',('config.json','model.safetensors','tokenizer_config.json','tokenizer.json')),('sft',('adapter_config.json','adapter_model.safetensors'))]:
        root=Path(plan['paths'][key])
        for name in files:
            if not (root/name).is_file():errors.append(f'Missing {key} file: {root/name}; supply an existing local '+('Qwen revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 snapshot' if key=='base' else 'shared M1A adapter (base-only initialization is not equivalent)'))
    if plan['operation']=='evaluate':
        for name in ('adapter_config.json','adapter_model.safetensors'):
            if not (Path(plan['paths']['actor'])/name).is_file():errors.append('Missing evaluated actor adapter file: '+name+'; supply --actor for a final policy, or use the explicit SFT adapter')
    if not Path(plan['paths']['data']).is_file():errors.append('Missing explicit data file; supply a local prompt-only dataset for future execution')
    if Path(plan['paths']['output']).exists():errors.append('Output already exists; choose a dedicated new output directory')
    if plan['paths']['resume'] and not Path(plan['paths']['resume']).is_dir():errors.append('Resume checkpoint directory missing; an adapter alone is not a continuation checkpoint')
    return errors
