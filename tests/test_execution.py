"""CPU integration contracts using ordinary Python fakes, no model or optimizer."""
import contextlib,copy,io,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import pz
from petitzero.execution import protocol as p
from petitzero.execution.durable import Journal,atomic,AmbiguousWork,validate_checkpoint
from petitzero.execution.runner import execute,summarize_evaluation

ROOT=Path(__file__).resolve().parents[1]
class FakeBoundary:
    """Plain Python bookkeeping/serialization, deliberately not a model/optimizer."""
    checkpoint_kind='SYNTHETIC_CPU_TEST_V1'
    calls=[];reward=1
    def __init__(self,context,run,journal,resume,identity):
        self.context=context;self.run=run;self.journal=journal;self.slot=identity['nominal_slot'] if identity else 0
        self.steps=dict(identity['actual_steps']) if identity else {r:0 for r in context['spec']['roles'] if r!='reference' and context['spec']['operation']!='evaluate'}
        self.calls.append(('construct',tuple(context['spec']['roles']),bool(resume)))
        if resume:
            state=p.read(resume/'synthetic_state.json');assert state['steps']==self.steps and state['slot']==self.slot
            self.calls.append(('restore_synthetic_dictionary',self.slot))
    def generate(self,row,request):
        self.calls.append(('fake_response',request['index']))
        return {'request':request,'band':row['band'],'score':{'reward':self.reward,'correct':bool(self.reward)},'synthetic':True,'output_ids':[151645]}
    def prepare(self,records,batch):
        self.calls.append(('prepare',batch));return records,{'N_positive':sum(r['score']['reward'] for r in records)}
    def update(self,records,totals,pass_index):
        self.calls.append(('fake_slot',pass_index));self.slot+=1
        for role in self.steps:self.steps[role]+=1
    def skip(self,slot):
        assert slot==self.slot+1;self.calls.append(('skip_without_optimizer_boundary',slot));self.slot=slot
    def export(self,stage):
        atomic(stage/'synthetic_state.json',{'slot':self.slot,'steps':self.steps,'sentinel_rng':[1,2,3]})
        return {'kind':self.checkpoint_kind,'trainable_roles':list(self.steps),'synthetic_dictionary_only':True}
    def close(self):self.calls.append(('close',))

class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='pz-runtime-test-');self.root=Path(self.tmp.name);self.base=self.root/'base';self.sft=self.root/'sft';self.base.mkdir();self.sft.mkdir();FakeBoundary.calls=[];FakeBoundary.reward=1
        bc={'model_type':'qwen2','hidden_size':1536,'attention_dropout':0,'torch_dtype':'bfloat16'}
        for name,value in [('config.json',bc),('tokenizer_config.json',{'chat_template':'SYNTHETIC_SENTINEL'}),('tokenizer.json',{}),('generation_config.json',{}),('vocab.json',{})]:atomic(self.base/name,value)
        (self.base/'merges.txt').write_text('synthetic');(self.base/'model.safetensors').write_bytes(b'NOT A TENSOR: base fixture')
        ac={'peft_type':'LORA','r':16,'lora_alpha':32,'lora_dropout':0,'bias':'none','target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],'task_type':'CAUSAL_LM'}
        atomic(self.sft/'adapter_config.json',ac);(self.sft/'adapter_model.safetensors').write_bytes(b'NOT A TENSOR: adapter fixture')
        self.patches=[patch.object(p,'BASE_METADATA',{name:p.file_hash(self.base/name) for name in p.BASE_METADATA}),patch.object(p,'SFT_WEIGHT',p.file_hash(self.sft/'adapter_model.safetensors')),patch.object(p,'SFT_CONFIG',p.file_hash(self.sft/'adapter_config.json'))]
        for x in self.patches:x.start()
        self.config=self.root/'config.json';self.cfg={'schema':p.SCHEMA,'base_revision':p.REVISION,'recipe':p.FIXED,'base_weight_hashes':{'model.safetensors':p.file_hash(self.base/'model.safetensors')}};atomic(self.config,self.cfg)
        self.data=self.root/'prompts.jsonl';self.data.write_bytes((ROOT/'data/synthetic_prompts.jsonl').read_bytes())
    def tearDown(self):
        for x in self.patches:x.stop()
        self.tmp.cleanup()
    def args(self,method='ppo_zero',operation='train',rollouts=1,**changes):
        values=dict(execute=True,operation=operation,method=method,base=str(self.base),sft=str(self.sft),actor=None,data=str(self.data),config=str(self.config),seed=17,output=str(self.root/'out'),checkpoint=None,continue_evaluation=False,rollouts=rollouts,samples=1,max_responses=32*rollouts if operation!='evaluate' else 32,max_updates=2*rollouts if operation!='evaluate' else 0,stop_after=None)
        values.update(changes);return SimpleNamespace(**values)
    def test_all_training_methods_reach_execution_and_roles(self):
        for method,roles in [('grpo',('actor','reference')),('rft',('actor',)),('ppo',('actor','reference','critic')),('ppo_zero',('actor','reference'))]:
            with self.subTest(method=method):
                a=self.args(method=method,output=str(self.root/method));x=execute(a,FakeBoundary)
                self.assertEqual(x['status'],'COMPLETE_NEW_EXECUTION');self.assertIn(('construct',roles,False),FakeBoundary.calls)
                ck=p.read(Path(a.output)/'checkpoints/rollout0001/identity.json')
                self.assertEqual(ck['actual_steps'],{'actor':2,'critic':2} if method=='ppo' else {'actor':2})
                self.assertEqual(ck['requests_committed'],32)
                self.assertEqual(p.read(Path(a.output)/'manifest.json')['spec']['namespace'],'NEW_EXECUTION')
    def test_actor_only_evaluation_own_summary(self):
        a=self.args(operation='evaluate',method='ppo',actor=str(self.sft));x=execute(a,FakeBoundary)
        self.assertEqual(x['status'],'COMPLETE_NEW_EXECUTION');self.assertIn(('construct',('actor',),False),FakeBoundary.calls)
        self.assertFalse(any(c[0]=='fake_slot' for c in FakeBoundary.calls))
        summary=p.read(Path(a.output)/'evaluation_summary.json');self.assertEqual(summary['bands']['all']['questions'],16)
        self.assertEqual(summary['bands']['all']['pass_k'],{'1':1.0})
        self.assertFalse((Path(a.output)/'checkpoints').exists())
    def test_real_cli_dispatch_with_fake_boundary(self):
        a=self.args(method='rft',operation='smoke')
        argv=['smoke','--execute','--method','rft','--base',a.base,'--sft',a.sft,'--data',a.data,'--config',a.config,'--output',a.output,'--max-responses','32','--max-updates','2']
        with contextlib.redirect_stdout(io.StringIO()):result=pz.main(argv,backend_factory=FakeBoundary)
        self.assertEqual(result['status'],'COMPLETE_NEW_EXECUTION');self.assertTrue(FakeBoundary.calls)
        self.assertEqual(p.read(Path(a.output)/'manifest.json')['spec']['namespace'],'ENGINEERING')
    def test_empty_positive_rft_nominal_not_actual(self):
        FakeBoundary.reward=0;a=self.args(method='rft');execute(a,FakeBoundary)
        ck=p.read(Path(a.output)/'checkpoints/rollout0001/identity.json')
        self.assertEqual(ck['nominal_slot'],2);self.assertEqual(ck['actual_steps'],{'actor':0})
        self.assertEqual(len([c for c in FakeBoundary.calls if c[0]=='skip_without_optimizer_boundary']),2)
        self.assertFalse(any(c[0]=='fake_slot' for c in FakeBoundary.calls))
    def test_synthetic_dictionary_resume_same_directory(self):
        a=self.args(method='ppo',rollouts=2,stop_after=1);x=execute(a,FakeBoundary);self.assertEqual(x['status'],'PAUSED_AT_COMMITTED_BOUNDARY')
        calls=len([c for c in FakeBoundary.calls if c[0]=='fake_response']);self.assertEqual(calls,32)
        a.checkpoint=x['checkpoint'];a.stop_after=None;y=execute(a,FakeBoundary)
        self.assertEqual(y['status'],'COMPLETE_NEW_EXECUTION');self.assertIn(('restore_synthetic_dictionary',2),FakeBoundary.calls)
        self.assertEqual(len([c for c in FakeBoundary.calls if c[0]=='fake_response']),64)
        identity=p.read(Path(a.output)/'checkpoints/rollout0002/identity.json');self.assertEqual(identity['actual_steps'],{'actor':4,'critic':4})
        a.checkpoint=str(Path(a.output)/'checkpoints/rollout0002');before=len(FakeBoundary.calls)
        self.assertEqual(execute(a,FakeBoundary)['status'],'COMPLETE_NO_NEW_MODEL_WORK');self.assertEqual(len(FakeBoundary.calls),before)
    def test_preflight_failures_before_backend(self):
        cases=[{'base':str(self.root/'missing')},{'max_responses':31},{'max_updates':1},{'rollouts':0},{'samples':33},{'execute':False},{'operation':'smoke','rollouts':3}]
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises((ValueError,FileNotFoundError)):execute(self.args(**changes),FakeBoundary)
                self.assertFalse(FakeBoundary.calls)
        bad=p.read(self.sft/'adapter_config.json');bad['r']=8;atomic(self.sft/'adapter_config.json',bad)
        with self.assertRaises(ValueError):execute(self.args(),FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_hidden_solution_schema_and_output_conflict(self):
        row=json.loads(self.data.read_text().splitlines()[0]);row['oracle_witness']='not allowed';self.data.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError,'Prompt-only'):execute(self.args(),FakeBoundary)
        self.data.write_bytes((ROOT/'data/synthetic_prompts.jsonl').read_bytes());(self.root/'out').mkdir()
        with self.assertRaisesRegex(ValueError,'Output exists'):execute(self.args(),FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_base_sharded_index_and_hash_mismatch(self):
        (self.base/'model.safetensors').unlink()
        for n in ['part-1.safetensors','part-2.safetensors']:(self.base/n).write_text('SYNTHETIC NON-TENSOR '+n)
        atomic(self.base/'model.safetensors.index.json',{'weight_map':{'a':'part-1.safetensors','b':'part-2.safetensors'}})
        self.cfg['base_weight_hashes']={n:p.file_hash(self.base/n) for n in ['part-1.safetensors','part-2.safetensors']};atomic(self.config,self.cfg)
        c=p.preflight(self.args());self.assertIn('model.safetensors.index.json',c['spec']['assets']['base'])
        (self.base/'part-2.safetensors').write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):execute(self.args(),FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def boundary(self):
        a=self.args(rollouts=2,stop_after=1);x=execute(a,FakeBoundary);a.checkpoint=x['checkpoint'];a.stop_after=None;FakeBoundary.calls=[];return a
    def test_resume_rejects_source_data_roles_changes(self):
        a=self.boundary()
        for changes in [{'method':'grpo'},{'seed':29}]:
            b=copy.copy(a)
            for k,v in changes.items():setattr(b,k,v)
            with self.assertRaisesRegex(ValueError,'changed'):execute(b,FakeBoundary)
        with patch.object(p,'source_hashes',return_value={'changed.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'changed'):execute(a,FakeBoundary)
        self.data.write_text(self.data.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):execute(a,FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_wrong_checkpoint_schema_or_payload(self):
        a=self.boundary();folder=Path(a.checkpoint);identity=p.read(folder/'identity.json');original=(folder/'identity.json').read_bytes()
        identity['schema']='OLD_M11';atomic(folder/'identity.json',identity)
        with self.assertRaisesRegex(ValueError,'schema'):execute(a,FakeBoundary)
        (folder/'identity.json').write_bytes(original);(folder/'synthetic_state.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'payload hash'):execute(a,FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_pending_complete_reconciliation_then_training_stops(self):
        a=self.boundary();root=Path(a.output);run=p.read(root/'manifest.json');req=p.read(root/'requests.json');j=Journal(root,run,req);q=req[32]
        j.add('request_start',request=q);record={'request':q,'synthetic':True};j.add('response',request=q,record=record)
        j2=Journal(root,run,req,reconcile=True);self.assertEqual(len(j2.records),33);self.assertTrue(j2.events[-1]['reconciled_without_model'])
        with self.assertRaisesRegex(ValueError,'Unsaved'):execute(a,FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_incomplete_request_and_update_never_replay(self):
        a=self.boundary();root=Path(a.output);run=p.read(root/'manifest.json');req=p.read(root/'requests.json');j=Journal(root,run,req);j.add('request_start',request=req[32])
        with self.assertRaises(AmbiguousWork):execute(a,FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
        # Separate journal, synthetic state only, to test a partially committed update.
        other=self.root/'journal_only';other.mkdir();j=Journal(other,run,req);j.add('update_start',slot=1)
        with self.assertRaisesRegex(AmbiguousWork,'update'):Journal(other,run,req)
    def test_evaluation_complete_record_reconciles_without_generation(self):
        a=self.args(operation='evaluate');execute(a,FakeBoundary);root=Path(a.output)
        events=[json.loads(x) for x in (root/'journal.jsonl').read_text().splitlines()]
        last=max(i for i,e in enumerate(events) if e['event']=='response')
        (root/'journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events[:last+1]))
        a.continue_evaluation=True;FakeBoundary.calls=[]
        self.assertEqual(execute(a,FakeBoundary)['status'],'COMPLETE_NO_NEW_MODEL_WORK')
        self.assertFalse(FakeBoundary.calls)
        j=Journal(root,p.read(root/'manifest.json'),p.read(root/'requests.json'))
        self.assertEqual(len(j.records),32);self.assertTrue(j.events[-1]['reconciled_without_model'])
    def test_synthetic_checkpoint_rejected_by_real_boundary_before_import(self):
        a=self.boundary()
        with self.assertRaisesRegex(ValueError,'boundary kind'):execute(a)
        import sys
        self.assertNotIn('petitzero.execution.native',sys.modules)
    def test_nonempty_update_cannot_report_unchanged_actual_steps(self):
        class NoStep(FakeBoundary):
            def update(self,records,totals,pass_index):pass
        with self.assertRaisesRegex(ValueError,'accounting'):execute(self.args(),NoStep)
    def test_relative_resume_path(self):
        a=self.boundary();previous=Path.cwd()
        try:
            os.chdir(self.root);a.output='out';a.checkpoint='out/checkpoints/rollout0001'
            self.assertEqual(execute(a,FakeBoundary)['status'],'COMPLETE_NEW_EXECUTION')
        finally:os.chdir(previous)
    def test_failed_response_attempt_is_counted_and_never_replayed(self):
        class Failed(FakeBoundary):
            def generate(self,row,request):raise RuntimeError('synthetic boundary failure')
        a=self.args()
        with self.assertRaisesRegex(RuntimeError,'synthetic boundary'):execute(a,Failed)
        counts=p.read(Path(a.output)/'counters.json');self.assertEqual(counts['generation_attempts'],1);self.assertEqual(counts['committed_responses'],0);self.assertEqual(counts['errors'],1)
        a.checkpoint=str(Path(a.output)/'checkpoints/rollout0000');FakeBoundary.calls=[]
        with self.assertRaises(AmbiguousWork):execute(a,FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_resume_nonadvancing_stop_fails_before_boundary(self):
        a=self.boundary();a.stop_after=1
        with self.assertRaisesRegex(ValueError,'advance'):execute(a,FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_changed_base_metadata_rejected_before_boundary(self):
        value=p.read(self.base/'config.json');value['num_hidden_layers']=1;atomic(self.base/'config.json',value)
        with self.assertRaisesRegex(ValueError,'metadata hash'):execute(self.args(),FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_output_inside_assets_rejected(self):
        with self.assertRaisesRegex(ValueError,'disjoint'):execute(self.args(output=str(self.base/'run')),FakeBoundary)
        self.assertFalse(FakeBoundary.calls)
    def test_relative_paths_and_help_no_native_import(self):
        previous=Path.cwd()
        try:
            os.chdir(self.root);a=self.args(base='base',sft='sft',data='prompts.jsonl',config='config.json',output='relative_out')
            result=execute(a,FakeBoundary);self.assertEqual(result['status'],'COMPLETE_NEW_EXECUTION')
        finally:os.chdir(previous)
        import sys
        self.assertNotIn('petitzero.execution.native',sys.modules)
        for command in ['train','evaluate','smoke','resume','plan']:
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as e:pz.parser().parse_args([command,'--help'])
            self.assertEqual(e.exception.code,0)

if __name__=='__main__':unittest.main()
