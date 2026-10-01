"""Nested evidence ownership and native-style enrichment, with no real backend."""
import copy,json,unittest
from pathlib import Path
import test_execution as fixtures
from petitzero.execution import protocol as p
from petitzero.execution.durable import Journal,atomic
from petitzero.execution.runner import execute

class MutatingBoundary(fixtures.FakeBoundary):
    def prepare(self,records,batch):
        for r in records:
            r.update(old_action_logps=[-1.0],reference_action_logps=[-1.0],old_values=[0.0],targets={'returns':[1.0],'advantages':[0.5]})
        atomic(self.context['output']/f'enriched_{batch}.json',records)
        return super().prepare(records,batch)
    def update(self,records,totals,pass_index):
        assert all(r['targets']['returns']==[1.0] and r['old_values']==[0.0] for r in records)
        return super().update(records,totals,pass_index)

class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ExecutionTests();self.fixture.setUp();self.root=self.fixture.root
    def tearDown(self):self.fixture.tearDown()
    def simple(self):
        root=self.root/'journal';root.mkdir();run={'run_id':'nested-synthetic'};requests=[{'index':0,'nested':{'ids':[1,2]}}]
        return root,run,requests,Journal(root,run,requests)
    def assert_disk(self,j):
        disk=[json.loads(x) for x in j.path.read_text().splitlines()]
        self.assertEqual(j.events,disk)
        return disk
    def test_response_caller_nested_and_top_level_mutation(self):
        root,run,requests,j=self.simple();q=copy.deepcopy(requests[0]);record={'request':q,'nested':{'values':[1,2]},'tuple':(3,4)}
        j.add('request_start',request=q);j.response(q,record);before=j.path.read_bytes();original=copy.deepcopy(j.events);commit=j.events[-1]['response_hash']
        record['new']='training';record['nested']['values'].append(9);q['nested']['ids'].append(8)
        self.assertEqual(before,j.path.read_bytes());self.assertEqual(j.events,original);self.assertEqual(commit,p.digest(j.events[1]['record']));self.assertEqual(j.events[1]['record']['tuple'],[3,4])
        j.records[0]['nested']['values'].append(99)
        self.assertEqual(j.events,original);j.add('audit',details={'ok':[True]});self.assert_disk(j)
        reopened=Journal(root,run,requests);self.assertEqual(reopened.records[0]['nested']['values'],[1,2]);self.assert_disk(reopened)
    def test_add_detaches_and_json_normalizes(self):
        root,run,requests,j=self.simple();value={'a':[(1,2)],'b':{1:'one'}};j.add('audit',payload=value);value['a'][0]=(9,9);value['b'][1]='changed'
        self.assertEqual(j.events[0]['payload'],{'a':[[1,2]],'b':{'1':'one'}});j.add('audit');self.assert_disk(Journal(root,run,requests))
    def test_reopened_and_reconciled_records_do_not_alias_events(self):
        root,run,requests,j=self.simple();q=requests[0];record={'request':q,'nested':{'x':[1]}}
        j.add('request_start',request=q);j.add('response',request=q,record=record)
        r=Journal(root,run,requests,reconcile=True);committed=copy.deepcopy(r.events);r.records[0]['nested']['x'].append(2)
        self.assertEqual(r.events,committed);r.add('audit');r2=Journal(root,run,requests);r2.records[0]['nested']['x'].append(3);self.assert_disk(r2)
    def test_mutating_prepare_pause_fresh_disk_resume_and_targets(self):
        args=self.fixture.args(method='ppo',operation='smoke',seed=9001,rollouts=2,stop_after=1)
        paused=execute(args,MutatingBoundary);root=Path(args.output);identity=p.read(Path(paused['checkpoint'])/'identity.json')
        disk=[json.loads(x) for x in (root/'journal.jsonl').read_text().splitlines()]
        self.assertEqual(p.digest(disk[:identity['journal_prefix_length']]),identity['journal_prefix_sha256'])
        self.assertEqual(identity['requests_committed'],32);self.assertEqual(identity['nominal_slot'],2);self.assertEqual(identity['actual_steps'],{'actor':2,'critic':2});self.assertEqual(identity['next_request']['index'],32)
        args.checkpoint=paused['checkpoint'];args.stop_after=None;result=execute(args,MutatingBoundary)
        self.assertEqual(result['status'],'COMPLETE_NEW_EXECUTION');self.assertEqual(sum(c[0]=='fake_response' for c in fixtures.FakeBoundary.calls),64);self.assertEqual(sum(c[0]=='construct' for c in fixtures.FakeBoundary.calls),2)
        j=Journal(root,p.read(root/'manifest.json'),p.read(root/'requests.json'));self.assertEqual(len(j.records),64);self.assertEqual([e['slot'] for e in j.updates],[1,2,3,4]);self.assertEqual(j.updates[-1]['actual_steps'],{'actor':4,'critic':4})
        self.assertEqual(len({p.digest(r['request']) for r in j.records}),64)
        for r in j.records:self.assertNotIn('old_values',r);self.assertNotIn('targets',r)
        enriched=[r for f in root.glob('enriched_*.json') for r in p.read(f)];self.assertEqual(len(enriched),64);self.assertTrue(all(r['targets']['advantages']==[0.5] for r in enriched))
        final=p.read(root/'checkpoints/rollout0002/identity.json');self.assertEqual(p.digest(j.events[:final['journal_prefix_length']]),final['journal_prefix_sha256'])
    def test_modified_journal_rejected_before_backend(self):
        args=self.fixture.boundary();root=Path(args.output);events=[json.loads(x) for x in (root/'journal.jsonl').read_text().splitlines()]
        next(e for e in events if e['event']=='response')['record']['score']['reward']=77
        (root/'journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        with self.assertRaisesRegex(ValueError,'Journal chain'):execute(args,fixtures.FakeBoundary)
        self.assertFalse(fixtures.FakeBoundary.calls)

if __name__=='__main__':unittest.main()
