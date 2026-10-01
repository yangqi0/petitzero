"""Finite synthetic CPU tests. No optimizer, model, tokenizer, GPU or network."""
import copy,csv,json,math,unittest
from pathlib import Path
import torch
from petitzero.countdown import check_expression
from petitzero.grpo import action_labels,selected_action_logps,group_advantages,grpo_loss
from petitzero.ppo import sampled_rewards,gae_targets,whiten_advantages,policy_loss,value_loss
from petitzero.ppo_runtime import compact_batch,stored
from petitzero.rft_runtime import run_optimizer_slot,select_responses
from petitzero.v2.methods import prepare_targets,objective,roles_for
from petitzero.offline import load_counts,analyze
from petitzero.data_logic import partition_groups,target_for,synthetic_fixture
ROOT=Path(__file__).resolve().parents[1]

def records():
    return [dict(prompt_ids=[8,9],output_ids=out,old_action_logps=[-1.]*len(out),reference_action_logps=[-1.]*len(out),old_values=[.25]*len(out),terminal_kind='NATIVE_EOS_TERMINAL',score={'reward':reward}) for out,reward in [([4,151645],1),([151643],0),([4,151645],1),([151645],0)]]

class MathTests(unittest.TestCase):
    def test_strict_fraction_and_rejections(self):
        self.assertTrue(check_expression(' 2*(3+4) ',[2,3,4],14).correct)
        self.assertTrue(check_expression('3/(2/2)',[3,2,2],3).correct)
        for s in ['Answer: 2*(3+4)','2*(3+4)=14','```2*(3+4)```','2**3+4','2.0*(3+4)','+2*(3+4)','2*(3+3)','2*(3+4)\n14']:
            self.assertFalse(check_expression(s,[2,3,4],14).correct,s)
        self.assertEqual(check_expression('2/(3-3)',[2,3,3],1).reason,'division_by_zero')
    def test_causal_shift_eos_not_pad(self):
        rr=records();batch,pos,mask=compact_batch(rr)
        self.assertEqual(pos[0].tolist(),[1,2]);self.assertEqual(mask.tolist(),[[True,True],[True,False],[True,True],[True,False]])
        self.assertEqual(batch['labels'][1].tolist(),[-100,-100,151643,-100])
        # Tiny vocabulary synthetic logits only; no network forward.
        logits=torch.zeros((1,4,6),requires_grad=True);labels=torch.tensor([[-100,-100,4,5]])
        lp,m=selected_action_logps(logits,labels)
        self.assertEqual(m.tolist(),[[False,True,True]])
        self.assertTrue(torch.allclose(lp[m],torch.full((2,),-math.log(6))))
        lp.sum().backward();self.assertEqual(float(logits.grad[0,3].abs().sum()),0.)
    def test_gae_credit_whitening(self):
        m=torch.tensor([[True,True],[True,False]])
        old=torch.tensor([[-1.,-1.],[-1.,float('nan')]])
        ref=torch.tensor([[-2.,-2.],[-1.,0.]])
        shaped=sampled_rewards(old,ref,torch.tensor([1.,0.]),m)
        self.assertTrue(torch.allclose(shaped,torch.tensor([[-.02,.98],[0.,0.]])))
        raw,ret=gae_targets(shaped,torch.zeros_like(shaped),m,torch.tensor([True,True]),torch.zeros(2))
        self.assertTrue(torch.allclose(raw,torch.tensor([[.911,.98],[0.,0.]]),atol=1e-6))
        white,info=whiten_advantages(raw,m);self.assertEqual(info['actions'],3)
        self.assertAlmostEqual(float(white[m].mean()),0.,places=6)
        self.assertAlmostEqual(float(white[m].var(unbiased=False)),1.,places=5)
        self.assertTrue(torch.equal(raw,ret));self.assertFalse(raw.requires_grad)
    def test_zero_targets_equal_zero_learned_boundary(self):
        zero=records();learned=records()
        for r in learned:r['old_values']=[0.]*len(r['output_ids'])
        z=prepare_targets('ppo_zero',zero);p=prepare_targets('ppo',learned)
        self.assertEqual(roles_for('ppo_zero'),('actor','reference'));self.assertIsNone(z['critic_prefit_token_weighted'])
        self.assertEqual(z['D'],6)
        for a,b in zip(zero,learned):
            for k in ('advantages','raw_advantages','returns','shaped_rewards'):self.assertEqual(a[k],b[k])
            self.assertTrue(all(v==0 for v in a['old_values']))
        _,_,mask=compact_batch(zero);current=stored(zero,'old_action_logps',mask).requires_grad_()
        a,_=objective('ppo_zero','actor',current,mask,zero,z);b,_=objective('ppo','actor',current,mask,learned,p)
        self.assertEqual(float(a.detach()),float(b.detach()));a.backward();self.assertTrue(torch.isfinite(current.grad).all())
    def test_full_denominator_microbatch_and_frozen_targets(self):
        rr=records();total=prepare_targets('ppo_zero',rr);snapshot=copy.deepcopy(rr)
        _,_,m=compact_batch(rr);cur=stored(rr,'old_action_logps',m)+.1
        full,_=objective('ppo_zero','actor',cur,m,rr,total)
        parts=[]
        for i,r in enumerate(rr):
            _,_,mm=compact_batch([r]);parts.append(objective('ppo_zero','actor',cur[i:i+1,:mm.shape[1]],mm,[r],total)[0])
        self.assertTrue(torch.allclose(sum(parts),full,atol=1e-7,rtol=1e-6));self.assertEqual(rr,snapshot)
        with self.assertRaises(ValueError):policy_loss(cur,cur.detach(),torch.ones_like(cur),m,total_actions=1)
    def test_grpo_constant_and_denominator(self):
        adv=group_advantages(torch.tensor([[0.,0.,0.,0.],[1.,1.,1.,1.],[0.,1.,0.,1.]]))
        self.assertTrue(torch.equal(adv[:2],torch.zeros_like(adv[:2])))
        self.assertAlmostEqual(float(adv[2,1]),.5/(.5+1e-4),places=6)
        cur=torch.zeros((2,2));mask=torch.tensor([[True,True],[True,False]])
        loss,_=grpo_loss(cur,cur,cur,torch.tensor([1.,-1.]),mask,total_completions=32)
        self.assertEqual(float(loss),0.)
    def test_rft_duplicates_and_true_empty_skip(self):
        rr=records();rewards,indices=select_responses(rr);self.assertEqual(indices,[0,2])
        total=prepare_targets('rft',rr);_,_,mask=compact_batch(rr);cur=torch.full(mask.shape,-2.,requires_grad=True)
        loss,_=objective('rft','actor',cur,mask,rr,total);self.assertAlmostEqual(float(loss.detach()),2.)
        # Sentinel is not an optimizer; every attribute access would fail.
        class Forbidden:
            def __getattribute__(self,name):raise AssertionError('Empty skip touched optimizer')
        result=run_optimizer_slot(Forbidden(),[],torch.zeros(4),1,lambda _:self.fail('backward'))
        self.assertTrue(result['skipped'])
    def test_value_clip(self):
        mask=torch.tensor([[True]]);loss,_=value_loss(torch.tensor([[1.]]),torch.tensor([[0.]]),torch.tensor([[1.]]),mask,total_actions=1)
        self.assertAlmostEqual(float(loss),.5*.8**2,places=6)
    def test_split_and_synthetic_band(self):
        # Explicit synthetic exclusion list, not a historical data rebuild.
        from itertools import combinations_with_replacement
        excluded=list(combinations_with_replacement(range(1,21),3))[:64]
        a=partition_groups(excluded);self.assertEqual([len(a[k]) for k in ('train','dev','final','unused')],[1024,64,256,132])
        self.assertEqual(len(set(sum((a[k] for k in a),[]))),1476)
        self.assertEqual(a,partition_groups(reversed(excluded)))
        self.assertEqual(target_for('synthetic',0,(2,3,4))['band'],'add_sub_solvable')
        self.assertTrue(synthetic_fixture()['correct'])

class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=analyze(load_counts(ROOT/'data/endpoint_counts.csv'))
    def test_m10_all_seed_metrics_histograms(self):
        lookup={(r['policy'],r['population'],r['band'],r['metric']):r['percent'] for r in self.data['seed_points']}
        hist={(r['policy'],r['population'],r['band'],r['c']):r['questions'] for r in self.data['histograms']}
        for r in csv.DictReader((ROOT/'data/source_tables/m10_policy.csv').read_text().splitlines()):
            for metric in ('greedy','pass@1','pass@2','pass@4','pass@8','pass@16','pass@32'):
                self.assertAlmostEqual(lookup[r['policy'],r['suite'],r['band'],metric],100*float(r[metric]),delta=1e-10)
            for c in range(33):self.assertEqual(hist[r['policy'],r['suite'],r['band'],c],int(r[f'c{c}']))
    def test_m11_seed_and_paired_source_precision(self):
        lookup={(r['policy'],r['population'],r['band'],r['metric']):r['percent'] for r in self.data['seed_points']}
        for r in csv.DictReader((ROOT/'data/source_tables/m11_seed_points.csv').read_text().splitlines()):self.assertAlmostEqual(lookup[r['policy'],r['suite'],r['band'],r['metric']],float(r['percent']),delta=1e-10)
        paired={(r['population'],r['band'],r['metric']):r for r in self.data['primary_zero_minus_learned']}
        for r in csv.DictReader((ROOT/'data/source_tables/m11_paired.csv').read_text().splitlines()):
            actual=paired[r['suite'],r['band'],r['metric']]
            for key in ('mean_difference_pp','sample_SD_difference_pp'):self.assertAlmostEqual(actual[key],float(r[key]),delta=1e-10)
            for i,s in enumerate((17,29,43)):self.assertAlmostEqual(actual['seed_differences'][i],float(r[f'seed{s}_difference_pp']),delta=1e-10)
        self.assertLess(paired['CONFIRM128_RESERVED','all','pass@32']['seed_differences'][0],0)
        for m in ('pass@1','pass@32'):self.assertLess(paired['HISTORICAL_FINAL256_OBSERVED','all',m]['seed_differences'][0],0)
    def test_secondary_both_branches(self):
        lookup={(r['population'],r['band'],r['metric']):r for r in self.data['secondary_matched17_29']}
        for r in csv.DictReader((ROOT/'data/source_tables/m11_secondary.csv').read_text().splitlines()):
            actual=lookup[r['suite'],r['band'],r['metric']]
            for key in ('learned_mean_percent','learned_SD_pp','zero_mean_percent','zero_SD_pp','paired_mean_difference_pp','paired_SD_difference_pp'):self.assertAlmostEqual(actual[key],float(r[key]),delta=1e-10)
if __name__=='__main__':unittest.main()
