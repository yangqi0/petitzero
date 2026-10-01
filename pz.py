"""PetitZero offline, planning and explicit future model-execution entry point. Author/owner: Yang Qi."""
import argparse,json
from pathlib import Path

def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('verify');v.add_argument('--numbers',nargs=3,type=int,required=True);v.add_argument('--target',type=int,required=True);v.add_argument('--expression',required=True)
    sub.add_parser('fixture')
    a=sub.add_parser('analyze');a.add_argument('--counts',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--figures',action='store_true')
    q=sub.add_parser('plan');q.add_argument('--operation',choices=['train','evaluate','smoke'],required=True);q.add_argument('--method',choices=['grpo','rft','ppo','ppo_zero'],required=True);q.add_argument('--seed',type=int,default=17)
    for name in ['base','sft','data','config','output']:q.add_argument('--'+name,required=True)
    q.add_argument('--actor',help='Explicit final actor adapter for evaluation; defaults to SFT');q.add_argument('--resume');q.add_argument('--responses',type=int,default=32);q.add_argument('--updates',type=int,default=2);q.add_argument('--validate-assets',action='store_true')
    for name in ['train','evaluate','smoke','resume']:
        e=sub.add_parser(name,help='Explicit execution; M12C learned-PPO smoke/resume validated only; requires --execute')
        e.add_argument('--execute',action='store_true')
        e.add_argument('--method',choices=['grpo','rft','ppo','ppo_zero'],required=True)
        e.add_argument('--base',required=True);e.add_argument('--sft');e.add_argument('--actor')
        for field in ['data','config','output']:e.add_argument('--'+field,required=True)
        e.add_argument('--seed',type=int,default=17)
        e.add_argument('--rollouts',type=int,default=1)
        e.add_argument('--samples',type=int,default=32)
        e.add_argument('--max-responses',type=int,required=True)
        e.add_argument('--max-updates',type=int,required=True)
        e.add_argument('--stop-after',type=int)
        e.add_argument('--checkpoint',required=name=='resume')
        e.add_argument('--continue-evaluation',action='store_true')
        if name=='resume':e.add_argument('--operation',choices=['train','smoke'],required=True)
        else:e.set_defaults(operation=name)
    return p

def main(argv=None,backend_factory=None):
    p=parser();args=p.parse_args(argv)
    if args.command in ('train','evaluate','smoke','resume'):
        from petitzero.execution.runner import execute
        try:result=execute(args,backend_factory=backend_factory)
        except (ValueError,FileNotFoundError) as exc:p.error(str(exc))
        print(json.dumps(result,indent=2));return result
    if args.command=='verify':
        from petitzero.countdown import check_expression
        print(json.dumps(check_expression(args.expression,args.numbers,args.target).as_dict()));return
    if args.command=='fixture':
        from petitzero.data_logic import synthetic_fixture
        print(json.dumps(synthetic_fixture()));return
    if args.command=='analyze':
        from petitzero.offline import load_counts,analyze,write_results,figures
        result=analyze(load_counts(args.counts));write_results(result,args.output)
        if args.figures:figures(result,args.output/'figures')
        print(json.dumps({'status':'OFFLINE_SAVED_COUNTS_ONLY','question_policy_rows':4992,'output':str(args.output)}));return
    from petitzero.planning import make_plan,validate_assets
    plan=make_plan(args)
    if args.validate_assets:
        errors=validate_assets(plan)
        if errors:p.error('\n'.join(errors))
        plan['asset_presence_only']='PASS; contents and model execution not validated'
    print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
