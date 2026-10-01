"""Run documented offline commands with fail-closed model/optimizer/network guards."""
import sys,os,runpy,importlib.abc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1')
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('transformers','peft','huggingface_hub','tokenizers'):
            raise RuntimeError('Forbidden model/tokenizer dependency import: '+fullname)
sys.meta_path.insert(0,NoModels())
def denied(*args,**kwargs):raise RuntimeError('Forbidden optimizer/model forward/GPU operation')
# CPU torch import itself is allowed. Do not query CUDA availability.
import torch
torch.optim.Optimizer.__init__=denied
torch.nn.Module.__init__=denied
torch.nn.Module._call_impl=denied
torch.cuda._lazy_init=denied
torch.cuda.init=denied
torch.cuda.is_available=denied
# Private historical runs, caches and credentials are unavailable to tested code.
def audit(event,args):
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo'):raise RuntimeError('Network forbidden')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        parts=p.parts
        if ('runs' in parts and ROOT not in p.parents and p!=ROOT) or '.cache' in parts or '.ssh' in parts or '.aws' in parts:
            raise RuntimeError('Private file access forbidden')
sys.addaudithook(audit)
if sys.argv[1:] == ['imports']:
    import importlib,pkgutil,petitzero
    modules=[m.name for m in pkgutil.walk_packages(petitzero.__path__,prefix='petitzero.')]
    for name in modules:importlib.import_module(name)
    assert not any(name in sys.modules for name in ('transformers','peft','tokenizers','huggingface_hub'))
    print('PASS guarded imports:',len(modules))
    raise SystemExit(0)
if sys.argv[1:] == ['tests']:
    import unittest
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
sys.argv=[str(ROOT/'pz.py')]+sys.argv[1:]
runpy.run_path(str(ROOT/'pz.py'),run_name='__main__')
