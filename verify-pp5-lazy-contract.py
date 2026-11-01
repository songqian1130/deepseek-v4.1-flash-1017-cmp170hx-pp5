"""Run the actual patched lazy tensor class against communication-order contracts."""
import argparse
import ast
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

p=argparse.ArgumentParser();p.add_argument('source');args=p.parse_args()
class Intermediate:
    def __init__(self,tensors):self.tensors=tensors
    def items(self):return self.tensors.items()
    def __getitem__(self,key):return self.tensors[key]

tree=ast.parse(open(args.source,encoding='utf-8').read())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AsyncIntermediateTensors')
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
ns={'IntermediateTensors':Intermediate,'Callable':Callable,'Any':Any,'PP5_DEFERRED_RECEIVE_COUNT':0}
exec(compile(ast.fix_missing_locations(module),'<actual-patched-class>','exec'),ns)
T=ns['AsyncIntermediateTensors'];calls=[]
class Handle:
    def wait(self):calls.append('wait')
def loader():
    calls.append('receive');return {'x':42},[Handle()],[lambda:calls.append('postprocess')]
obj=T({},comm_loader=loader)
assert not calls
calls.append('prepare-attention-metadata')
assert obj['x']==42
assert calls==['prepare-attention-metadata','receive','wait','postprocess']
assert list(obj.items())==[('x',42)]
obj.wait_for_comm();assert len(calls)==4
assert ns['PP5_DEFERRED_RECEIVE_COUNT']==1
calls.clear();stock=T({'x':7},comm_handles=[Handle()],comm_postprocess=[lambda:calls.append('postprocess')])
assert not calls and stock['x']==7 and calls==['wait','postprocess']
assert ns['PP5_DEFERRED_RECEIVE_COUNT']==1
print('PP5_LAZY_RECEIVE_CONTRACT_DONE cases=defer-order,exactly-once,postprocess,stock-path')
