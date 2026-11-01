"""Device mapping correctness, captured dynamic boundaries and host launch cost."""
import argparse
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata

p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--out',required=True);args=p.parse_args()
stop=threading.Event();began=time.monotonic()
def heartbeat():
    while not stop.wait(10):print(f'[HB {time.monotonic()-began:.0f}s] phase=verify-fast-token-mapping',flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
try:
    spec=importlib.util.spec_from_file_location('fast_mapping',args.module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    original=CommonAttentionMetadata.token_to_req_indices
    cases=[([0,1],1),([0,1,2,3,4],4),([0,3,3,7],9),([0,0,0,2,2],2),([0,0,0],5),([0,55,56,59,60],64),([0,512,1024,2048],2048),([0,8,8,8,8],4)]
    for starts,n in cases:
        cm=SimpleNamespace(query_start_loc=torch.tensor(starts,device='cuda',dtype=torch.int32),query_start_loc_cpu=torch.tensor(starts,dtype=torch.int32),num_actual_tokens=n,_token_to_req_indices_cache=None)
        size=max(starts[-1],n);a=torch.full((size+7,),-9,device='cuda',dtype=torch.int32);b=a.clone()
        expected=original(cm,a).clone();cm._token_to_req_indices_cache=None;actual=m.token_to_req_indices(cm,b)
        assert torch.equal(actual,expected),(starts,n)
        assert torch.equal(a,b),(starts,n,'capacity/tail mismatch')
        assert m.token_to_req_indices(cm,b).data_ptr()==actual.data_ptr()
    # The CPU total remains valid while per-request boundaries change on GPU.
    source=torch.tensor([0,1,4,4],device='cuda',dtype=torch.int32)
    cm=SimpleNamespace(query_start_loc=source,query_start_loc_cpu=torch.tensor([0,2,2,4],dtype=torch.int32),num_actual_tokens=4,_token_to_req_indices_cache=None)
    buf=torch.empty(4,device='cuda',dtype=torch.int32)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cm._token_to_req_indices_cache=None;m.token_to_req_indices(cm,buf)
        graph=torch.cuda.CUDAGraph()
        cm._token_to_req_indices_cache=None
        with torch.cuda.graph(graph):output=m.token_to_req_indices(cm,buf)
        for starts in [[0,1,4,4],[0,0,2,4],[0,3,3,4],[0,4,4,4]]:
            source.copy_(torch.tensor(starts,dtype=torch.int32));graph.replay()
            expect=torch.repeat_interleave(torch.arange(3,device='cuda',dtype=torch.int32),source[1:]-source[:-1],output_size=4)
            assert torch.equal(output,expect),starts
    torch.cuda.synchronize()
    cm=SimpleNamespace(query_start_loc=torch.tensor([0,1],device='cuda',dtype=torch.int32),query_start_loc_cpu=torch.tensor([0,1],dtype=torch.int32),num_actual_tokens=1,_token_to_req_indices_cache=None)
    buffer=torch.empty(2048,device='cuda',dtype=torch.int32);bench={}
    for name,fn in [('stock',original),('fused',m.token_to_req_indices)]:
        trials=[]
        for repeat in range(3):
            torch.cuda.synchronize();t=time.perf_counter()
            for i in range(300):
                cm._token_to_req_indices_cache=None;fn(cm,buffer)
            torch.cuda.synchronize();trials.append((time.perf_counter()-t)*1e6/300)
        bench[name]={'host_wall_us_per_call':trials}
    out={'ok':True,'synthetic_cases':len(cases),'dynamic_graph_cases':4,'benchmark':bench,'scope':'Per-token mapping, duplicate query starts, device boundary changes, fixed output buffers. Not full-model acceptance.'}
    Path(args.out).write_text(json.dumps(out,indent=2));print(json.dumps(out));print('PP5_FAST_TOKEN_MAPPING_VERIFIED')
finally:stop.set()
