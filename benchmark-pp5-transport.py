"""Bounded idle-service transport calibration; no model allocation or edits."""
import datetime
import json
import os
from pathlib import Path
import statistics
import threading
import time
import re
import urllib.request
import torch
import torch.distributed as dist

rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
assert world==5
done=threading.Event();phase=['init'];start=time.monotonic()
def hb():
    while not done.wait(5):
        if rank==0:print(f'[HB {time.monotonic()-start:.0f}s] phase={phase[0]}',flush=True)
threading.Thread(target=hb,daemon=True).start()
try:
    if rank==0:
        metrics=urllib.request.urlopen('http://127.0.0.1:9010/metrics',timeout=5).read().decode()
        nums=re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*?\})?\s+([0-9.eE+-]+)',metrics,re.M)
        assert nums and all(float(n)==0 for n in nums),'service busy; refusing calibration'
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=40),device_id=torch.device('cuda',rank))
    cpu=dist.new_group(backend='gloo',timeout=datetime.timedelta(seconds=40))
    token=torch.empty(3,dtype=torch.int32,device='cuda')
    data=torch.empty(65536,dtype=torch.bfloat16,device='cuda')
    items=[torch.empty(16384,dtype=torch.bfloat16,device='cuda') for _ in range(4)]
    metadata={'hidden':((1,4,4096),'bf16'),'pre_mix':((1,4,4096),'bf16'),'residual':((1,4,4096),'bf16'),'shadow_x':((1,4,4096),'bf16')}
    results=[]
    for mode in ['broadcast-12B','chain-128KiB','metadata-four-tensors-and-broadcast']:
        phase[0]=mode
        timings=[]
        for iteration in range(80):
            dist.barrier();torch.cuda.synchronize()
            t=time.perf_counter()
            if mode=='broadcast-12B':
                dist.broadcast(token,src=4)
            elif mode=='chain-128KiB':
                if rank>0:dist.recv(data,src=rank-1)
                if rank<4:dist.send(data,dst=rank+1)
            else:
                if rank>0:
                    received=[None];dist.recv_object_list(received,src=rank-1,group=cpu)
                    for item in items:dist.recv(item,src=rank-1)
                if rank<4:
                    dist.send_object_list([metadata],dst=rank+1,group=cpu)
                    for item in items:dist.send(item,dst=rank+1)
                dist.broadcast(token,src=4)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            value=torch.tensor(elapsed,device='cuda',dtype=torch.float64)
            dist.all_reduce(value,op=dist.ReduceOp.MAX)
            if iteration>=15 and rank==0:timings.append(value.item()*1000)
        if rank==0:
            results.append({'mode':mode,'trials':len(timings),'median_ms':statistics.median(timings),'mean_ms':statistics.mean(timings),'min_ms':min(timings),'max_ms':max(timings)})
            print(json.dumps(results[-1]),flush=True)
    if rank==0:
        out=Path(os.environ['PP5_TRANSPORT_OUT'])
        out.write_text(json.dumps({'env':{k:os.environ.get(k) for k in ['NCCL_ALGO','NCCL_PROTO','NCCL_P2P_DISABLE','NCCL_IB_DISABLE']},'results':results,'scope':'Independent process-group calibration while serving model is idle. Does not reproduce all vLLM scheduling or GPU dependencies.'},indent=2))
        print('TRANSPORT_CALIBRATION_DONE',flush=True)
    dist.destroy_process_group()
finally:done.set()
