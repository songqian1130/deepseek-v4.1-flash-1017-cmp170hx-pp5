"""Bounded real-table graph replay verification before a full-model reload."""
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import torch
from safetensors import safe_open
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode

root=Path(os.environ['PP5_FULL_TEST_ROOT']);start=time.monotonic();done=threading.Event()
def hb():
    while not done.wait(5):print(f'[HB {time.monotonic()-start:.0f}s] phase=full-host-break-real-rows',flush=True)
threading.Thread(target=hb,daemon=True).start()
try:
    spec=importlib.util.spec_from_file_location('candidate_disk_engram_mmap',root/'disk_engram_mmap.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE),parallel_config=SimpleNamespace(enable_dbo=False),scheduler_config=SimpleNamespace(max_num_batched_tokens=4))
    module.get_current_vllm_config=lambda:config
    manifest=Path(os.environ['DS41_ENGRAM_MANIFEST']);index=json.loads(manifest.read_text())['weight_map']
    results=[]
    torch.cuda.set_device(0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    for layer in [1,14]:
        key=f'layers.{layer}.engram.embed.weight'
        header,_=module._read_header(str(manifest.parent/index[key]))
        rows,dim=header[key]['shape']
        reader=module.DiskEngramRows(rows,dim,32,layer)
        source=torch.zeros((1,12),dtype=torch.int32,device='cuda')
        # Allocate all staging buffers before graph capture.
        reader.prefetch(source,0,rows);reader.consume(source);torch.cuda.synchronize()
        cap=BreakableCUDAGraphCapture()
        with cap:
            ids=source+0
            reader.prefetch(ids,0,rows)
            weights,scales,compact=reader.consume(ids)
            selected=compact.flatten().to(torch.int64)
            actual_w=weights.view(torch.uint8).index_select(0,selected)
            actual_s=scales.view(torch.uint8).index_select(0,selected)
        torch.cuda.synchronize()
        assert cap.num_eager_breaks==2,(cap.num_graphs,cap.num_eager_breaks)
        patterns=[list(range(12)),[5]*12,[rows-1-i for i in range(12)],[3,3,7,7,0,0,9,9,11,11,1,1],[0]*12,list(range(1024,1036)),list(range(12))]
        for k,pattern in enumerate(patterns):
            source.copy_(torch.tensor([pattern],dtype=torch.int32))
            cap.replay();torch.cuda.synchronize()
            expected=[]
            for suffix in ['weight','scale']:
                key=f'layers.{layer}.engram.embed.{suffix}'
                with safe_open(str(manifest.parent/index[key]),framework='pt',device='cpu') as f:
                    sl=f.get_slice(key)
                    expected.append(torch.cat([sl[row:row+1].view(torch.uint8) for row in pattern]))
            assert torch.equal(actual_w.cpu(),expected[0]),('weight bytes',layer,k)
            assert torch.equal(actual_s.cpu(),expected[1]),('scale bytes',layer,k)
        results.append({'layer':layer,'cases':len(patterns),'weight_and_scale_byte_equal':True,'graphs':cap.num_graphs,'host_breaks':cap.num_eager_breaks,'read_calls':reader.read_calls})
        reader.executor.shutdown(wait=True)
        print(json.dumps(results[-1]),flush=True)
    (root/'full-host-break-verification.json').write_text(json.dumps({'ok':True,'results':results,'scope':'Real FP8 weight/scale bytes through capture/replay, changing IDs/unique counts. Full-model quality and performance require separate validation.'},indent=2))
    print('FULL_HOST_BREAK_VERIFICATION_DONE',flush=True)
finally:done.set()
