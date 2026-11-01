"""CPU-only row-gather calibration on fresh random pages and cached repeats."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import statistics
import threading
import time

import numpy as np

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
done=threading.Event();began=time.monotonic();phase=['setup']
def beat():
    while not done.wait(5):print(f'[HB {time.monotonic()-began:.0f}s] phase={phase[0]}',flush=True)
threading.Thread(target=beat,daemon=True).start()
try:
    path='/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram_mmap.py'
    spec=importlib.util.spec_from_file_location('actual_mmap_reader',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=Path('/srv/models/DeepSeek-V4.1-Flash-1017-official')
    index=json.loads((model/'model.safetensors.index.json').read_text())
    key='layers.1.engram.embed.weight'
    header,_=module._read_header(str(model/index['weight_map'][key]))
    rows,dim=header[key]['shape']
    table=module.MmapEngramTable(str(model),1,rows,dim,dim//32,0,rows,workers=16)
    rng=np.random.default_rng(9711);results=[]
    def stats(values):
        values=sorted(values)
        return {'mean_ms':statistics.mean(values),'median_ms':statistics.median(values),'p95_ms':values[int((len(values)-1)*.95)]}
    for count in [32,64]:
        for chunk in [1024,16,8,4,2]:
            phase[0]=f'rows-{count}-chunk-{chunk}';table.chunk_rows=chunk
            cold=[];hot=[];majs=0
            for _ in range(48):
                ids=rng.integers(0,table.rows,size=count,dtype=np.int64)
                weights=np.empty((count,256),np.uint8);scales=np.empty((count,8),np.uint8)
                before=resource.getrusage(resource.RUSAGE_SELF).ru_majflt;t=time.perf_counter()
                table.gather_into(ids,weights,scales);cold.append((time.perf_counter()-t)*1000)
                majs+=resource.getrusage(resource.RUSAGE_SELF).ru_majflt-before
                expected_w=weights.copy();expected_s=scales.copy()
                # Reordered duplicate ids exercise the existing inverse mapping.
                duplicate_ids=np.concatenate([ids[::-1],ids[:3]])
                w=np.empty((count+3,256),np.uint8);s=np.empty((count+3,8),np.uint8)
                t=time.perf_counter();table.gather_into(duplicate_ids,w,s);hot.append((time.perf_counter()-t)*1000)
                assert np.array_equal(w,np.concatenate([expected_w[::-1],expected_w[:3]]))
                assert np.array_equal(s,np.concatenate([expected_s[::-1],expected_s[:3]]))
            row={'rows':count,'chunk_rows':chunk,'fresh_random_pages':stats(cold),'cached_repeat':stats(hot),'major_faults':majs,'byte_parity':True}
            results.append(row);print(json.dumps(row),flush=True)
    result={'ok':True,'cases':results,'scope':'Standalone actual reader; fresh random pages are comparable distributions, not identical cold pages; no service configuration changed.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print('MMAP_CHUNK_CALIBRATION_DONE',flush=True)
finally:done.set()
