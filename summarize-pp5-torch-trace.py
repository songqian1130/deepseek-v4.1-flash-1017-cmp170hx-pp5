"""Summarize Chrome Torch traces without treating NCCL wait as useful compute."""
import argparse
import collections
import gzip
import json
from pathlib import Path

def union_us(events):
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in events if e.get('dur',0)>0)
    total=0;end=None
    for a,b in intervals:
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    results=[]
    for path in sorted(args.directory.glob('*.pt.trace.json*')):
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8') as f:d=json.load(f)
        events=[e for e in d['traceEvents'] if e.get('ph')=='X' and 'dur' in e]
        steps=[e for e in events if e.get('name','').startswith('ProfilerStep#') and e.get('cat')=='user_annotation']
        start=min((e['ts'] for e in steps),default=min(e['ts'] for e in events))
        end=max((e['ts']+e['dur'] for e in steps),default=max(e['ts']+e['dur'] for e in events))
        kernels=[e for e in events if e.get('cat')=='kernel' and e['ts']>=start and e['ts']<end]
        nccl=[e for e in kernels if 'nccl' in e.get('name','').lower()]
        compute=[e for e in kernels if 'nccl' not in e.get('name','').lower()]
        memcpy=[e for e in events if e.get('cat') in ('gpu_memcpy','gpu_memset') and e['ts']>=start and e['ts']<end]
        gpu_api=[e for e in events if e.get('cat')=='cuda_runtime']
        interior=steps[2:-2] if len(steps)>4 else steps
        interior_intervals=[(e['ts'],e['ts']+e['dur']) for e in interior]
        interior_kernels=[e for e in kernels if any(a<=e['ts']<b for a,b in interior_intervals)]
        interior_compute=[e for e in interior_kernels if 'nccl' not in e['name'].lower()]
        interior_apis=collections.defaultdict(lambda:{'calls':0,'sum_us':0})
        for e in gpu_api:
            if any(a<=e['ts']<b for a,b in interior_intervals):
                item=interior_apis[e['name']];item['calls']+=1;item['sum_us']+=e['dur']
        top=collections.defaultdict(lambda:{'calls':0,'sum_us':0,'max_us':0})
        for e in kernels:
            x=top[e['name']];x['calls']+=1;x['sum_us']+=e['dur'];x['max_us']=max(x['max_us'],e['dur'])
        apis=collections.defaultdict(lambda:{'calls':0,'sum_us':0,'max_us':0})
        for e in gpu_api:
            x=apis[e['name']];x['calls']+=1;x['sum_us']+=e['dur'];x['max_us']=max(x['max_us'],e['dur'])
        result={'file':path.name,'baseTimeNanoseconds':d.get('baseTimeNanoseconds'),'window_us':end-start,'steps':len(steps),'categories':dict(collections.Counter(e.get('cat','') for e in events)),
                'compute_kernel_sum_us':sum(e['dur'] for e in compute),'compute_kernel_union_us':union_us(compute),'nccl_kernel_sum_us':sum(e['dur'] for e in nccl),'nccl_kernel_union_us':union_us(nccl),'gpu_memcpy_union_us':union_us(memcpy),'all_gpu_union_us':union_us(kernels+memcpy),
                'interior_steps':len(interior),'interior_compute_us_per_step':sum(e['dur'] for e in interior_compute)/len(interior) if interior else None,
                'interior_gpu_kernel_calls_per_step':len(interior_kernels)/len(interior) if interior else None,
                'top_kernels':sorted([{'name':name,**stats} for name,stats in top.items()],key=lambda x:x['sum_us'],reverse=True)[:25],
                'cuda_api':sorted([{'name':name,**stats} for name,stats in apis.items()],key=lambda x:x['sum_us'],reverse=True)[:15],
                'interior_cuda_api':sorted([{'name':name,**stats} for name,stats in interior_apis.items()],key=lambda x:x['sum_us'],reverse=True),
                'per_step':[{'name':e['name'],'ts':e['ts'],'dur':e['dur']} for e in steps]}
        results.append(result)
    assert results,'No trace files found'
    (args.directory/'trace-summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for r in results:
        print(json.dumps({k:v for k,v in r.items() if k not in ('top_kernels','per_step','categories','cuda_api')},ensure_ascii=False))
        print('TOP_KERNELS',json.dumps(r['top_kernels'][:8],ensure_ascii=False))
        print('CUDA_API',json.dumps(r['cuda_api'][:5],ensure_ascii=False))
    print('TORCH_TRACE_SUMMARY_DONE')

if __name__=='__main__':main()
