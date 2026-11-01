"""Attribute interior profiled CPU step time to innermost recorded operations."""
import argparse
import collections
import gzip
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
result=[]
for path in sorted(args.directory.glob('*.pt.trace.json*')):
    with gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8') as f:
        data=json.load(f)
    events=[x for x in data['traceEvents'] if x.get('ph')=='X' and x.get('dur',0)>0]
    steps=sorted([x for x in events if x.get('cat')=='user_annotation' and x.get('name','').startswith('ProfilerStep#')],key=lambda x:x['ts'])[2:-2]
    selftime=collections.Counter();categories=collections.Counter();stepdata=[]
    for step in steps:
        lo=step['ts'];hi=lo+step['dur'];tid=step['tid']
        ops=[e for e in events if e.get('tid')==tid and e.get('cat') in ('cpu_op','cuda_runtime','cuda_driver','user_annotation') and e is not step and lo<=e['ts']<hi]
        markers=[]
        for i,e in enumerate(ops):
            markers += [(e['ts'],1,i),(min(e['ts']+e['dur'],hi),0,i)]
        active=set();prev=lo
        for t,isstart,i in sorted(markers):
            dt=t-prev
            if dt>0:
                if active:
                    leaf=min((ops[j] for j in active),key=lambda x:x['dur'])
                    selftime[leaf['name']]+=dt;categories[leaf.get('cat')]+=dt
                else:selftime['<unrecorded host gap>']+=dt
            if isstart:active.add(i)
            else:active.discard(i)
            prev=t
        selftime['<unrecorded host gap>']+=hi-prev
        stepdata.append({'step':step['name'],'duration_ms':step['dur']/1000,'tid':tid,'largest_ops':[{'name':e['name'],'cat':e.get('cat'),'offset_ms':(e['ts']-lo)/1000,'duration_ms':e['dur']/1000} for e in sorted(ops,key=lambda x:x['dur'],reverse=True)[:12]]})
    row={'file':path.name,'steps':len(steps),'mean_step_ms':sum(x['dur'] for x in steps)/len(steps)/1000,'mean_self_ms':{k:v/len(steps)/1000 for k,v in selftime.most_common(20)},'sample_step':stepdata[len(stepdata)//2]}
    result.append(row);print(json.dumps(row,ensure_ascii=False))
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('HOST_HOTSPOTS_ANALYZED')
