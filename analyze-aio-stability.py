"""Summarize complete or partial A-B-A reports without discarding slow rounds."""
import argparse
import json
from pathlib import Path
import statistics as st

def pct(values,p):
    v=sorted(values)
    return v[round((len(v)-1)*p)] if v else 0

def pressure(text):
    return {line.split()[0]:int(line.rsplit('total=',1)[1]) for line in text.splitlines()}

def summary(rows):
    speeds=[r['aggregate_decode_tok_s'] for r in rows]
    tokens=sum(r['completion_tokens'] for r in rows)
    cpu={key:sum(r['cpu_delta'].get(key,0) for r in rows)/1000/tokens
         for key in ('usage_usec','user_usec','system_usec')}
    io={}
    for r in rows:
        for dev,fields in r['io_delta'].items():
            io.setdefault(dev,{})
            for k,v in fields.items(): io[dev][k]=io[dev].get(k,0)+v
    gaps=[b-a for r in rows for req in r['requests'] for a,b in zip(req['sse_offsets_s'],req['sse_offsets_s'][1:])]
    psi={}
    for name in ('cpu_pressure','io_pressure','memory_pressure'):
        psi[name]={kind:sum(pressure(r['after'].get(name,'' )).get(kind,0)-pressure(r['before'].get(name,'')).get(kind,0) for r in rows)/1e6 for kind in ('some','full')}
    return dict(n=len(rows),median=st.median(speeds),mean=st.mean(speeds),min=min(speeds),max=max(speeds),
                cv_percent=st.pstdev(speeds)/st.mean(speeds)*100,p05=pct(speeds,.05),
                e2e_weighted=tokens/sum(r['wall_s'] for r in rows),
                per_request_mean=st.mean(r['per_request_decode_mean'] for r in rows),
                ttft_median_s=st.median(r['ttft_mean_s'] for r in rows),
                cpu_ms_per_token=cpu,io=io,psi_s=psi,
                sse_gap_p99_s=pct(gaps,.99),max_sse_gap_s=max(gaps),
                rounds_gap_over1s=[r['round'] for r in rows if r['max_gap_s']>1],
                rounds_below80pctmedian=[r['round'] for r in rows if r['aggregate_decode_tok_s']<st.median(speeds)*.8],
                throttled_usec=sum(r['cpu_delta'].get('throttled_usec',0) for r in rows),
                memory_events_delta={k:sum(r['after']['memory_events'].get(k,0)-r['before']['memory_events'].get(k,0) for r in rows) for k in rows[0]['after']['memory_events']})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    args=p.parse_args()
    report={}
    for name in ('baseline-a1','candidate-b','baseline-a2'):
        path=args.root/(name+'.json')
        if not path.exists(): continue
        data=json.loads(path.read_text(encoding='utf-8'))
        report[name]={'done':data.get('done',False),'phases':{}}
        for phase in ('warmup','repeated','varied','natural'):
            rows=[r for r in data['rounds'] if r['phase']==phase]
            if rows: report[name]['phases'][phase]=summary(rows)
    out=args.root/'analysis.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    for name,block in report.items():
        print(name,'done=',block['done'])
        for phase,values in block['phases'].items():
            print(phase,json.dumps(values))

if __name__=='__main__': main()
