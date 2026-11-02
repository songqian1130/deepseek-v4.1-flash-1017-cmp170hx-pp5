"""Bounded four-stream benchmark with cgroup accounting and SSE gap evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

PROMPTS = [
    '用中文解释为什么昼夜交替发生，并给出一个生活中的例子。',
    'Explain in English how a hash table provides average constant-time lookup, including one caveat.',
    'Write a Python function named clamp that takes x, low, and high and returns x limited to that inclusive range. Include only the function and a short comment.',
    'Explain why Earth has seasons and give one example of how seasons affect daily life.',
]

def quantile(values, q):
    ordered = sorted(values)
    return ordered[round((len(ordered)-1)*q)] if ordered else 0

def request(base, prompt, barrier, natural):
    payload = dict(model='deepseek-v4.1-flash-1017', messages=[dict(role='user',content=prompt)],
                   max_tokens=192, temperature=0, seed=123, stream=True,
                   stream_options=dict(include_usage=True))
    if not natural:
        payload['ignore_eos'] = True
    req = urllib.request.Request(base+'/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type':'application/json'})
    barrier.wait(timeout=20)
    start = time.monotonic()
    times, parts, usage, done = [], [], {}, False
    with urllib.request.urlopen(req, timeout=120) as response:
        for raw in response:
            if not raw.startswith(b'data:'):
                continue
            value = raw[5:].strip()
            if value == b'[DONE]':
                done = True
                break
            event = json.loads(value)
            if event.get('usage'):
                usage = event['usage']
            texts = [v for choice in event.get('choices',[]) for k,v in (choice.get('delta') or {}).items()
                     if k in ('reasoning_content','reasoning','content') and isinstance(v,str) and v]
            if texts:
                times.append(time.monotonic())
                parts.extend(texts)
    end = time.monotonic()
    if not done or not times or not usage.get('completion_tokens'):
        raise RuntimeError('incomplete stream or usage')
    tokens = usage['completion_tokens']
    if not natural and tokens != 192:
        raise RuntimeError(f'fixed length not honored: {tokens}')
    gaps = [b-a for a,b in zip(times,times[1:])]
    text = ''.join(parts)
    return dict(start=start, first=times[0], end=end, prompt=prompt, text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(), completion_tokens=tokens, usage=usage,
                ttft_s=times[0]-start, decode_s=end-times[0], decode_tok_s=(tokens-1)/(end-times[0]),
                sse_offsets_s=[t-start for t in times], max_sse_gap_s=max(gaps,default=0),
                p95_sse_gap_s=quantile(gaps,.95))

def read_fields(path):
    try:
        return {k:int(v) for k,v in (line.split() for line in path.read_text().splitlines())}
    except (OSError,ValueError):
        return {}

def snapshot(group):
    result = dict(wall=time.time(), mono=time.monotonic(), cpu=read_fields(group/'cpu.stat'),
                  memory_events=read_fields(group/'memory.events'), io={})
    for name in ('cpu','io','memory'):
        try:
            result[name+'_pressure'] = (group/(name+'.pressure')).read_text()
        except OSError:
            pass
    try:
        for line in (group/'io.stat').read_text().splitlines():
            dev,*values = line.split()
            result['io'][dev] = {k:int(v) for k,v in (v.split('=') for v in values)}
        result['memory_current'] = int((group/'memory.current').read_text())
        result['memory_stat'] = read_fields(group/'memory.stat')
        result['loadavg'] = Path('/proc/loadavg').read_text().strip()
        result['host_vmstat'] = {k:v for k,v in read_fields(Path('/proc/vmstat')).items()
                                 if k in ('pgmajfault','pswpin','pswpout','pgfault')}
    except OSError as exc:
        result['error'] = str(exc)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--container',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--acceptance',action='store_true')
    args=parser.parse_args()
    pid=int(subprocess.check_output(['docker','inspect','-f','{{.State.Pid}}',args.container],text=True))
    relative=Path(f'/proc/{pid}/cgroup').read_text().strip().split('::',1)[1]
    group=Path('/sys/fs/cgroup')/relative.lstrip('/')
    if not (group/'cpu.stat').exists():
        raise RuntimeError('cgroup unavailable')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    stop=threading.Event()
    state={'phase':'init','round':0}
    sample_path=args.out.with_suffix('.telemetry.jsonl')
    def telemetry():
        with sample_path.open('w') as stream:
            tick=0
            while not stop.is_set():
                stream.write(json.dumps(dict(snapshot(group),**state))+'\n')
                stream.flush()
                if tick%10 == 0:
                    print(f'[HB] phase={state["phase"]} round={state["round"]}',flush=True)
                tick+=1
                stop.wait(1)
    thread=threading.Thread(target=telemetry,daemon=True)
    thread.start()
    report=dict(base=args.base,container=args.container,cgroup=str(group),rounds=[],
                protocol='4 warmup fixed;12 repeated fixed;12 varied fixed;10 natural EOS;192 max tokens;seed123',
                decode_metric='(completion_tokens-1)/(DONE-first content); SSE gaps are chunk gaps')
    try:
        phases = [('warmup',4,False),('repeated',6,False)] if args.acceptance else [('warmup',4,False),('repeated',12,False),('varied',12,False),('natural',10,True)]
        report['acceptance'] = args.acceptance
        for phase,count,natural in phases:
            for index in range(count):
                state.update(phase=phase,round=index+1)
                prompts=PROMPTS if phase!='varied' else [
                    f'Case {index+1}. Discuss example number {(index+1)*7919+i*131}. '+p for i,p in enumerate(PROMPTS)]
                before=snapshot(group)
                barrier=threading.Barrier(4)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures=[pool.submit(request,args.base,p,barrier,natural) for p in prompts]
                    results=[f.result() for f in futures]
                after=snapshot(group)
                start=min(r['start'] for r in results)
                first=min(r['first'] for r in results)
                end=max(r['end'] for r in results)
                tokens=sum(r['completion_tokens'] for r in results)
                cpu={k:after['cpu'][k]-before['cpu'].get(k,0) for k in after['cpu']}
                io={dev:{k:v-before['io'].get(dev,{}).get(k,0) for k,v in values.items()}
                    for dev,values in after['io'].items()}
                record=dict(phase=phase,round=index+1,requests=results,before=before,after=after,
                            completion_tokens=tokens,wall_s=end-start,
                            aggregate_decode_tok_s=(tokens-4)/(end-first),
                            aggregate_e2e_tok_s=tokens/(end-start),
                            per_request_decode_mean=statistics.mean(r['decode_tok_s'] for r in results),
                            ttft_mean_s=statistics.mean(r['ttft_s'] for r in results),
                            max_gap_s=max(r['max_sse_gap_s'] for r in results),cpu_delta=cpu,io_delta=io,
                            cpu_ms_per_token=cpu['usage_usec']/1000/tokens,
                            cpu_avg_cores=cpu['usage_usec']/1e6/(after['mono']-before['mono']))
                report['rounds'].append(record)
                temporary=args.out.with_suffix('.tmp')
                temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                os.replace(temporary,args.out)
                print(json.dumps({k:v for k,v in record.items() if k not in ('requests','before','after','io_delta') }),flush=True)
        report['done']=True
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('[HB] phase=BENCH_DONE',flush=True)
    finally:
        stop.set()
        thread.join(timeout=3)

if __name__=='__main__':
    main()
