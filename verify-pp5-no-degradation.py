"""Compare unchanged GPU arithmetic under stock and lazy PP scheduling."""
import argparse
import json
from pathlib import Path
import threading
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:9010');p.add_argument('--reference-dir',type=Path);args=p.parse_args()
stop=threading.Event();start=time.monotonic();phase=['setup']
def heartbeat():
    while not stop.wait(10):print(f'[HB {time.monotonic()-start:.0f}s] phase={phase[0]}',flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
def post(path,data,timeout=100):
    req=urllib.request.Request(args.base+path,data=json.dumps(data,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read())
def configure(lazy):return post('/collective_rpc',{'method':'pp5_perf_configure','args':[json.dumps({'lazy_receive':lazy,'timings':False,'reset':True})],'timeout':30})
prompts=[
    '算出17乘以23，并写出计算过程。',
    '所有A都是B，所有B都是C。能否推出所有A都是C？说明理由。',
    'Write a Python function that returns the median of a non-empty list without changing the original list.',
    'A bag has 3 red balls and 2 blue balls. Draw two without replacement. What is the probability both are red? Show the fraction.',
    '阅读后回答：甲先于乙到达，丙晚于乙到达，丁早于甲到达。请给出四人的到达顺序。',
]
args.out.mkdir(parents=True,exist_ok=True);pairs=[];ok=False
try:
    for i,prompt in enumerate(prompts):
        responses=[]
        if args.reference_dir:
            responses.append(json.loads((args.reference_dir/f'case{i}-lazy1.json').read_text()))
        for lazy in ([True] if args.reference_dir else [False,True]):
            phase[0]=f'logprob-parity-{i}-lazy-{lazy}'
            if not args.reference_dir:configure(lazy)
            body={'model':'deepseek-v4.1-flash-1017','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':64,'ignore_eos':True,'logprobs':True,'top_logprobs':5,'stream':False}
            data=post('/v1/chat/completions',body,120)
            (args.out/f'case{i}-lazy{int(lazy)}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
            responses.append(data)
        a,b=[x['choices'][0] for x in responses]
        # Compare full returned message and every recorded logprob entry, ignoring request IDs/timing.
        msg_a=a['message'];msg_b=b['message'];log_a=a.get('logprobs');log_b=b.get('logprobs')
        assert log_a is not None and log_b is not None, 'No logprobs returned; cannot claim score parity'
        def flatten(log):
            rows=[]
            if isinstance(log,list):
                for x in log:rows+=flatten(x)
            elif isinstance(log,dict):
                if isinstance(log.get('logprob'),(int,float)):rows.append((log.get('token'),float(log['logprob'])))
                for k,v in log.items():
                    if k!='logprob' and isinstance(v,(list,dict)):rows+=flatten(v)
            return rows
        fa,fb=flatten(log_a),flatten(log_b)
        assert fa and len(fa)==len(fb),(i,len(fa),len(fb))
        maxdiff=max(abs(x[1]-y[1]) for x,y in zip(fa,fb))
        tokens_equal=all(x[0]==y[0] for x,y in zip(fa,fb))
        equal=msg_a==msg_b and tokens_equal and maxdiff<=1e-5
        pairs.append({'case':i,'prompt':prompt,'messages_equal':msg_a==msg_b,'logprob_entries':len(fa),'top_tokens_equal':tokens_equal,'max_logprob_abs_diff':maxdiff,'passed':equal})
        print(json.dumps(pairs[-1],ensure_ascii=False),flush=True)
    ok=all(x['passed'] for x in pairs)
    (args.out/'summary.json').write_text(json.dumps({'ok':ok,'pairs':pairs,'scope':'Five fixed reasoning/code/math prompts, first64 generation tokens with top5 logprobs; scheduling parity, not general model quality proof.'},ensure_ascii=False,indent=2))
    assert ok,'Output/score parity did not hold; inspect raw responses'
    print('PP5_NO_DEGRADATION_PARITY_DONE',flush=True)
finally:
    if not args.reference_dir:configure(ok)
    stop.set()
