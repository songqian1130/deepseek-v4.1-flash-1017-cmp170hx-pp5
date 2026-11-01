"""Bounded complete-answer acceptance for math, code, multi-turn and context."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:9010');p.add_argument('--model',default='deepseek-v4.1-flash-1017');args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=True);stop=threading.Event();began=time.monotonic();phase=['setup']
def heartbeat():
    while not stop.wait(10):print(f'[HB {time.monotonic()-began:.0f}s] phase={phase[0]}',flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
def answer(label,messages,tokens=768):
    phase[0]=label
    body={'model':args.model,'messages':messages,'temperature':0,'max_tokens':tokens,'stream':False}
    req=urllib.request.Request(args.base.rstrip('/')+'/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=150) as r:d=json.loads(r.read())
    (args.out/(label+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2))
    text=d['choices'][0]['message'].get('content') or ''
    assert text and d['choices'][0]['finish_reason']=='stop',(label,'missing complete final answer',d)
    assert not re.search(r'[\ufffd\ue000-\uf8ff]',text),label
    return text,d
def user(text):return [{'role':'user','content':text}]
checks=[]
def record(name,passed,detail=''):
    checks.append({'name':name,'passed':bool(passed),'detail':detail})
    print(json.dumps(checks[-1],ensure_ascii=False),flush=True)
try:
    text,_=answer('arithmetic',user('计算17×23，简短说明过程并给出答案。'))
    record('arithmetic',bool(re.search(r'\b391\b',text)))
    text,_=answer('probability',user('袋中有3个红球和2个蓝球。不放回抽两球，求都是红球的概率。用最简分数回答并说明过程。'))
    record('probability',bool(re.search(r'3\s*[/／]\s*10|\\frac\s*\{3\}\s*\{10\}|0\.3\b|30\s*%',text)))
    text,_=answer('logic',user('甲先于乙到达，丙晚于乙到达，丁早于甲到达。给出从早到晚的完整顺序。'))
    positions=[text.find(x) for x in '丁甲乙丙']
    record('ordering',positions==sorted(positions) and min(positions)>=0)
    text,_=answer('median-code',user('Write only a Python code block defining median(values). It must return the median of a non-empty numeric list, handle even and odd lengths, and not mutate values. Use no imports.'))
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    code=blocks[-1] if blocks else text
    namespace={'__builtins__':{'sorted':sorted,'len':len,'ValueError':ValueError,'float':float,'int':int,'list':list,'tuple':tuple,'sum':sum,'range':range}}
    exec(compile(code,'<median-answer>','exec'),namespace)
    fn=namespace['median'];ok=True
    for values,want in [([3,1,2],2),([4,1,3,2],2.5),([5],5),([-5,-1,-3,-7],-4),([2,2,9,1],2)]:
        original=list(values);ok=ok and fn(values)==want and values==original
    record('median-code',ok,'Five executable tests; original list preserved')
    text,_=answer('multi-turn',[{'role':'user','content':'记住：我订了17盒，每盒23件，每件售价4元。'}, {'role':'assistant','content':'已记住这些数量与单价。'}, {'role':'user','content':'如果全部售出后退款36元，净收入多少？写出计算式。'}])
    record('multi-turn',bool(re.search(r'\b1528\b',text)))
    distractors='\n'.join(f'档案{i:04d}：这里是常规维护记录，记录编号为{i*7+11}，与目标口令无关。' for i in range(240))
    context=distractors[:len(distractors)//2]+'\n关键记录：项目青松的交付口令是Jade-7391，交付批次为第27批。\n'+distractors[len(distractors)//2:]
    text,d=answer('long-context',user(context+'\n请只从关键记录回答：项目青松的交付口令和批次是什么？'),tokens=256)
    record('long-context','Jade-7391' in text and '27' in text,f"prompt_tokens={d['usage']['prompt_tokens']}")
    # Mixed concurrent request sizes exercise independent requests and PP ordering.
    tasks=[('concurrent-a',user('13乘以19等于多少？只给出答案。'),256),('concurrent-b',user('把字符串 abcde 反转，只输出反转结果。'),256)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(answer,*x) for x in tasks]
        texts=[f.result()[0] for f in futures]
    record('concurrent-arithmetic','247' in texts[0]);record('concurrent-string','edcba' in texts[1])
    result={'ok':all(x['passed'] for x in checks),'checks':checks,'elapsed_s':time.monotonic()-began,'scope':'Complete answers and concurrency/context smoke; not an exhaustive intelligence benchmark. Weight/quantization unchanged and separate score-parity evidence required.'}
    (args.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));assert result['ok'],checks
    print('PP5_QUALITY_BATTERY_DONE',flush=True)
finally:stop.set()
