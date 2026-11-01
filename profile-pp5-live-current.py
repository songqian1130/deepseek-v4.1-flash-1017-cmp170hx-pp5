"""Bounded external profiling of the existing PP5 workers; no reload or model edits."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--container',default='deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r4')
    p.add_argument('--native',action='store_true')
    p.add_argument('--tokens',type=int,default=256)
    p.add_argument('--skip-sampling',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();done=threading.Event();phase=['setup']
    def heartbeat():
        while not done.wait(5):print(f'[HB {time.monotonic()-began:.0f}s] phase={phase[0]}',flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def command(a,timeout=15):return subprocess.check_output(a,text=True,timeout=timeout)
    top=command(['docker','top',args.container,'-eo','pid,args'])
    workers={int(m.group(2)):int(m.group(1)) for m in re.finditer(r'^\s*(\d+)\s+VLLM::Worker_PP(\d+)\s*$',top,re.M)}
    assert set(workers)==set(range(5)),top
    (args.out/'workers.json').write_text(json.dumps(workers,indent=2))
    (args.out/'gpu-before.csv').write_text(command(['nvidia-smi','--query-gpu=index,name,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu','--format=csv']))
    hz=os.sysconf('SC_CLK_TCK')
    def snapshot():
        result={'at':time.time(),'workers':{}}
        for rank,pid in workers.items():
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            try:
                io={k:int(v) for k,v in (line.split(':') for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
            except (PermissionError, FileNotFoundError):
                io={}
            result['workers'][rank]={'pid':pid,'user_ticks':int(fields[11]),'sys_ticks':int(fields[12]),
                'minor_faults':int(fields[7]),'major_faults':int(fields[9]),'io':io}
        result['vmstat']={k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines())
                           if k in ['pgmajfault','pgfault','pswpin','pswpout']}
        return result
    def request(label,tokens):
        phase[0]=label
        body={'model':'deepseek-v4.1-flash-1017','messages':[{'role':'user','content':
            '用中文详细解释为什么昼夜交替发生，再依次解释四季、月相、潮汐，每一项给出生活中的例子。'}],
              'max_tokens':tokens,'temperature':0,'stream':True,'stream_options':{'include_usage':True},'ignore_eos':True}
        start=time.perf_counter();first=None;last=None;usage={};parts=[];chunks=[]
        req=urllib.request.Request('http://127.0.0.1:9010/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=100) as response:
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith('data:'):continue
                value=line[5:].strip()
                if value=='[DONE]':break
                item=json.loads(value)
                if item.get('usage'):usage=item['usage']
                for choice in item.get('choices') or []:
                    delta=choice.get('delta') or {}
                    text=''.join(delta.get(k) or '' for k in ['reasoning_content','reasoning','content'])
                    if text:
                        now=time.perf_counter();first=first or now;last=now;chunks.append(now-start);parts.append(text)
        elapsed=time.perf_counter()-start;n=usage.get('completion_tokens',0)
        assert n==tokens,(label,n,tokens)
        result={'label':label,'request':body,'usage':usage,'elapsed_s':elapsed,'ttft_s':first-start,
                'decode_s':last-first,'completion_tokens':n,'decode_tok_s':n/(last-first),
                'inter_token_tok_s':(n-1)/(last-first),
                'end_to_end_tok_s':n/elapsed,'chunk_timestamps_s':chunks,'text':''.join(parts)}
        (args.out/(label+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(f'[HB] phase={label}-done tokens={n} decode_tok_s={result["decode_tok_s"]:.3f}',flush=True)
        return result
    monitor_stop=threading.Event();gpu=None;profile_processes=[];logs=[]
    def watch():
        with (args.out/'process-samples.jsonl').open('w') as f:
            while not monitor_stop.is_set():
                item=snapshot();item['phase']=phase[0];f.write(json.dumps(item)+'\n');f.flush();monitor_stop.wait(.2)
    try:
        request('warmup',32)
        phase[0]='baseline-1'
        threading.Thread(target=watch,daemon=True).start()
        gpu_file=(args.out/'gpu-samples.csv').open('w');logs.append(gpu_file)
        gpu=subprocess.Popen(['timeout','--foreground','180','nvidia-smi',
            '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu,pcie.link.gen.current,pcie.link.width.current',
            '--format=csv,noheader,nounits','-lms','200'],stdout=gpu_file,stderr=subprocess.STDOUT)
        boundaries={};results=[]
        boundaries['baseline-1']={'before':snapshot()};results.append(request('baseline-1',args.tokens));boundaries['baseline-1']['after']=snapshot()
        if not args.skip_sampling:
            phase[0]='attach-samplers'
            for rank,pid in workers.items():
                log=(args.out/f'pyspy-rank{rank}.log').open('w');logs.append(log)
                profile_processes.append(subprocess.Popen(['timeout','--foreground','35','/tmp/pp5-profile-tools/bin/py-spy','record',
                    '--pid',str(pid),'--duration','20','--rate','50','--threads','--idle',
                    *(['--native'] if args.native else ['--nonblocking']),'--format','raw','--full-filenames',
                    '--output',str(args.out/f'rank{rank}.raw')],stdout=log,stderr=subprocess.STDOUT))
            time.sleep(1)
            for proc in profile_processes:assert proc.poll() is None,'Sampler attach failed'
            boundaries['sampled']={'before':snapshot()};results.append(request('sampled',args.tokens));boundaries['sampled']['after']=snapshot()
            for proc in profile_processes:assert proc.wait(timeout=35)==0,'Sampler failed'
        boundaries['baseline-2']={'before':snapshot()};results.append(request('baseline-2',args.tokens));boundaries['baseline-2']['after']=snapshot()
        summary={'workers':workers,'clock_ticks_per_second':hz,'boundaries':boundaries,
                 'requests':[{k:v for k,v in x.items() if k not in ['request','text','chunk_timestamps_s']} for x in results]}
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary['requests'],indent=2),flush=True)
        print('LIVE_PROFILE_DONE',flush=True)
    finally:
        monitor_stop.set();done.set()
        for proc in [*profile_processes,gpu]:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
        for handle in logs:handle.close()


if __name__=='__main__':main()
