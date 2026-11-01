"""One isolated runner-V1 experiment with latent Torch profiling and r4 rollback."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

OLD='deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r4'
NEW='deepseek-v41-pp5-alex-engram-mmap-20261031-r5-v1-profile'
BASE='http://127.0.0.1:9010'
PROMPT='用中文详细解释为什么昼夜交替发生，再依次解释四季、月相、潮汐，每一项给出生活中的例子。'

def main():
    global OLD,NEW
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline-container',default=OLD)
    parser.add_argument('--new-container',default=NEW)
    parser.add_argument('--label',default='v1')
    parser.add_argument('--full-backend',type=Path)
    parser.add_argument('--worker-source',type=Path)
    parser.add_argument('--model-runner-source',type=Path)
    parser.add_argument('--mapping-module',type=Path)
    parser.add_argument('--extension-dir',type=Path)
    parser.add_argument('--hotspot-ab',action='store_true')
    args=parser.parse_args()
    OLD=args.baseline_container;NEW=args.new_container
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();phase=['setup'];done=threading.Event()
    def heartbeat():
        while not done.wait(15):print(f'[HB {time.monotonic()-began:.0f}s] phase={phase[0]}',flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def cmd(argv,deadline=30):return subprocess.check_output(argv,text=True,stderr=subprocess.STDOUT,timeout=deadline)
    def http(path,method='GET'):
        with urllib.request.urlopen(urllib.request.Request(BASE+path,method=method),timeout=20) as r:return r.read().decode()
    def ready(name,seconds):
        until=time.monotonic()+seconds
        log_next=0
        while time.monotonic()<until:
            state=json.loads(cmd(['docker','inspect',name]))[0]['State']
            if not state['Running']:
                raise RuntimeError(f'{name} exited: {state}; '+cmd(['docker','logs','--tail','35',name])[-9000:])
            try:
                if 'deepseek-v4.1-flash-1017' in http('/v1/models'):return
            except (urllib.error.URLError,TimeoutError):pass
            if time.monotonic()>=log_next:
                print('[HB] '+cmd(['docker','logs','--tail','1',name]).strip()[-260:],flush=True)
                log_next=time.monotonic()+35
            time.sleep(3)
        raise TimeoutError(f'{name} startup deadline {seconds}s exceeded')
    def request(label,tokens):
        phase[0]=label
        body={'model':'deepseek-v4.1-flash-1017','messages':[{'role':'user','content':PROMPT}],'max_tokens':tokens,'ignore_eos':True,'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
        req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        start=time.perf_counter();first=last=None;usage={};parts=[];chunks=[]
        with urllib.request.urlopen(req,timeout=120) as response:
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
                        now=time.perf_counter();first=first or now;last=now;parts.append(text);chunks.append(now-start)
        n=usage.get('completion_tokens',0);assert n==tokens,(label,usage)
        result={'label':label,'request':body,'usage':usage,'elapsed_s':time.perf_counter()-start,'ttft_s':first-start,'decode_s':last-first,'decode_tok_s':n/(last-first),'chunk_timestamps_s':chunks,'text':''.join(parts)}
        (out/(label+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(f'[HB] phase={label}-done decode_tok_s={result["decode_tok_s"]:.4f}',flush=True)
        return result
    stopped=False;created=False;success=False
    try:
        old=json.loads(cmd(['docker','inspect',OLD]))[0]
        assert old['State']['Running'],'r4 must be running before experiment'
        metrics=http('/metrics')
        nums=re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*?\})?\s+([0-9.eE+-]+)',metrics,re.M)
        assert nums and all(float(n)==0 for n in nums),'active requests; refusing interruption'
        profdir=out/'traces';profdir.mkdir(exist_ok=True)
        profiler={'profiler':'torch','torch_profiler_dir':str(profdir),'ignore_frontend':True,'torch_profiler_with_stack':False,'torch_profiler_record_shapes':False,'torch_profiler_dump_cuda_time_total':False,'delay_iterations':8,'max_iterations':20,'warmup_iterations':2,'active_iterations':12}
        config=old['Config'];host=old['HostConfig']
        original_cmd=list(config['Cmd'])
        model_cmd=list(original_cmd)
        if '--profiler-config' in model_cmd:
            n=model_cmd.index('--profiler-config');del model_cmd[n:n+2]
        if args.full_backend:
            proof=json.loads((args.full_backend.parent/'full-host-break-verification.json').read_text())
            assert proof['ok'] and len(proof['results'])==2,'real-row graph proof required'
            n=model_cmd.index('--compilation-config')+1
            graphs=json.loads(model_cmd[n]);graphs['cudagraph_mode']='FULL_AND_PIECEWISE'
            model_cmd[n]=json.dumps(graphs)
        launch=['docker','create','--name',NEW,'--init','--gpus','"device=0,1,2,3,4"','--network','host','--shm-size',str(host['ShmSize']),'--memory',str(host['Memory']),'--memory-swap',str(host['MemorySwap']),'--ulimit','memlock=-1','--ulimit','nofile=1048576:1048576','--stop-timeout','120']
        replaced_targets=set()
        if args.full_backend:replaced_targets.add('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/disk_engram_mmap.py')
        if args.worker_source:replaced_targets.add('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py')
        if args.model_runner_source:replaced_targets.add('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
        if args.mapping_module:replaced_targets.add('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/pp5_fast_token_mapping.py')
        if args.extension_dir:replaced_targets.add('/opt/pp5-performance')
        for bind in host['Binds']:
            if bind.split(':')[1] not in replaced_targets:launch+=['-v',bind]
        launch+=['-v',f'{out}:{out}:rw']
        if args.full_backend:
            launch+=['-v',f'{args.full_backend}:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/disk_engram_mmap.py:ro']
        if args.worker_source:
            launch+=['-v',f'{args.worker_source}:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py:ro']
        if args.model_runner_source:
            launch+=['-v',f'{args.model_runner_source}:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro','-e','VLLM_PP5_SHARED_MAPPING=1']
        if args.mapping_module:
            launch+=['-v',f'{args.mapping_module}:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/pp5_fast_token_mapping.py:ro']
        if args.extension_dir:
            launch+=['-v',f'{args.extension_dir}:/opt/pp5-performance:ro','-e','PYTHONPATH=/opt/pp5-performance','-e','VLLM_SERVER_DEV_MODE=1']
            if '--worker-extension-cls' not in model_cmd:
                model_cmd+=['--worker-extension-cls','pp5_perf_extension.PP5PerformanceExtension']
        for env in config['Env']:
            if not env.startswith('VLLM_USE_V2_MODEL_RUNNER='):launch+=['-e',env]
        launch+=['-e','VLLM_USE_V2_MODEL_RUNNER=0','--entrypoint',config['Entrypoint'][0],old['Image'],*model_cmd,'--profiler-config',json.dumps(profiler)]
        (out/'experiment.json').write_text(json.dumps({'image_id':old['Image'],'baseline_container':OLD,'experiment_container':NEW,'runner':'V1','full_backend':str(args.full_backend) if args.full_backend else None,'profiler_inactive_during_baselines':profiler,'original_cmd':original_cmd,'experiment_cmd':model_cmd},indent=2))
        phase[0]='stop-idle-r4';print('[HB] phase=stop-idle-r4',flush=True)
        cmd(['docker','stop','--time','60',OLD],90);stopped=True
        cmd(launch,30);created=True
        phase[0]='load-runner-v1';cmd(['docker','start',NEW])
        ready(NEW,1000)
        (out/'startup-server.log').write_text(cmd(['docker','logs',NEW]))
        print('[HB] phase=V1_HEALTH_READY',flush=True)
        if args.hotspot_ab:
            def rpc(config):
                body={'method':'pp5_perf_configure','args':[json.dumps(config)],'timeout':40}
                req=urllib.request.Request(BASE+'/collective_rpc',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=60) as r:return json.loads(r.read())
            def state(label):
                req=urllib.request.Request(BASE+'/collective_rpc',data=json.dumps({'method':'pp5_perf_state','timeout':40}).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=60) as r:result=json.loads(r.read())
                (out/(label+'-state.json')).write_text(json.dumps(result,indent=2))
            for label,lazy,timed,tokens in [('control-warmup',False,False,32),('control-baseline-1',False,False,256),('control-timing',False,True,128),('lazy-warmup',True,False,32),('lazy-baseline-1',True,False,256),('lazy-timing',True,True,128),('lazy-baseline-2',True,False,256),('control-baseline-2',False,False,256),('lazy-final',True,False,256)]:
                (out/(label+'-configure.json')).write_text(json.dumps(rpc({'lazy_receive':lazy,'timings':timed,'reset':True}),indent=2))
                request(label,tokens);state(label)
            phase[0]='lazy-smoke'
            print(cmd(['timeout','--foreground','180','python3','/tmp/verify-pp5-ssd-async.py','--base',BASE+'/v1','--model','deepseek-v4.1-flash-1017','--out',str(out/'smoke.json')],190),flush=True)
            (out/'server.log').write_text(cmd(['docker','logs',NEW]))
            (out/'container-inspect.json').write_text(cmd(['docker','inspect',NEW]))
            (out/'health.txt').write_text(http('/health'))
            success=True
            print('PP5_HOTSPOT_AB_DONE '+args.label,flush=True)
            return
        request(args.label+'-warmup',32)
        request(args.label+'-baseline-1',256)
        phase[0]='start-profiler';(out/'start-profile.txt').write_text(http('/start_profile','POST'))
        request(args.label+'-profiled',96)
        phase[0]='stop-profiler';(out/'stop-profile.txt').write_text(http('/stop_profile','POST'))
        request(args.label+'-baseline-2',256)
        phase[0]='verify-final'
        (out/'server.log').write_text(cmd(['docker','logs',NEW]))
        (out/'container-inspect.json').write_text(cmd(['docker','inspect',NEW]))
        (out/'health.txt').write_text(http('/health'))
        assert list(profdir.glob('*.json*')),'profiler produced no trace'
        success=True
        print('RUNNER_V1_PROFILE_DONE '+args.label,flush=True)
    except BaseException as e:
        (out/'failure.txt').write_text(repr(e))
        print('EXPERIMENT_FAILED '+repr(e),flush=True)
        raise
    finally:
        if not success and stopped:
            phase[0]='rollback-r4'
            if created:
                try:cmd(['docker','stop','--time','45',NEW],70)
                except Exception as e:print('STOP_NEW_ERROR '+repr(e),flush=True)
                try:(out/'failed-server.log').write_text(cmd(['docker','logs',NEW]))
                except Exception:pass
            cmd(['docker','start',OLD]);ready(OLD,1000)
            print('R4_ROLLBACK_HEALTH_READY',flush=True)
        done.set()

if __name__=='__main__':main()
