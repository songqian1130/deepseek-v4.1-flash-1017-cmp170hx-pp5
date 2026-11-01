"""Activate the baked performance image once, keeping verified r7 rollback."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--image',required=True);p.add_argument('--engram-chunk-rows',type=int);args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=True);began=time.monotonic();done=threading.Event();phase=['preflight']
def heartbeat():
    while not done.wait(15):print(f'[HB {time.monotonic()-began:.0f}s] phase={phase[0]}',flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
def command(argv,timeout=30):return subprocess.check_output(argv,text=True,stderr=subprocess.STDOUT,timeout=timeout)
def http(path,body=None,timeout=15):
    req=urllib.request.Request('http://127.0.0.1:9010'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read().decode()
def ready(name,seconds=1000):
    until=time.monotonic()+seconds;nextlog=0
    while time.monotonic()<until:
        state=json.loads(command(['docker','inspect',name]))[0]['State']
        assert state['Running'],state
        try:
            if 'deepseek-v4.1-flash-1017' in http('/v1/models'):return
        except Exception:pass
        if time.monotonic()>=nextlog:
            print('[HB] '+command(['docker','logs','--tail','1',name]).strip()[-240:],flush=True);nextlog=time.monotonic()+40
        time.sleep(3)
    raise TimeoutError('Finite model startup deadline exceeded')
stopped=False;created=False;ok=False
try:
    old=json.loads(command(['docker','inspect',args.old]))[0]
    assert old['State']['Running']
    image=json.loads(command(['docker','image','inspect',args.image]))[0]
    counts=re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*?\})?\s+([0-9.eE+-]+)',http('/metrics'),re.M)
    assert counts and all(float(x)==0 for x in counts),'Active requests; refusing to interrupt'
    assert command(['docker','ps','-aq','--filter','name=^/'+args.new+'$']).strip()=='','New name exists'
    model_cmd=list(old['Config']['Cmd'])
    for flag in ['--profiler-config','--worker-extension-cls']:
        if flag in model_cmd:
            i=model_cmd.index(flag);del model_cmd[i:i+2]
    host=old['HostConfig']
    launch=['docker','create','--name',args.new,'--init','--gpus','"device=0,1,2,3,4"','--network','host','--shm-size',str(host['ShmSize']),'--memory',str(host['Memory']),'--memory-swap',str(host['MemorySwap']),'--ulimit','memlock=-1','--ulimit','nofile=1048576:1048576','--stop-timeout','120']
    binds=[]
    for bind in host['Binds']:
        target=bind.split(':')[1]
        if target.startswith('/usr/local/lib/python') or target=='/opt/pp5-performance':continue
        # Keep the model and compilation caches; omit old measurement directories.
        if 'performance-breakdown-' in target or 'performance-hotspots-' in target:continue
        launch+=['-v',bind];binds.append(bind)
    launch+=['-v',f'{args.out}:{args.out}:rw']
    for env in old['Config']['Env']:
        if env.split('=',1)[0] in ['VLLM_SERVER_DEV_MODE','PYTHONPATH','VLLM_USE_V2_MODEL_RUNNER','VLLM_PP5_LAZY_RECEIVE','VLLM_PP5_SHARED_MAPPING']:continue
        if args.engram_chunk_rows is not None and env.startswith('VLLM_ENGRAM_MMAP_CHUNK_ROWS='):continue
        launch+=['-e',env]
    if args.engram_chunk_rows is not None:
        assert args.engram_chunk_rows>0
        launch+=['-e',f'VLLM_ENGRAM_MMAP_CHUNK_ROWS={args.engram_chunk_rows}']
    launch+=['-e','VLLM_USE_V2_MODEL_RUNNER=0','-e','VLLM_PP5_LAZY_RECEIVE=1','-e','VLLM_PP5_SHARED_MAPPING=1','--entrypoint','/usr/local/bin/vllm',args.image,*model_cmd]
    # Create before stopping the existing model: mount/config errors leave it running.
    command(launch);created=True
    (args.out/'launch.json').write_text(json.dumps({'image_id':image['Id'],'command':model_cmd,'retained_binds':binds,'lazy_receive':True,'shared_mapping':True,'engram_chunk_rows':args.engram_chunk_rows,'profiler':False,'extension_rpc':False},indent=2))
    phase[0]='stop-idle-r7';command(['docker','stop','--time','60',args.old],90);stopped=True
    phase[0]='load-r8-baked';command(['docker','start',args.new]);ready(args.new)
    phase[0]='r8-health-ready';print('[HB] phase=R8_HEALTH_READY',flush=True)
    (args.out/'container-inspect.json').write_text(command(['docker','inspect',args.new]))
    (args.out/'startup-server.log').write_text(command(['docker','logs',args.new]))
    # Reuse the request implementation without rerunning the experiment driver.
    phase[0]='short-smoke'
    print(command(['timeout','--foreground','180','python3','/tmp/verify-pp5-ssd-async.py','--base','http://127.0.0.1:9010/v1','--model','deepseek-v4.1-flash-1017','--out',str(args.out/'smoke.json')],190),flush=True)
    phase[0]='parity-to-r7'
    reference=args.out.parent/'r7-lazy-receive/quality-parity'
    print(command(['timeout','--foreground','180','python3','/tmp/verify-pp5-no-degradation.py','--out',str(args.out/'quality-parity'),'--reference-dir',str(reference)],190),flush=True)
    phase[0]='final-health';assert 'deepseek-v4.1-flash-1017' in http('/v1/models')
    ok=True;print('PP5_BAKED_PERFORMANCE_ACTIVATION_DONE',flush=True)
except BaseException as exc:
    (args.out/'failure.txt').write_text(repr(exc));print('ACTIVATION_FAILED '+repr(exc),flush=True);raise
finally:
    if not ok and created:
        try:command(['docker','logs',args.new]);command(['docker','stop','--time','40',args.new],60)
        except Exception:pass
    if not ok and stopped:
        phase[0]='rollback-r7';command(['docker','start',args.old]);ready(args.old)
        if '--worker-extension-cls' in old['Config']['Cmd']:
            http('/collective_rpc',{'method':'pp5_perf_configure','args':[json.dumps({'lazy_receive':True,'timings':False})],'timeout':30})
        print('PREVIOUS_FAST_ROLLBACK_HEALTH_READY '+args.old,flush=True)
    done.set()
