"""Verify the baked PP5 image, effective sources and idle service state."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
p.add_argument('--container',required=True)
p.add_argument('--image',required=True)
p.add_argument('--native-reference',type=Path,required=True)
p.add_argument('--engram-chunk-rows',type=int)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
stop=threading.Event();began=time.monotonic()
def heartbeat():
    while not stop.wait(10):print(f'[HB {time.monotonic()-began:.0f}s] phase=final-source-and-health',flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
def cmd(argv,timeout=45):return subprocess.check_output(argv,text=True,stderr=subprocess.STDOUT,timeout=timeout)
def http(path):
    with urllib.request.urlopen('http://127.0.0.1:9010'+path,timeout=10) as r:return r.status,r.read().decode()
try:
    live=json.loads(cmd(['docker','inspect',args.container]))[0]
    image=json.loads(cmd(['docker','image','inspect',args.image]))[0]
    assert live['Image']==image['Id'] and live['State']['Running'] and not live['State']['OOMKilled']
    env=dict(x.split('=',1) for x in live['Config']['Env'] if '=' in x)
    for k,v in {'VLLM_USE_V2_MODEL_RUNNER':'0','VLLM_PP5_LAZY_RECEIVE':'1','VLLM_PP5_SHARED_MAPPING':'1'}.items():assert env[k]==v
    if args.engram_chunk_rows is not None:assert env['VLLM_ENGRAM_MMAP_CHUNK_ROWS']==str(args.engram_chunk_rows)
    assert env.get('VLLM_SERVER_DEV_MODE','0')=='0'
    command=live['Config']['Cmd']
    assert '--profiler-config' not in command and '--worker-extension-cls' not in command
    assert not any('/usr/local/lib/python' in x.split(':')[1] for x in live['HostConfig']['Binds'])
    assert http('/health')[0]==200
    models=json.loads(http('/v1/models')[1]);assert any(x['id']=='deepseek-v4.1-flash-1017' for x in models['data'])
    counts=re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*?\})?\s+([0-9.eE+-]+)',http('/metrics')[1],re.M)
    assert counts and all(float(x)==0 for x in counts)
    package='/usr/local/lib/python3.12/dist-packages/vllm'
    print(cmd(['docker','exec',args.container,'python3','/opt/source-alignment/verify-pp5-reference-alignment.py','--package',package,'--manifest','/opt/pp5-performance-source/alignment-manifest.json','--reference-engram','/opt/source-alignment/reference-engram.py','--out',str(args.out/'source-verification.json')],100),flush=True)
    print(cmd(['docker','exec',args.container,'python3','/opt/pp5-performance-source/verify-pp5-lazy-contract.py',package+'/v1/worker/gpu_worker.py']),flush=True)
    inspect_code='''
import hashlib,json,pathlib
p=pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm')
m=json.loads(pathlib.Path('/opt/pp5-performance-source/alignment-manifest.json').read_text())
paths=list(m['approved_performance_revision']['files'])+[m['approved_performance_revision']['new_helper'],m['approved_adapter_revision']['path']]
sources={k:hashlib.sha256((p.parent/k).read_bytes()).hexdigest() for k in paths}
native={str(x.relative_to(p)):{'size':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in p.rglob('*.so')}
print(json.dumps({'source_sha256':sources,'native':native}))
'''
    hashes=json.loads(cmd(['docker','exec',args.container,'python3','-c',inspect_code]))
    native_reference=json.loads(args.native_reference.read_text())['files']
    assert hashes['native']==native_reference, 'Native binary set/hash differs from verified r4/r6 lineage'
    logs=cmd(['docker','logs',args.container])
    severe=[x for x in logs.splitlines() if re.search(r'Traceback \(most recent|CUDA out of memory|torch\.OutOfMemoryError|EngineCore.*failed|\bERROR\b',x,re.I)]
    assert not severe,severe
    remnants=[]
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            if path.read_text().strip() in ('bpftrace','py-spy'):remnants.append(str(path.parent))
        except (OSError,PermissionError):pass
    assert not remnants,remnants
    rollback={}
    for name in ['deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r4','deepseek-v41-pp5-alex-engram-mmap-20261031-r5-v1-profile','deepseek-v41-pp5-alex-engram-mmap-20261031-r6-v1-full-profile','deepseek-v41-pp5-alex-engram-mmap-20261101-r7-lazy-profile','deepseek-v41-pp5-alex-engram-mmap-20261101-r8-performance']:
        state=json.loads(cmd(['docker','inspect',name]))[0]['State'];assert not state['Running'];rollback[name]=state['Status']
    (args.out/'final-container-inspect.json').write_text(json.dumps([live],indent=2))
    (args.out/'final-server.log').write_text(logs)
    (args.out/'source-and-native-hashes.json').write_text(json.dumps(hashes,indent=2))
    (args.out/'gpu-final.csv').write_text(cmd(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,temperature.gpu,power.draw,pcie.link.gen.current,pcie.link.width.current','--format=csv']))
    (args.out/'memory-final.txt').write_text(Path('/proc/meminfo').read_text())
    result={'ok':True,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'health':200,'container':args.container,'image_id':image['Id'],'engram_chunk_rows':int(env.get('VLLM_ENGRAM_MMAP_CHUNK_ROWS','1024')),'source_bind_mounts':False,'profiler_configured':False,'debug_rpc_enabled':False,'active_requests':counts,'native_binaries_unchanged':len(native_reference),'severe_errors':severe,'sampler_remnants':remnants,'rollback_retained':rollback,'source_sha256':hashes['source_sha256']}
    (args.out/'final-acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True);print('PP5_PERFORMANCE_FINAL_ACCEPTANCE_DONE',flush=True)
finally:stop.set()
