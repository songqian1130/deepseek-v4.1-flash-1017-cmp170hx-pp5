"""Clone the measured container with only its port and identity changed."""
import json
from pathlib import Path
import subprocess

ROOT=Path('/run/media/xpz/models/deploy-llm/aio-stability-20261102')
SOURCE='deepseek-v41-aiultrayang-pp5-ssd-matched-20261102'
DEST='deepseek-v41-pp5-aio-9010-20261102'
source=json.loads(subprocess.check_output(['docker','inspect',SOURCE]))[0]
config,host=source['Config'],source['HostConfig']
command=[v.replace('--port=18023','--port=9010') for v in config['Cmd']]
assert command.count('--port=9010')==1
assert host['NetworkMode']=='host' and host['Memory']==112*1024**3
assert host['MemorySwap']==host['Memory'] and host['Init']
assert len(config['Entrypoint'])==1
args=['docker','create','--name',DEST,'--gpus','all','--network','host','--init',
      '--shm-size',str(host['ShmSize']),'--memory',str(host['Memory']),
      '--memory-swap',str(host['MemorySwap']),'--stop-timeout',str(config['StopTimeout'])]
for limit in host['Ulimits']:
    args+=['--ulimit',f"{limit['Name']}={limit['Soft']}:{limit['Hard']}"]
for mount in host['Binds']: args+=['-v',mount]
for env in config['Env']: args+=['-e',env]
for k,v in (config.get('Labels') or {}).items(): args+=['--label',f'{k}={v}']
args+=['--label','local.deepseek.promoted=aio-stability-20261102',
       '--entrypoint',config['Entrypoint'][0],source['Image']]+command
existing=subprocess.run(['docker','inspect',DEST],capture_output=True)
if existing.returncode:
    subprocess.run(args,check=True,timeout=60)
else:
    assert json.loads(existing.stdout)[0]['State']['Status']=='created'
dest=json.loads(subprocess.check_output(['docker','inspect',DEST]))[0]
assert dict(v.split('=',1) for v in dest['Config']['Env']) == dict(v.split('=',1) for v in config['Env']), 'Env values'
for key in ('Entrypoint','WorkingDir','User','StopTimeout'):
    assert dest['Config'][key]==config[key],key
for key in ('Binds','Memory','MemorySwap','ShmSize','Ulimits','DeviceRequests','NetworkMode','Init'):
    assert dest['HostConfig'][key]==host[key],key
assert dest['Image']==source['Image'] and dest['Config']['Cmd']==command
(ROOT/'reports'/'promotion-container.json').write_text(json.dumps(dest,indent=2)+'\n')
print('CLONE_VERIFIED port 18023 -> 9010; image/env/mounts/resources preserved',flush=True)
