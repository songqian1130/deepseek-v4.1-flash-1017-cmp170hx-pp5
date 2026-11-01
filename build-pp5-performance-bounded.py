"""Bound a Docker image build, emit heartbeats, retain complete build output."""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--image',required=True);a=p.parse_args()
done=threading.Event();began=time.monotonic()
def beat():
    while not done.wait(10):print(f'[HB {time.monotonic()-began:.0f}s] phase=build-r8-cached',flush=True)
threading.Thread(target=beat,daemon=True).start()
try:
    print('[HB 0s] phase=build-r8-cached',flush=True)
    with (a.context/'build-retry.log').open('w') as log:
        subprocess.run(['timeout','--foreground','600','docker','build','--network','host','-f','Containerfile','-t',a.image,'.'],cwd=a.context,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=620)
    result=subprocess.check_output(['docker','image','inspect',a.image],text=True,timeout=20)
    (a.context/'image-inspect.json').write_text(result)
    print(json.loads(result)[0]['Id'],flush=True)
    print('R8_PERFORMANCE_IMAGE_BUILD_DONE',flush=True)
finally:done.set()
