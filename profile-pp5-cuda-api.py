"""Bounded live CUDA runtime uprobes. Run as root on the GPU host."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--duration', type=int, default=40)
    ap.add_argument('--focused', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    done = threading.Event()
    phase = ['setup']
    def heartbeat():
        while not done.wait(5):
            print(f'[HB {time.monotonic()-start:.0f}s] phase={phase[0]}', flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    top = subprocess.check_output(['docker', 'top', 'deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r4', '-eo', 'pid,args'], text=True, timeout=10)
    workers = {int(m.group(2)): int(m.group(1)) for m in re.finditer(r'^\s*(\d+)\s+VLLM::Worker_PP(\d+)\s*$', top, re.M)}
    assert set(workers) == set(range(5)), top
    libs = sorted({line.split()[-1] for line in Path(f'/proc/{workers[0]}/maps').read_text().splitlines() if 'libcudart' in line})
    condition = ' || '.join(f'pid == {pid}' for pid in workers.values())
    funcs = ['cudaMemcpyAsync', 'cudaMemcpy', 'cudaStreamSynchronize', 'cudaEventSynchronize', 'cudaDeviceSynchronize', 'cudaGraphLaunch', 'cudaLaunchKernel']
    if args.focused: funcs = funcs[:5]
    parts = ['BEGIN { printf("PROBE_READY\\n"); }', f'interval:s:{args.duration} {{ exit(); }}']
    for lib in libs:
        lib = f'/proc/{workers[0]}/root{lib}'
        for i, func in enumerate(funcs):
            extra = ''
            if 'Memcpy' in func:
                extra = f' @kind[tid] = arg3; @bytes[tid] = arg2; @stream[tid] = arg4; @copy_bytes[pid, "{func}", arg3] = sum(arg2); @copy_count[pid, "{func}", arg3] = count();'
            returns = ''
            if 'Memcpy' in func:
                returns = (f' @copy_us[pid,@kind[tid],@bytes[tid]] = sum($us); @copy_calls[pid,@kind[tid],@bytes[tid]] = count(); '
                           f'@copy_hist[pid,@kind[tid]] = hist($us); '
                           f'if ($us > 10000 && @shown[pid] < 4) {{ printf("SLOW_COPY pid=%d kind=%d bytes=%d stream=%lx us=%d\\n",pid,@kind[tid],@bytes[tid],@stream[tid],$us); @shown[pid] = @shown[pid]+1; }} '
                           'delete(@kind[tid]); delete(@bytes[tid]); delete(@stream[tid]); ')
            parts.append(f'uprobe:{lib}:{func} /{condition}/ {{ @began[tid,{i}] = nsecs;{extra} }}')
            parts.append(f'uretprobe:{lib}:{func} /@began[tid,{i}]/ {{ '
                         f'$us = (nsecs-@began[tid,{i}])/1000; @calls[pid,"{func}"] = count(); '
                         f'@total_us[pid,"{func}"] = sum($us); @max_us[pid,"{func}"] = max($us); '
                         f'@hist_us[pid,"{func}"] = hist($us);{returns} delete(@began[tid,{i}]); }}')
    script = args.out/'cuda-api.bt'
    script.write_text('\n'.join(parts)+'\n')
    (args.out/'workers.json').write_text(json.dumps(workers, indent=2))
    proc = None
    try:
        phase[0] = 'attach'
        with (args.out/'bpftrace.out').open('w') as output, (args.out/'bpftrace.err').open('w') as error:
            proc = subprocess.Popen(['timeout', '--foreground', str(args.duration+10), 'bpftrace', '-B', 'line', str(script)], stdout=output, stderr=error)
            deadline = time.monotonic()+12
            while 'PROBE_READY' not in (args.out/'bpftrace.out').read_text():
                assert proc.poll() is None, (args.out/'bpftrace.err').read_text()
                assert time.monotonic() < deadline, 'probe attach timed out'
                time.sleep(.2)
            phase[0] = 'request'
            body = {'model':'deepseek-v4.1-flash-1017','messages':[{'role':'user','content':'用中文详细解释为什么昼夜交替发生，再依次解释四季、月相、潮汐，每一项给出生活中的例子。'}], 'max_tokens':args.tokens,'temperature':0,'stream':True,'stream_options':{'include_usage':True},'ignore_eos':True}
            began = time.perf_counter()
            first = last = None
            usage = {}
            chunks = []
            texts = []
            request = urllib.request.Request('http://127.0.0.1:9010/v1/chat/completions', data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=60) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith('data:'): continue
                    data = line[5:].strip()
                    if data == '[DONE]': break
                    item = json.loads(data)
                    if item.get('usage'): usage = item['usage']
                    for choice in item.get('choices') or []:
                        delta = choice.get('delta') or {}
                        txt = ''.join(delta.get(k) or '' for k in ['reasoning_content','reasoning','content'])
                        if txt:
                            now = time.perf_counter()
                            first = first or now
                            last = now
                            chunks.append(now-began)
                            texts.append(txt)
            n = usage.get('completion_tokens', 0)
            assert n == args.tokens, usage
            result = {'usage':usage,'request':body,'elapsed_s':time.perf_counter()-began,'ttft_s':first-began,'decode_s':last-first,'decode_tok_s':n/(last-first),'chunk_timestamps_s':chunks,'text':''.join(texts)}
            (args.out/'request.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print(f'[HB] request_done decode_tok_s={result["decode_tok_s"]:.6f}', flush=True)
            phase[0] = 'probe-finalize'
            code = proc.wait(timeout=45)
            assert code == 0, (code, (args.out/'bpftrace.err').read_text())
            print('CUDA_API_PROFILE_DONE', flush=True)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)
        done.set()

if __name__ == '__main__': main()
