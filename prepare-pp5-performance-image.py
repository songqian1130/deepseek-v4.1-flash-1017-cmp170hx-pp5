"""Prepare a reproducible r8 image and an explicit, reviewed parity manifest."""
import hashlib
import json
from pathlib import Path
import shutil

base=Path(__file__).parent
out=base/'performance-hotspots-20261101/r8-shared-mapping'
for source,target in [(base/'performance-breakdown-20261031/r6-full-build/alignment-manifest.json','alignment-manifest.json'),
                      (base/'verify-pp5-lazy-contract.py','verify-pp5-lazy-contract.py'),
                      (base/'performance-hotspots-20261101/fast-token-mapping-verification.json','fast-token-mapping-verification.json')]:
    shutil.copyfile(source,out/target)
manifest=json.loads((out/'alignment-manifest.json').read_text())
approved={}
for name in ['gpu_worker.py','gpu_model_runner.py']:
    key='vllm/v1/worker/'+name
    approved[key]={'original_sha256':manifest['expected'][key],'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
    manifest['expected'][key]=approved[key]['sha256']
    if key not in manifest['allowed_changed']:manifest['allowed_changed'].append(key)
    if key in manifest['restored_exact']:manifest['restored_exact'].remove(key)
key='vllm/models/deepseek_v4_1/common/pp5_fast_token_mapping.py'
manifest['extra_hashes'][key]=hashlib.sha256((out/'pp5_fast_token_mapping.py').read_bytes()).hexdigest()
manifest['adapter_files'].append(key)
manifest['approved_performance_revision']={'files':approved,'new_helper':key,'quality_constraint':'No weight/quantization/attention math/sequence limits/speculation changes. Preserve output and logprobs.'}
(out/'alignment-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
(out/'Containerfile').write_text('''FROM local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r6-v1-full
COPY . /opt/pp5-performance-source/
RUN cp /opt/pp5-performance-source/gpu_worker.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py && \\
    cp /opt/pp5-performance-source/gpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py && \\
    cp /opt/pp5-performance-source/pp5_fast_token_mapping.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/pp5_fast_token_mapping.py && \\
    python3 /opt/source-alignment/verify-pp5-reference-alignment.py --package /usr/local/lib/python3.12/dist-packages/vllm --manifest /opt/pp5-performance-source/alignment-manifest.json --reference-engram /opt/source-alignment/reference-engram.py --out /opt/pp5-performance-source/verified.json && \\
    python3 /opt/pp5-performance-source/verify-pp5-lazy-contract.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py
ENV VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PP5_LAZY_RECEIVE=1 VLLM_PP5_SHARED_MAPPING=1
LABEL local.deepseek.pp5-performance="lazy-receive-and-shared-device-query-map" local.deepseek.weights="unchanged"
''',encoding='utf-8',newline='\n')
(out/'build.sh').write_text('''#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
START=$(date +%s)
heartbeat() { while sleep 10; do echo "[HB $(( $(date +%s)-START ))s] phase=build-r8"; done; }
heartbeat & HB=$!
trap 'kill "$HB" 2>/dev/null || true; wait "$HB" 2>/dev/null || true' EXIT
echo '[HB 0s] phase=build-r8'
timeout --foreground 240 docker build --network host -f Containerfile -t local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r8-perf .
docker image inspect local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r8-perf > image-inspect.json
echo 'R8_PERFORMANCE_IMAGE_BUILD_DONE'
''',encoding='utf-8',newline='\n')
print('R8_PERFORMANCE_BUILD_CONTEXT_PREPARED')
