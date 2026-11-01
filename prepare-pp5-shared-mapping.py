"""Share the existing per-step request mapping across DeepSeek KV groups."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
raw=a.source.read_bytes();manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
# Accept only the source hash covered by the fixed reference audit.
def hashes(node):
    if isinstance(node,dict):
        for k,v in node.items():
            if k.endswith('v1/worker/gpu_model_runner.py'):
                if isinstance(v,str):yield v
                elif isinstance(v,dict):yield v.get('sha256');yield v.get('lf_sha256')
            yield from hashes(v)
    elif isinstance(node,list):
        for x in node:yield from hashes(x)
assert hashlib.sha256(raw).hexdigest() in set(hashes(manifest))
s=raw.decode();original=s
assert s.count('import itertools\n')==1
s=s.replace('import itertools\n','import itertools\nimport os\n',1)
needle='''        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )'''
replacement=needle+'''
        # A per-step token/request map is independent of a KV group's block
        # table. Give shallow CommonAttentionMetadata copies a shared cache.
        self._pp5_shared_token_req_map = torch.empty(
            self.max_num_tokens, dtype=torch.int32, device=self.device
        )'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
needle='''        # Cache attention metadata builds across hybrid KV-cache groups'''
replacement='''        if (
            self.model_config.hf_config.model_type == "deepseek_v41"
            and ubatch_slices is None
            and os.environ.get("VLLM_PP5_SHARED_MAPPING", "0") == "1"
        ):
            # Compute from the authoritative GPU query boundaries exactly once.
            # copy(cm_base) below retains the cache view; every consumer reads
            # the same stable buffer. Other models/microbatch paths are intact.
            from vllm.models.deepseek_v4_1.common.pp5_fast_token_mapping import (
                token_to_req_indices as build_shared_mapping,
            )
            build_shared_mapping(cm_base, self._pp5_shared_token_req_map)

'''+needle
assert s.count(needle)==1;s=s.replace(needle,replacement);ast.parse(s)
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'gpu_model_runner.py').write_text(s,encoding='utf-8',newline='\n')
(a.out/'shared-mapping.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/vllm/v1/worker/gpu_model_runner.py',tofile='b/vllm/v1/worker/gpu_model_runner.py')),encoding='utf-8',newline='\n')
(a.out/'shared-mapping-manifest.json').write_text(json.dumps({'original_sha256':hashlib.sha256(raw).hexdigest(),'candidate_sha256':hashlib.sha256(s.encode()).hexdigest(),'scope':'Reuse original GPU query mapping across shallow per-KV-group metadata copies; no change to attention math or KV indexing.'},indent=2))
print('PP5_SHARED_MAPPING_CANDIDATE_PREPARED')
