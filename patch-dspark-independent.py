from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/utils.py')
text = path.read_text()
old_import = 'import torch.nn as nn\n\nfrom vllm.config import ModelConfig, VllmConfig, replace\n'
new_import = 'import os\nimport torch.nn as nn\n\nfrom vllm.config import ModelConfig, VllmConfig, replace\n'
if old_import not in text:
    raise SystemExit('DSpark utils import anchor not found')
text = text.replace(old_import, new_import, 1)
old = '''    # The target model has already consumed the process-global offloader budget.\n    # DSpark is a second model owned by the last PP rank, so give its layer stack\n    # an independent budget before construction. Without this reset the draft\n    # layers are all materialized on the already-full GPU before their weights\n    # can be wrapped.\n    set_offloader(create_offloader(draft_vllm_config.offload_config))\n'''
new = '''    # Permit DSpark-only UVA offload while leaving the target model's\n    # offload_config untouched. This is intentionally opt-in.\n    dspark_cpu_gb = float(os.environ.get("VLLM_DSPARK_CPU_OFFLOAD_GB", "0"))\n    if dspark_cpu_gb > 0:\n        draft_vllm_config.offload_config = replace(\n            draft_vllm_config.offload_config,\n            offload_backend="uva",\n            uva=replace(\n                draft_vllm_config.offload_config.uva,\n                cpu_offload_gb=dspark_cpu_gb,\n                cpu_offload_params=set(),\n            ),\n        )\n\n    # The target model has already consumed the process-global offloader budget.\n    # DSpark is a second model owned by the last PP rank, so give its layer stack\n    # an independent budget before construction. Without this reset the draft\n    # layers are all materialized on the already-full GPU before their weights\n    # can be wrapped.\n    set_offloader(create_offloader(draft_vllm_config.offload_config))\n'''
if old not in text:
    raise SystemExit('DSpark utils offloader anchor not found')
path.write_text(text.replace(old, new, 1))
print('DSPARK_INDEPENDENT_OFFLOAD_PATCHED')
