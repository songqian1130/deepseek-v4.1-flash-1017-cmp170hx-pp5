#!/usr/bin/env python3
"""Install the mmap reader while preserving the PP5 image's custom model code."""

import importlib.util
from pathlib import Path
import shutil


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return source.replace(old, new, 1)


spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("vllm package is unavailable in the pinned PP5 image")

package_dir = Path(next(iter(spec.submodule_search_locations)))
common_dir = package_dir / "models/deepseek_v4_1/common"
engram_path = common_dir / "engram.py"
source = engram_path.read_text(encoding="utf-8")

source = replace_once(
    source,
    "        block_size: int = 32,\n        cpu_offload: bool = False,\n    ):\n",
    "        block_size: int = 32,\n        cpu_offload: bool = False,\n        layer_id: int | None = None,\n    ):\n",
    "ParallelEngramEmbedding constructor",
)
source = replace_once(
    source,
    "            from .disk_engram import DiskEngramRows\n\n            self.disk_table = DiskEngramRows(num_embeddings, dim, block_size)\n",
    "            if layer_id is None:\n                raise ValueError(\"Disk Engram requires an explicit model layer id\")\n"
    "            from .disk_engram_mmap import DiskEngramRows\n\n"
    "            self.disk_table = DiskEngramRows(\n"
    "                num_embeddings, dim, block_size, layer_id=layer_id\n"
    "            )\n",
    "DiskEngramRows import and constructor",
)
source = replace_once(
    source,
    "            tuple(size for order in layout.primes[layer_hash_index] for size in order),\n"
    "            cpu_offload=engram_config.cpu_offload if engram_config else True,\n",
    "            tuple(size for order in layout.primes[layer_hash_index] for size in order),\n"
    "            cpu_offload=engram_config.cpu_offload if engram_config else True,\n"
    "            layer_id=layout.layer_ids[layer_hash_index],\n",
    "Engram layer-id forwarding",
)

shutil.copyfile(Path("/tmp/disk_engram_mmap.py"), common_dir / "disk_engram_mmap.py")
shutil.copyfile(Path("/tmp/engram_mmap.py"), common_dir / "engram_mmap.py")
engram_path.write_text(source, encoding="utf-8")

for relative in (
    "models/deepseek_v4_1/common/engram.py",
    "models/deepseek_v4_1/common/engram_mmap.py",
    "models/deepseek_v4_1/common/disk_engram_mmap.py",
):
    path = package_dir / relative
    compile(path.read_bytes(), str(path), "exec")

print("[HB] phase=install-done layer-id-forwarding=verified reader=mmap")
