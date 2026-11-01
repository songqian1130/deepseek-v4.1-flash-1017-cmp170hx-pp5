#!/usr/bin/env python3
"""Apply the pinned Engram mmap diff inside the PP5 runtime image."""

import importlib.util
import subprocess
from pathlib import Path


spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("vllm package is unavailable in the pinned PP5 image")

package_dir = Path(next(iter(spec.submodule_search_locations)))
install_root = package_dir.parent
patch = Path("/tmp/pp5-engram-mmap.patch")
subprocess.run(
    ["patch", "--dry-run", "--batch", "-p1", "-i", str(patch)],
    cwd=install_root,
    check=True,
)
subprocess.run(
    ["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
    cwd=install_root,
    check=True,
)

for relative in (
    "model_executor/model_loader/weight_utils.py",
    "models/deepseek_v4_1/common/engram.py",
    "models/deepseek_v4_1/common/engram_mmap.py",
    "models/deepseek_v4_1/nvidia/model.py",
):
    source = package_dir / relative
    compile(source.read_bytes(), str(source), "exec")
