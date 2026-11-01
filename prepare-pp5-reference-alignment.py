"""Stage an immutable reference-based overlay; retain only SSD Engram integration."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

COMMIT = "b3d5e18081285f75eb55e1e7a1c45e209d612e35"
ENGRAM = "vllm/models/deepseek_v4_1/common/engram.py"
RESTORE = [
    "vllm/model_executor/model_loader/utils.py",
    "vllm/models/deepseek_v4_1/nvidia/model.py",
    "vllm/tokenizers/deepseek_v41.py",
    "vllm/v1/worker/gpu/cudagraph_utils.py",
    "vllm/v1/worker/gpu/model_runner.py",
]
SUFFIXES = {".py", ".pyi", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".c"}


def replace_once(text, old, new):
    assert text.count(old) == 1, f"Expected one anchor: {old[:100]!r}"
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive = subprocess.run(["git", "-C", str(args.repo), "archive", COMMIT, "vllm"],
                             check=True, capture_output=True, timeout=30).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        ref = {e.name: tar.extractfile(e).read() for e in tar
               if e.isfile() and Path(e.name).suffix in SUFFIXES}
    current = (args.audit / "runtime" / ENGRAM).read_text(encoding="utf-8")
    original = ref[ENGRAM].decode("utf-8")
    source = replace_once(original, "import weakref\n", "import os\nimport weakref\n")
    current_tree = ast.parse(current)
    loader = next(n for n in current_tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_disk_engram_weight_loader")
    loader_source = ast.get_source_segment(current, loader)
    source = replace_once(source, "@triton.jit\ndef _engram_lookup_kernel",
                          loader_source + "\n\n\n@triton.jit\ndef _engram_lookup_kernel")
    source = replace_once(source,
        "        block_size: int = 32,\n        cpu_offload: bool = False,\n    ):\n",
        "        block_size: int = 32,\n        cpu_offload: bool = False,\n        layer_id: int | None = None,\n    ):\n")
    disk_start = current.index("        self.disk_table = None\n")
    disk_end = current.index("        def _alloc(", disk_start)
    source = replace_once(source, "        def _alloc(shape:",
                          current[disk_start:disk_end] + "        def _alloc(shape:")
    source = replace_once(source, "        weight, scales = self._storage()\n",
        "        vocab_start, vocab_end = self.vocab_start_idx, self.vocab_end_idx\n"
        "        if self.disk_table is not None:\n"
        "            weight, scales, indices = self.disk_table.consume(indices)\n"
        "            vocab_start, vocab_end = 0, weight.shape[0]\n"
        "        else:\n"
        "            weight, scales = self._storage()\n")
    source = replace_once(source,
        "            self.vocab_start_idx,\n            self.vocab_end_idx,\n            rows,\n",
        "            vocab_start,\n            vocab_end,\n            rows,\n")
    source = replace_once(source,
        "            cpu_offload=engram_config.cpu_offload if engram_config else True,\n",
        "            cpu_offload=engram_config.cpu_offload if engram_config else True,\n"
        "            layer_id=layout.layer_ids[layer_hash_index],\n")
    source = replace_once(source,
        "        if self.embed_tokens.cpu_offload and not get_current_vllm_config().parallel_config.enable_dbo:\n",
        "        if (self.embed_tokens.disk_table is None and self.embed_tokens.cpu_offload\n"
        "                and not get_current_vllm_config().parallel_config.enable_dbo):\n")
    source = replace_once(source,
        '        """Gather this layer\'s rows before decoder layers consume them."""\n',
        '        """Gather this layer\'s rows before decoder layers consume them."""\n'
        "        if self.embed_tokens.disk_table is not None:\n"
        "            self.embed_tokens.disk_table.prefetch(\n"
        "                hash_ids, self.embed_tokens.vocab_start_idx,\n"
        "                self.embed_tokens.vocab_end_idx,\n"
        "            )\n"
        "            return\n")
    source = replace_once(source, "        rows = self._ready_rows(hash_ids.shape[0])\n",
        "        if self.embed_tokens.disk_table is not None:\n"
        "            self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])\n"
        "            rows = self.staged_rows[: hash_ids.shape[0]]\n"
        "        else:\n"
        "            rows = self._ready_rows(hash_ids.shape[0])\n")
    output = {p: ref[p] for p in RESTORE}
    output[ENGRAM] = source.encode("utf-8")
    extras = ["vllm/models/deepseek_v4_1/common/disk_engram_mmap.py",
              "vllm/models/deepseek_v4_1/common/engram_mmap.py"]
    output.update({p: (args.audit / "runtime" / p).read_bytes() for p in extras})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in output.items():
        compile(data, name, "exec")
        path = args.out / "aligned" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    patch = "".join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
                    fromfile="reference/" + ENGRAM, tofile="ssd-adapter/" + ENGRAM))
    (args.out / "engram-ssd-only.patch").write_bytes(patch.encode("utf-8"))
    expected = {p: hashlib.sha256(b).hexdigest() for p, b in ref.items()}
    expected[ENGRAM] = hashlib.sha256(output[ENGRAM]).hexdigest()
    runtime = json.loads((args.audit / "runtime-manifest.json").read_text(encoding="utf-8"))["files"]
    extra_hashes = {p: d["sha256"] for p, d in runtime.items() if p not in ref}
    manifest = {"commit": COMMIT, "reference_files": len(ref), "expected": expected,
                "allowed_changed": [ENGRAM], "extra_hashes": extra_hashes,
                "restored_exact": RESTORE, "adapter_files": extras,
                "native_binary_provenance": "Inherited from r3; source-to-binary provenance not established"}
    (args.out / "alignment-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    (args.out / "reference-engram.py").write_bytes(ref[ENGRAM])
    print(json.dumps({"staged_files":len(output), "reference_files":len(ref),
                      "restored_exact":RESTORE, "allowed_changed":[ENGRAM]}))
    print("REFERENCE_OVERLAY_PREPARED")


if __name__ == "__main__":
    main()
