#!/usr/bin/env python3
"""Compare a few mmap rows with safetensors without loading whole tables."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sysconfig
import time

import numpy as np
import torch
from safetensors import safe_open

_HELPER_PATH = (
    Path(sysconfig.get_path("platlib"))
    / "vllm/models/deepseek_v4_1/common/engram_mmap.py"
)
_SPEC = importlib.util.spec_from_file_location("ds41_engram_mmap", _HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Engram mmap helper is unavailable: {_HELPER_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
MmapEngramTable = _MODULE.MmapEngramTable
_read_header = _MODULE._read_header


def read_rows(path: str, name: str, ids: np.ndarray) -> np.ndarray:
    rows = []
    with safe_open(path, framework="pt", device="cpu") as tensors:
        table = tensors.get_slice(name)
        for row in ids:
            value = table[int(row) : int(row) + 1].contiguous()
            rows.append(value.view(torch.uint8).numpy()[0])
    return np.stack(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    started = time.monotonic()
    report = []
    for layer_id in (1, 14):
        weight_name = f"layers.{layer_id}.engram.embed.weight"
        scale_name = f"layers.{layer_id}.engram.embed.scale"
        weight_path = os.path.realpath(os.path.join(args.model_path, weight_map[weight_name]))
        scale_path = os.path.realpath(os.path.join(args.model_path, weight_map[scale_name]))
        weight_header, _ = _read_header(weight_path)
        scale_header, _ = _read_header(scale_path)
        weight_meta = weight_header[weight_name]
        scale_meta = scale_header[scale_name]
        rows, dim = weight_meta["shape"]
        scale_rows, scale_cols = scale_meta["shape"]
        if rows != scale_rows or weight_meta["dtype"] != "F8_E4M3" or scale_meta["dtype"] != "F8_E8M0":
            raise ValueError(f"Unexpected Engram metadata for layer {layer_id}")

        rng = np.random.default_rng(20261031 + layer_id)
        random_ids = rng.choice(rows, size=max(0, args.sample_count - 3), replace=False)
        ids = np.unique(np.concatenate(([0, rows // 2, rows - 1], random_ids))).astype(np.int64)
        table = MmapEngramTable(
            model_path=args.model_path,
            layer_id=layer_id,
            rows=rows,
            dim=dim,
            scale_cols=scale_cols,
            start_row=0,
            local_rows=rows,
            workers=args.workers,
            chunk_rows=64,
        )
        try:
            mmap_weights = np.empty((ids.size, dim), dtype=np.uint8)
            mmap_scales = np.empty((ids.size, scale_cols), dtype=np.uint8)
            table.gather_into(ids, mmap_weights, mmap_scales)
            disk_weights = read_rows(weight_path, weight_name, ids)
            disk_scales = read_rows(scale_path, scale_name, ids)
            np.testing.assert_array_equal(mmap_weights, disk_weights)
            np.testing.assert_array_equal(mmap_scales, disk_scales)
        finally:
            table.pool.shutdown(wait=True)

        report.append(
            {
                "layer": layer_id,
                "rows": int(rows),
                "dim": int(dim),
                "sample_rows": ids.tolist(),
                "weight_shard": os.path.basename(weight_path),
                "scale_shard": os.path.basename(scale_path),
                "weight_dtype": weight_meta["dtype"],
                "scale_dtype": scale_meta["dtype"],
                "byte_equal": True,
            }
        )
        print(f"[HB {int(time.monotonic()-started)}s] phase=layer-verified layer={layer_id}", flush=True)

    print(json.dumps({"ok": True, "layers": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
