"""Read DeepSeek Engram safetensors rows through the Linux page cache."""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger("vllm.engram_mmap")


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        size_prefix = f.read(8)
        if len(size_prefix) != 8:
            raise ValueError(f"truncated safetensors header prefix: {path}")
        header_size = struct.unpack("<Q", size_prefix)[0]
        file_size = os.fstat(f.fileno()).st_size
        if header_size > file_size - 8:
            raise ValueError(f"safetensors header exceeds file size: {path}")
        header_bytes = f.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
        header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")
    header.pop("__metadata__", None)
    return header, 8 + header_size


class MmapEngramTable:
    """Map one rank-local row range without pinning the backing table."""

    def __init__(
        self,
        model_path: str,
        layer_id: int,
        rows: int,
        dim: int,
        scale_cols: int,
        start_row: int,
        local_rows: int,
        workers: int = 16,
        chunk_rows: int = 1024,
    ) -> None:
        model_path = os.path.realpath(model_path)
        if not os.path.isdir(model_path):
            raise ValueError(f"Engram mmap requires a local model directory: {model_path}")
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        prefix = f"layers.{layer_id}.engram.embed"
        names = {"weight": f"{prefix}.weight", "scale": f"{prefix}.scale"}
        entries = {}
        for kind, name in names.items():
            filename = weight_map.get(name)
            if filename is None:
                raise KeyError(f"Engram mmap index is missing {name}")
            index_path = os.path.normpath(filename)
            if os.path.isabs(index_path):
                index_path = os.path.basename(index_path)
            indexed_path = os.path.abspath(os.path.join(model_path, index_path))
            if os.path.commonpath((model_path, indexed_path)) != model_path:
                raise ValueError(f"Engram shard escapes model directory: {filename}")
            if not os.path.isfile(indexed_path):
                raise FileNotFoundError(f"Engram safetensors shard is missing: {indexed_path}")
            # Hugging Face cache entries may be symlinks to the actual shard.
            path = os.path.realpath(indexed_path)
            header, data_start = _read_header(path)
            meta = header.get(name)
            if meta is None:
                raise KeyError(f"Engram mmap shard header is missing {name}: {path}")
            if not isinstance(meta, dict):
                raise ValueError(f"invalid safetensors metadata for {name}: {path}")
            entries[kind] = (path, data_start, meta)

        weight_path, weight_data_start, weight_meta = entries["weight"]
        scale_path, scale_data_start, scale_meta = entries["scale"]
        if weight_meta["dtype"] != "F8_E4M3" or scale_meta["dtype"] != "F8_E8M0":
            raise ValueError(
                "Engram mmap expects F8_E4M3 weights and F8_E8M0 scales, got "
                f"{weight_meta['dtype']} and {scale_meta['dtype']}"
            )
        if weight_meta["shape"] != [rows, dim]:
            raise ValueError(
                f"Engram layer {layer_id} weight shape {weight_meta['shape']} "
                f"does not match {(rows, dim)}"
            )
        if scale_meta["shape"] != [rows, scale_cols]:
            raise ValueError(
                f"Engram layer {layer_id} scale shape {scale_meta['shape']} "
                f"does not match {(rows, scale_cols)}"
            )
        if start_row < 0 or local_rows < 1 or start_row + local_rows > rows:
            raise ValueError(
                f"Engram layer {layer_id} local row range "
                f"[{start_row}, {start_row + local_rows}) exceeds {rows}"
            )

        weight_begin, weight_end = weight_meta["data_offsets"]
        scale_begin, scale_end = scale_meta["data_offsets"]
        if weight_end - weight_begin != rows * dim or scale_end - scale_begin != rows * scale_cols:
            raise ValueError(f"Engram layer {layer_id} has invalid safetensors offsets")
        for path, data_start, begin, end in (
            (weight_path, weight_data_start, weight_begin, weight_end),
            (scale_path, scale_data_start, scale_begin, scale_end),
        ):
            file_size = os.path.getsize(path)
            if begin < 0 or end < begin or data_start + end > file_size:
                raise ValueError(f"Engram safetensors data range is invalid: {path}")

        self.rows = local_rows
        self.dim = dim
        self.scale_cols = scale_cols
        self.weight = np.memmap(
            weight_path,
            dtype=np.uint8,
            mode="r",
            offset=weight_data_start + weight_begin + start_row * dim,
            shape=(local_rows, dim),
        )
        self.scales = np.memmap(
            scale_path,
            dtype=np.uint8,
            mode="r",
            offset=scale_data_start + scale_begin + start_row * scale_cols,
            shape=(local_rows, scale_cols),
        )
        self._advise_random(self.weight)
        self._advise_random(self.scales)
        self.chunk_rows = max(1, int(chunk_rows))
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        logger.info(
            "Engram mmap: layer=%d rows=%d..%d dim=%d weight=%s scale=%s",
            layer_id,
            start_row,
            start_row + local_rows,
            dim,
            weight_path,
            scale_path,
        )

    @staticmethod
    def _advise_random(array: np.memmap) -> None:
        mapping = getattr(array, "_mmap", None)
        if mapping is None or not hasattr(mmap, "MADV_RANDOM"):
            return
        try:
            mapping.madvise(mmap.MADV_RANDOM)
        except (OSError, ValueError):
            logger.warning("Engram mmap MADV_RANDOM failed; keeping kernel defaults")

    def gather_into(
        self,
        ids: np.ndarray,
        weight_out: np.ndarray,
        scale_out: np.ndarray,
    ) -> None:
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        count = ids.size
        if weight_out.shape[0] < count or weight_out.shape[1:] != (self.dim,):
            raise ValueError(f"Engram weight output shape {weight_out.shape} for {count} rows")
        if scale_out.shape[0] < count or scale_out.shape[1:] != (self.scale_cols,):
            raise ValueError(f"Engram scale output shape {scale_out.shape} for {count} rows")
        if count == 0:
            return
        if ids.min() < 0 or ids.max() >= self.rows:
            raise IndexError(f"Engram local row ids exceed [0, {self.rows})")

        unique, inverse = np.unique(ids, return_inverse=True)
        unique_weight = np.empty((unique.size, self.dim), dtype=np.uint8)
        unique_scale = np.empty((unique.size, self.scale_cols), dtype=np.uint8)
        ranges = [
            (start, min(start + self.chunk_rows, unique.size))
            for start in range(0, unique.size, self.chunk_rows)
        ]

        def read_rows(bounds: tuple[int, int]) -> None:
            start, end = bounds
            rows = unique[start:end]
            unique_weight[start:end] = self.weight[rows]
            unique_scale[start:end] = self.scales[rows]

        if len(ranges) <= 1:
            read_rows(ranges[0])
        else:
            for _ in self.pool.map(read_rows, ranges):
                pass
        weight_out[:count] = unique_weight[inverse]
        scale_out[:count] = unique_scale[inverse]


__all__ = ["MmapEngramTable"]
