"""Safetensors mmap backend with asynchronous hash-index staging."""

from __future__ import annotations

import logging
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import wraps
from pathlib import Path

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    _weak_ref_capture_arg,
    is_breakable_cudagraph_enabled,
)
from vllm.config import get_current_vllm_config
from vllm.models.deepseek_v4_1.common.engram_mmap import MmapEngramTable, _read_header

logger = logging.getLogger("vllm.disk_engram_mmap")


def host_break_during_capture(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            return fn(*args, **kwargs)
        weak_args = tuple(_weak_ref_capture_arg(arg) for arg in args)
        weak_kwargs = {key: _weak_ref_capture_arg(value) for key, value in kwargs.items()}
        return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))
    return wrapped


class DiskEngramRows:
    def __init__(self, num_embeddings, dim, block_size, layer_id, manifest=None):
        index_path = Path(manifest or os.environ["DS41_ENGRAM_MANIFEST"]).resolve()
        if index_path.is_dir():
            index_path = index_path / "model.safetensors.index.json"
        if index_path.name != "model.safetensors.index.json":
            raise ValueError("DS41_ENGRAM_MANIFEST must name the model safetensors index")
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        weight_name = f"layers.{layer_id}.engram.embed.weight"
        weight_file = weight_map.get(weight_name)
        if weight_file is None:
            raise KeyError(f"Safetensors index is missing {weight_name}")
        weight_path = (index_path.parent / weight_file).resolve()
        weight_header, _ = _read_header(str(weight_path))
        weight_meta = weight_header.get(weight_name)
        if weight_meta is None or weight_meta.get("shape") != [num_embeddings, dim]:
            raise ValueError(f"Engram layer {layer_id} weight shape mismatch")
        scale_name = weight_name.removesuffix("weight") + "scale"
        scale_file = weight_map.get(scale_name)
        if scale_file is None:
            raise KeyError(f"Safetensors index is missing {scale_name}")
        scale_path = (index_path.parent / scale_file).resolve()
        scale_header, _ = _read_header(str(scale_path))
        scale_meta = scale_header.get(scale_name)
        if weight_meta["dtype"] != "F8_E4M3" or scale_meta is None:
            raise ValueError("Engram mmap expects the official FP8 weight and scale tensors")
        if scale_meta["dtype"] != "F8_E8M0" or scale_meta["shape"] != [num_embeddings, dim // block_size]:
            raise ValueError("Engram mmap scale shape or dtype mismatch")
        runtime_config = get_current_vllm_config()
        graph_mode = runtime_config.compilation_config.cudagraph_mode
        uses_graphs = graph_mode.has_full_cudagraphs() or graph_mode.has_piecewise_cudagraphs()
        if uses_graphs and not is_breakable_cudagraph_enabled():
            raise ValueError("Disk Engram mmap graphs require VLLM_USE_BREAKABLE_CUDAGRAPH=1")
        if uses_graphs and not torch.cuda.is_available():
            raise ValueError("Disk Engram mmap graph execution requires CUDA")
        if runtime_config.parallel_config.enable_dbo:
            raise ValueError("Disk Engram mmap currently requires DBO to be disabled")
        self.rows = num_embeddings
        self.dim = dim
        self.scale_dim = dim // block_size
        self.max_tokens = int(runtime_config.scheduler_config.max_num_batched_tokens)
        self.max_heads = None
        self.max_rows = None
        self.table = MmapEngramTable(
            model_path=str(index_path.parent), layer_id=layer_id, rows=num_embeddings,
            dim=dim, scale_cols=self.scale_dim, start_row=0, local_rows=num_embeddings,
            workers=int(os.environ.get("VLLM_ENGRAM_MMAP_WORKERS", "16")),
            chunk_rows=int(os.environ.get("VLLM_ENGRAM_MMAP_CHUNK_ROWS", "1024")),
        )
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-mmap")
        self.pending: tuple[Future, int, int, tuple[int, ...]] | None = None
        self.host_weights = self.host_scales = self.host_compact = self.host_ids = None
        self.device_weights = self.device_scales = self.device_compact = None
        self.device = None
        self.h2d_event = None
        self.read_calls = self.read_rows = 0
        self.read_seconds = 0.0
        logger.info("Engram mmap async ids: layer=%d rows=%d dim=%d", layer_id, num_embeddings, dim)

    def _allocate_host_buffers(self, heads: int) -> None:
        if heads < 1 or self.max_tokens < 1:
            raise ValueError("Engram mmap requires positive head and token capacities")
        self.max_heads = heads
        self.max_rows = self.max_tokens * heads
        self.host_ids = torch.empty((self.max_tokens, heads), dtype=torch.int64, pin_memory=True)
        self.host_weights = torch.empty((self.max_rows, self.dim), dtype=torch.uint8, pin_memory=True)
        self.host_scales = torch.empty((self.max_rows, self.scale_dim), dtype=torch.uint8, pin_memory=True)
        self.host_compact = torch.empty((self.max_tokens, heads), dtype=torch.int32, pin_memory=True)

    def _allocate_device_buffers(self, device: torch.device) -> None:
        if self.device is not None:
            if self.device != device:
                raise RuntimeError(f"Engram mmap device changed from {self.device} to {device}")
            return
        self.device = device
        self.device_weights = torch.empty((self.max_rows, self.dim), dtype=torch.uint8, device=device)
        self.device_scales = torch.empty((self.max_rows, self.scale_dim), dtype=torch.uint8, device=device)
        self.device_compact = torch.empty((self.max_tokens, self.max_heads), dtype=torch.int32, device=device)

    def _read_from_staged_ids(self, tokens: int, heads: int, vocab_start: int, vocab_end: int) -> int:
        ids = self.host_ids.numpy()[:tokens, :heads]
        valid = (ids >= vocab_start) & (ids < vocab_end)
        unique, inverse = np.unique(ids[valid], return_inverse=True)
        compact = np.full(ids.shape, -1, dtype=np.int32)
        compact[valid] = inverse
        np.copyto(self.host_compact.numpy()[:tokens, :heads], compact)
        started = time.monotonic()
        self.table.gather_into(unique, self.host_weights.numpy()[:unique.size], self.host_scales.numpy()[:unique.size])
        self.read_calls += 1
        self.read_rows += unique.size
        self.read_seconds += time.monotonic() - started
        return int(unique.size)

    @host_break_during_capture
    def prefetch(self, indices: torch.Tensor, vocab_start: int, vocab_end: int) -> None:
        if self.pending is not None:
            raise RuntimeError("Previous Engram row job has not been consumed")
        if indices.shape[0] > self.max_tokens:
            raise ValueError("Engram mmap input exceeds configured token capacity")
        if self.h2d_event is not None:
            self.h2d_event.synchronize()
        heads = indices.shape[1]
        if self.max_heads is None:
            self._allocate_host_buffers(heads)
        elif heads != self.max_heads:
            raise ValueError(f"Engram mmap head count changed from {self.max_heads} to {heads}")
        tokens = int(indices.shape[0])
        self.host_ids[:tokens, :heads].copy_(indices, non_blocking=True)
        ids_ready = torch.cuda.Event()
        ids_ready.record(torch.cuda.current_stream(indices.device))
        future = self.executor.submit(self._read_after_event, ids_ready, tokens, heads, vocab_start, vocab_end)
        self.pending = (future, 0, indices.data_ptr(), tuple(indices.shape))

    def _read_after_event(self, ids_ready: torch.cuda.Event, tokens: int, heads: int, vocab_start: int, vocab_end: int) -> int:
        ids_ready.synchronize()
        return self._read_from_staged_ids(tokens, heads, vocab_start, vocab_end)

    @host_break_during_capture
    def consume(self, indices: torch.Tensor):
        if self.pending is None:
            raise RuntimeError("Disk Engram rows were not prefetched")
        future, _, pointer, shape = self.pending
        if pointer != indices.data_ptr() or shape != tuple(indices.shape):
            raise RuntimeError("Disk Engram row job belongs to a different input")
        unique_count = future.result(timeout=60)
        self._allocate_device_buffers(indices.device)
        if unique_count:
            self.device_weights[:unique_count].copy_(self.host_weights[:unique_count], non_blocking=True)
            self.device_scales[:unique_count].copy_(self.host_scales[:unique_count], non_blocking=True)
        tokens, heads = indices.shape
        self.device_compact[:tokens, :heads].copy_(self.host_compact[:tokens, :heads], non_blocking=True)
        self.h2d_event = torch.cuda.Event()
        self.h2d_event.record(torch.cuda.current_stream(indices.device))
        self.pending = None
        return self.device_weights, self.device_scales, self.device_compact[:tokens, :heads]
