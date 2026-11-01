# DeepSeek V4.1 Flash on 5 x CMP170HX

This repository records the PP5 deployment, host-specific adaptations, benchmarks, and correctness checks for DeepSeek V4.1 Flash on five CMP170HX GPUs. It is an engineering workbench for the existing Linux/Docker deployment, not a portable installer.

## Current Deployment

The active baseline is **r10**:

- PP5/TP1 with layer partition `8,8,8,8,8`; all model experts remain on GPU.
- Original DeepSeek V4.1 Flash checkpoint and quantization.
- Engram tables are read from the checkpoint through SSD-backed mmap.
- V1 model runner with `FULL_AND_PIECEWISE` CUDA graphs and the validated host breaks needed around SSD access.
- Configured limits: 1M model length, 2048 batched tokens, 8 sequences, and GPU memory utilization 0.97. These are engine limits, not evidence that a full 1M-token request was tested.
- PP scheduling includes the local lazy-receive optimization and shared token-to-request mapping. The lazy-receive measurements and correctness status are summarized in [Results](docs/RESULTS.md).

**r23** is a stopped experimental Engram-ID staging candidate. Its warmed serial decode was slightly faster than r10, but its measured four-request per-request speed was lower. It was not promoted.

## Key Findings

The largest measured PP scheduling change was delaying the start of intermediate-tensor reception until those tensors are needed. This lets each PP stage prepare its attention metadata while an earlier stage is still computing. It does not remove tensor transfers or model computation. The reported unsampled A/B rates were 43.95 and 43.72 tok/s with lazy receive enabled, versus 16.54 tok/s with it disabled. The raw run artifact and complete workload metadata have not yet been checked into this repository, so treat that A/B as preliminary; see [Results](docs/RESULTS.md).

## Repository Guide

- [docs/ENGINEERING_NOTES.md](docs/ENGINEERING_NOTES.md): topology, local adaptations, design rationale, and operational constraints.
- [docs/RESULTS.md](docs/RESULTS.md): measured performance, correctness evidence, and evidence gaps.
- `benchmarks/2026-11/`: committed r10/r23 serial and concurrent results, plus quality and log-probability summaries.
- Root scripts and patches: image builds, launch profiles, source alignment, profiling, and experiments.
- `tests/`: synthetic Engram mmap tests.
- `include/`: supporting C++ experiments.

## Validation

Run the local synthetic Engram tests with:

```bash
python -m unittest discover -s tests -v
```

`verify-pp5-no-degradation.py` is the targeted scheduling-parity harness. It compares deterministic responses, token identities, and returned top-token log-probabilities. It requires a serving instance exposing the `pp5_perf_configure` collective RPC; the regular r10 launcher does not enable the experimental runtime-control extension. A completed run and its raw output must be archived before claiming lazy-receive numerical parity.

For performance comparisons, keep prompts, generation limits, warmup state, concurrency, and measurement method fixed. Report single-request decode speed separately from end-to-end speed and concurrent aggregate throughput.

## References

- PP5 source alignment: `344303947/dsv41-flash-pp5-170hx`, source revision `b3d5e18081285f75eb55e1e7a1c45e209d612e35`.
- SSD access patterns: `AlexBrandes/cmp170hx-qwen38-flash-next`, source revision `c0db9613d146eda17338b84f2440ead5505f53c5`.
- Qwen-specific hashing, scheduling, and quantization were not transplanted into the DeepSeek model path.

## Deployment Notes

Launchers assume a Linux GPU host, Docker with NVIDIA support, pinned local images, and the official checkpoint at the configured path. Review host paths, ports, bind addresses, and storage locations before using them elsewhere. The service has historically listened on `0.0.0.0` without API authentication; restrict network access before exposing it beyond a trusted host.

Do not commit credentials, private keys, model weights, or raw host logs containing secrets. Keep deployment credentials in environment variables or a local credential manager.
