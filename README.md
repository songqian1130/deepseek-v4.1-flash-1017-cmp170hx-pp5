# DeepSeek V4.1 Flash PP5 on CMP170HX

This repository contains the five-GPU PP5 deployment workbench for DeepSeek V4.1 Flash: the SSD-backed Engram adapter, isolated image/build launchers, correctness checks, benchmark tools, and the findings collected while adapting the reference deployment to a 5 x CMP170HX host.

## Current Deployment

The last verified active service is **r10**. It uses PP5/TP1 with layers `8,8,8,8,8`, the original model weights and quantization, all experts resident on GPU, SSD mmap for Engram, V1 runner, and `FULL_AND_PIECEWISE` graphs with the validated SSD host breaks. The runtime profile retains a 1M-token configured context, 2048 batched-token limit, and 8 sequences. These are configuration limits; a full 1M-token request has not been validated.

The retained **r23** image/container is an experimental asynchronous Engram-ID staging candidate. It is stopped and was not promoted: it improves warmed single-request decode but loses per-request speed in the measured 4-concurrent workload. See [the measurements](docs/RESULTS.md).

## Repository Contents

- Root scripts and patches: isolated PP5 builds, launch profiles, Engram adapters, source-alignment checks, and prior experiments.
- `tests/`: synthetic safetensors/mmap behavior tests.
- `benchmarks/2026-11/`: matched r10/r23 serial and 4-concurrent results, plus candidate correctness and logprob summaries.
- `docs/ENGINEERING_NOTES.md`: implementation decisions, verified bottlenecks, tradeoffs, and rejected directions.
- `include/`: supporting C++ experiments.

This is a deployment/research workbench, not a portable one-command installer. Launchers assume a Linux GPU host, Docker with NVIDIA support, the pinned local images, and an official checkpoint mounted at the configured path. Review and adapt host paths, ports, bind addresses, and storage locations before running them elsewhere. The service has historically been launched without API authentication; do not expose it to untrusted networks.

## Validation

Run the synthetic Engram tests with:

```bash
python -m unittest discover -s tests -v
```

The benchmark scripts stream real completions and report decode speed separately from end-to-end speed. For 4-concurrent runs, aggregate throughput is reported separately from mean per-request decode speed. Compare matched prompts, warmup state, and request counts; do not compare an aggregate concurrent rate to a single-request rate.

## Reference Provenance

- PP5 behavior was aligned to `344303947/dsv41-flash-pp5-170hx`, pinned at `b3d5e18081285f75eb55e1e7a1c45e209d612e35`.
- SSD access patterns were informed by `AlexBrandes/cmp170hx-qwen38-flash-next`, pinned at `c0db9613d146eda17338b84f2440ead5505f53c5`.
- Qwen-specific hashing, scheduling, and quantization were not transplanted into the DeepSeek model path.

No credentials or private SSH keys belong in this repository. Deployment scripts should read credentials from environment variables or a local credential manager.
