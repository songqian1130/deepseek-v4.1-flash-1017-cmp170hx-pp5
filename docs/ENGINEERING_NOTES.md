# Engineering Notes

## Deployment Baseline

- Target: five CMP170HX GPUs with PP5/TP1 and decoder-layer partition `8,8,8,8,8`.
- Keep model experts on GPU. The active r10 configuration uses no CPU expert offload.
- Use the original DeepSeek V4.1 Flash checkpoint and quantization. Do not transfer Qwen-specific quantization, hashing, or scheduling changes from the SSD reference project.
- Runtime limits are configured for a 1M-token model length, 2048 batched tokens, and 8 sequences. These settings do not prove that a 1M-token request fits or completes.
- The PP5 behavior is aligned to `344303947/dsv41-flash-pp5-170hx`, source revision `b3d5e18081285f75eb55e1e7a1c45e209d612e35`. The reference repository's separate eight-GPU TP8 example is not the deployment target.

## Main Scheduling Optimization: Lazy Receive

### Problem

In the pinned reference path, a PP rank starts receiving intermediate tensors before preparing attention metadata. The receive path includes Gloo metadata work. That early operation serialized PP stages: later stages waited at the receive point even though they could prepare their own attention metadata without the upstream activation tensor.

### Changed ordering

For each model step, a non-first PP rank now creates a deferred receiver instead of immediately starting the receive. It prepares the local attention metadata first. When the model runner is about to consume the upstream intermediate tensors, it invokes the same receive path and waits for the tensors as before.

This lets later PP stages do local preparation while upstream stages compute. It does not calculate model layers out of order, remove a transfer, change the tensors, or skip model arithmetic. It moves independent preparation out of the waiting portion of the step.

The optimization is deliberately limited to the V1 runner, PP5/TP1 DeepSeek V4.1 path, with DBO disabled. The implementation is in `prepare-pp5-lazy-receive.py`; the runtime gate is `VLLM_PP5_LAZY_RECEIVE`.

### Performance evidence and status

An unsampled run reported 43.95 and 43.72 tok/s with lazy receive enabled, versus 16.54 tok/s with it disabled. The raw report and full workload metadata are missing, so these numbers are preliminary and should not be presented as a fully reproducible benchmark. See [Results](RESULTS.md).

Scheduling changes should preserve output tokens and model scores, but this must be checked rather than assumed. The existing `verify-pp5-no-degradation.py` harness compares five fixed prompts, returned messages, token identities, and top-token log-probabilities. The committed zero-difference score report compares r10 with r23, not lazy receive enabled with disabled. Lazy-receive numerical parity therefore remains an open acceptance item.

## Other Local Runtime Changes

- **Shared token mapping:** r10 can reuse the GPU-computed token-to-request map across DeepSeek KV-group metadata copies. This is separate from lazy receive and is not established as the cause of the large 16.54-to-43.8 tok/s change.
- **SSD Engram host breaks:** Engram FP8 weights and E8M0 scale tensors stay byte-for-byte in the original safetensors shards. Lookup uses SSD-backed mmap. Explicit layer IDs are required because layers 1 and 14 have matching shapes. The breakable CUDA Graph path places required host work outside captured segments while retaining graph execution around it.
- **Container and host settings:** the service runs in an isolated Docker image. The launcher disables NCCL P2P and IB for this host, which has no useful peer-to-peer path in the tested setup. These are host-specific deployment settings, not portable defaults.

## Engram Storage Details

- Resolve shards through the model index and validate paths against the model root, including checkpoint-cache symlinks.
- Validate representative mmap rows against safetensors slices byte-for-byte before full service startup.
- `chunk_rows=4` can improve parallelism for cold, small row batches, but may add overhead when pages are cached. Compare cold and warm page-cache conditions.
- SSD mmap replaces the large Engram table residency path. It does not offload the MoE experts to CPU.

## Rejected or Unpromoted Candidates

- **r23 asynchronous Engram-ID staging:** passed the bounded quality battery and the r10/r23 top-5 log-probability comparison. Warm serial decode reached about 45.2 tok/s versus about 42.5 tok/s on r10, but mean per-request decode in the measured four-concurrent workload was lower (24.03 versus 27.19 tok/s). It remains stopped and was not promoted.
- **DSpark:** not adopted. Earlier experiments did not establish a viable gain within the available per-GPU memory headroom.
- **Alternate PP partitions and NCCL tuning:** treat as experiments, not production defaults. Candidate layer splits failed in the adapted lazy safetensors load path, and tested P2P/NCCL variations regressed on this host.
- **CPU expert offload, reduced precision, shortened context, or reduced reasoning:** these change deployment requirements and are not acceptable substitutes for the all-GPU expert baseline.

## Benchmark Practice

- Run A/B cases sequentially on the same host; overlapping inference clients invalidate the comparison.
- Keep prompts, token budgets, sampling settings, warmup, concurrency, and request counts fixed.
- Report TTFT, decode interval, completion tokens, and end-to-end time separately. For concurrent tests, report aggregate throughput and per-request decode speed separately.
- Preserve raw response data, exact launch configuration, image ID, and phase/heartbeat logs with every result.
- Do not replace a working baseline until a candidate has passed startup, health, output, numerical, and workload-specific performance checks.
