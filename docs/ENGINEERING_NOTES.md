# Engineering Notes

## Topology and Fidelity

- The deployment target is the reference repository's five-GPU PP5 topology: TP=1 and layer split `8,8,8,8,8`.
- Keep all experts resident on GPU. CPU expert offload, lower precision, shorter context, reduced reasoning, and skipped layers are not acceptable performance shortcuts for this deployment.
- The top-level reference's eight-GPU TP8 example is a different topology. Do not transplant it into this five-GPU setup.
- Pin and inspect source commits. An image label alone does not prove that installed files match the reference. The audited PP5 source revision is `b3d5e18081285f75eb55e1e7a1c45e209d612e35`.

## Engram on SSD

- Engram's FP8 weights and E8M0 scale tensors remain byte-for-byte in the original safetensors shards; only row lookup changes to mmap-backed reads.
- Layers 1 and 14 have matching tensor shapes. Select tables by explicit layer ID rather than shape alone.
- Resolve shard paths relative to the model index, validate them against the model root, and account for model-cache symlinks used by the checkpoint.
- Validate representative rows against the safetensors slice byte-for-byte before launching a full service.
- `chunk_rows=4` improves parallelism for cold, small row batches but can add overhead when pages are hot. Measure both conditions; do not infer SSD cost from a single cached run.

## PP Scheduling Findings

- A blocking Gloo metadata receive before attention-metadata preparation serialized PP ranks that could otherwise prepare independently. The lazy-receive change delayed the same receive until the intermediate tensors were consumed; it did not remove communication or model computation.
- Sharing token/request boundary mappings reduced duplicate GPU-side preparation. Preserve the GPU-computed boundaries as authoritative and test graph capture/replay when changing this path.
- Rank CPU time and GPU utilization were imbalanced, but arbitrary PP repartitioning failed in the adapted lazy safetensors load path. Do not assume a nominally balanced layer count is supported without a complete load and correctness test.
- P2P/NCCL variants tested on this host regressed. The host reports no useful GPU peer-to-peer path, so changing NCCL algorithms is not a default optimization.

## Candidate Decision

- r23 adds pinned host ID staging and asynchronous GPU-to-host index transfer before CPU unique/mmap gather. The weights, row values, quantization, layers, and reasoning settings are unchanged.
- r23 passed the bounded quality battery and exact top-5 logprob comparison. Warm single-request decode reached about 45.2 tok/s versus about 42.5 tok/s on r10.
- In five 4-concurrent rounds, r23's warmed per-request decode averaged 24.03 tok/s versus 27.19 tok/s on r10. It remains a stopped candidate; r10 is the active deployment.
- The first serial round after r23 startup averaged 28.70 tok/s, then stabilized near 45.2 tok/s. Record startup state and warmup separately.

## Benchmark Discipline

- Run A/B workloads sequentially on the same host; overlapping clients invalidate the comparison.
- Keep prompts, token limits, sampling parameters, concurrency, and request count fixed. Record TTFT, decode duration, completion tokens, and end-to-end time separately.
- For concurrent traffic, report aggregate throughput and per-request decode speed. Aggregate rates can rise or fall with output length and are not a substitute for request latency.
- Separate cold page-cache/startup behavior from steady state. A service restart may reload model weights and rebuild runtime state; the first few requests can differ from later ones.
- Long host tasks need finite deadlines, visible heartbeats, and an explicit recovery path. Never replace a running baseline until the candidate can be stopped and the baseline health-checked.

## Tested Directions to Revisit Carefully

- DSpark was not adopted: this V1 setup has limited per-GPU headroom, and earlier DSpark experiments did not establish a viable performance gain.
- CPU expert residency, changing precision, and reducing context were excluded because they alter the deployment requirements.
- More PP layer repartitioning and P2P tuning are low-priority until the relevant unsupported paths or hardware limitations change.
- Async Engram staging is retained for future controlled testing, but its concurrency regression must be addressed before promotion.
