# Performance and Correctness Results

## Current AIO Deployment (2026-11-02)

The sequential r10 → AIO → r10 comparison measured four-request aggregate decode medians of 107.414 → 113.041 → 107.606 tok/s for repeated fixed-length prompts, and 105.244 → 115.505 → 105.821 tok/s for varied fixed-length prompts. Container CPU time per generated token fell about 22%. Production port 9010 then passed semantic smoke 3/3 and six formal concurrent rounds with median 113.533 tok/s (range 112.992–114.528).

The candidate's 136 formal requests had no SSE chunk gap above one second (maximum 0.550 s), OOM or CPU throttling. The older severe stall coincided with inference-time Triton JIT; this supports a first-shape compilation explanation, not a universal stability guarantee. Keep compiler caches and distinguish warm throughput from the first request using a new shape.

This compares whole deployments on different pinned source revisions, not a single-variable AIO backend substitution. AIO retains lazy receive and shared mapping; their necessity or benefit within AIO was not isolated. Numerical parity for lazy receive on/off remains unverified. See the [full protocol and limitations](AIO_STABILITY_REPORT_20261102.md) and committed raw data in `benchmarks/2026-11/aio-stability/`. The historical r10/r23 measurements below use their original workloads and should not be compared directly with this new protocol.

## Lazy-Receive A/B (Preliminary)

An unsampled PP5 A/B run reported the following single-request decode rates:

| Lazy receive | Run | Decode speed |
|---|---:|---:|
| Enabled | 1 | 43.95 tok/s |
| Enabled | 2 | 43.72 tok/s |
| Disabled | 1 | 16.54 tok/s |

The two enabled runs average 43.84 tok/s, about 2.65 times the recorded disabled run. The result is consistent with PP metadata preparation previously sitting behind an early receive operation. The original raw output, prompts, generation limits, warmup state, and exact timing command are not currently archived here. Until those are added, this is a reported preliminary result rather than a reproducible benchmark result. It does not establish output or numerical equivalence; that is a separate acceptance check below.

## r10 and r23 Serial Requests

These runs used three serial prompts per round on the same five-GPU host. Decode speed is generated completion tokens divided by the client-observed decode interval.

| Version | Startup round mean | Warm rounds | 9-request mean |
|---|---:|---:|---:|
| r10 | 42.53 tok/s | 42.52 tok/s | 42.515 tok/s |
| r23 | 28.70 tok/s | 45.17 / 45.18 tok/s | 39.680 tok/s including its slow first round |

r23's warmed serial rounds were about 6.3% faster than r10. Its first three requests after startup were substantially slower.

## Four Concurrent Requests

Four synchronized prompts were run together. Aggregate throughput and per-request decode speed are different measurements and are shown separately.

| Version | First-round aggregate | Warm-round aggregate median | Warm mean per-request decode |
|---|---:|---:|---:|
| r10 | 90.75 tok/s | 95.44 tok/s | 27.19 tok/s |
| r23 | 56.48 tok/s | 83.28 tok/s | 24.03 tok/s |

Across the four warm rounds, r23 aggregate decode throughput averaged 87.56 tok/s versus 96.38 tok/s for r10. Mean per-request decode speed was about 11.6% lower on r23. Token counts vary by prompt and stop condition; use the complete round JSON when interpreting these figures.

## Correctness Evidence

- r10 and r23 each passed the bounded quality battery 8/8. It included executable median-function cases, multi-turn recall, a 4729-token retrieval prompt, and two concurrent requests.
- Five fixed prompts produced 1920 generation/top-5 log-probability records when comparing r10 with r23. Messages and tokens matched, and the maximum log-probability absolute difference was 0.
- This r10/r23 comparison does **not** isolate lazy receive on versus off.
- `verify-pp5-no-degradation.py` is intended to compare lazy-receive modes on five fixed reasoning, code, and math prompts. It checks full returned messages, token identities, and top-token log-probabilities for the first 64 generated tokens; its pass condition is a maximum absolute log-probability difference no greater than `1e-5`.
- A completed lazy-receive on/off run and its raw responses are not yet archived in this repository. Numerical parity for this scheduling change remains unverified here.
- Neither the 50 tok/s target nor a real 1M-token request has been validated.

## Reproducibility

Committed r10/r23 raw reports are in `benchmarks/2026-11/`. The lazy-receive A/B figures above were reported during testing, but their raw report is missing. A reproducible follow-up should save the exact prompts and token limits, model/image IDs, launch arguments, warmup state, per-request token counts, timing output, and lazy-receive mode for each run.
