# Verified Results

Results below were measured on the same 5-GPU host with the same three serial prompts or four synchronized concurrent prompts. Decode tok/s uses generated completion tokens divided by the client-observed decode interval. Concurrent aggregate throughput and single-request speed are different metrics.

## Serial Requests

| Version | Startup round mean | Warm rounds | 9-request mean |
|---|---:|---:|---:|
| r10 | 42.53 tok/s | 42.52 tok/s | 42.515 tok/s |
| r23 | 28.70 tok/s | 45.17 / 45.18 tok/s | 39.680 tok/s including its slow first round |

r23's warmed serial rounds were about 6.3% faster than r10. Its first three requests after startup were substantially slower.

## Four Concurrent Requests

| Version | First-round aggregate | Warm-round aggregate median | Warm mean per-request decode |
|---|---:|---:|---:|
| r10 | 90.75 tok/s | 95.44 tok/s | 27.19 tok/s |
| r23 | 56.48 tok/s | 83.28 tok/s | 24.03 tok/s |

For the four warm rounds, r23 aggregate decode throughput averaged 87.56 tok/s versus 96.38 tok/s for r10. Mean per-request decode speed was about 11.6% lower on r23. Token counts vary by prompt and stop condition, so per-request decode and the full per-round JSON should be considered alongside aggregate throughput.

## Correctness

- r10 and r23 each passed the bounded quality battery 8/8, including executable median-function cases, multi-turn recall, a 4729-token retrieval prompt, and two concurrent requests.
- Five fixed prompts produced 1920 generation/top-5 logprob records. Messages and tokens matched, and the maximum logprob absolute difference was 0.
- These checks establish parity for the tested inputs; they are not an exhaustive model-quality evaluation.
- Neither the 50 tok/s target nor a real 1M-token request has been validated.

Raw reports are in `benchmarks/2026-11/`.
