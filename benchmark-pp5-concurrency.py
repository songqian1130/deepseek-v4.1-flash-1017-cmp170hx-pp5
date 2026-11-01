"""Measure synchronized concurrent requests and aggregate decode throughput."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request

PROMPTS = [
    "用中文解释为什么昼夜交替发生，并给出一个生活中的例子。",
    "Explain in English how a hash table provides average constant-time lookup, including one caveat.",
    "Write a Python function named clamp that takes x, low, and high and returns x limited to that inclusive range. Include only the function and a short comment.",
    "Explain why Earth has seasons and give one example of how seasons affect daily life.",
]


def request(base, model, prompt, barrier):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 192,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    barrier.wait(timeout=30)
    started = time.perf_counter()
    first = None
    finished = None
    usage = {}
    parts = []
    with urllib.request.urlopen(req, timeout=1200) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            event = line[5:].strip()
            if event == "[DONE]":
                finished = time.perf_counter()
                break
            payload = json.loads(event)
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                for key in ("reasoning_content", "reasoning", "content"):
                    value = delta.get(key)
                    if isinstance(value, str) and value:
                        if first is None:
                            first = time.perf_counter()
                        parts.append(value)
    if first is None or finished is None:
        raise RuntimeError("stream ended without generated text")
    completion = usage.get("completion_tokens") or 0
    return {
        "prompt": prompt,
        "text": "".join(parts),
        "started_at": started,
        "first_token_at": first,
        "finished_at": finished,
        "ttft_s": first - started,
        "decode_s": finished - first,
        "completion_tokens": completion,
        "decode_tok_s": completion / (finished - first),
        "usage": usage,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rounds = []
    began = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(10):
            print(f"[HB {time.monotonic()-began:.0f}s] phase=concurrency-test", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        for index in range(args.rounds):
            print(f"[HB {time.monotonic()-began:.0f}s] phase=round-start round={index+1}", flush=True)
            barrier = threading.Barrier(len(PROMPTS))
            with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
                futures = [pool.submit(request, args.base, args.model, prompt, barrier) for prompt in PROMPTS]
                results = [future.result() for future in futures]
            start = min(item["started_at"] for item in results)
            first = min(item["first_token_at"] for item in results)
            finish = max(item["finished_at"] for item in results)
            total_tokens = sum(item["completion_tokens"] for item in results)
            group = {
                "round": index + 1,
                "requests": [
                    {key: value for key, value in item.items() if key not in ("started_at", "first_token_at", "finished_at")}
                    for item in results
                ],
                "request_count": len(results),
                "completion_tokens": total_tokens,
                "batch_wall_s": finish - start,
                "aggregate_decode_window_s": finish - first,
                "aggregate_end_to_end_tok_s": total_tokens / (finish - start),
                "aggregate_decode_tok_s": total_tokens / (finish - first),
                "mean_ttft_s": sum(item["ttft_s"] for item in results) / len(results),
                "mean_per_request_decode_tok_s": sum(item["decode_tok_s"] for item in results) / len(results),
            }
            rounds.append(group)
            print(json.dumps({key: value for key, value in group.items() if key != "requests"}, ensure_ascii=False), flush=True)
    finally:
        stop.set()
    report = {
        "benchmark": "synchronized-4-concurrent-streaming-requests",
        "base": args.base,
        "model": args.model,
        "concurrency": len(PROMPTS),
        "rounds": rounds,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[HB] phase=concurrency-done report={args.out}", flush=True)


if __name__ == "__main__":
    main()
