#!/usr/bin/env python3
"""Matched real-text decode benchmark for an OpenAI-compatible endpoint."""
import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = [
    "用中文解释为什么昼夜交替发生，并给出一个生活中的例子。",
    "Explain in English how a hash table provides average constant-time lookup, including one caveat.",
    "Write a Python function named clamp that takes x, low, and high and returns x limited to that inclusive range. Include only the function and a short comment.",
]


def request(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
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
    started = time.perf_counter()
    first_token_at = None
    finished_at = None
    text_parts = []
    usage = {}
    with urllib.request.urlopen(req, timeout=1200) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            event = line[5:].strip()
            if event == "[DONE]":
                finished_at = time.perf_counter()
                break
            payload = json.loads(event)
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                for field in ("reasoning_content", "reasoning", "content"):
                    part = delta.get(field)
                    if isinstance(part, str) and part:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        text_parts.append(part)
                if choice.get("finish_reason") is not None:
                    finished_at = time.perf_counter()
    if finished_at is None:
        finished_at = time.perf_counter()
    elapsed = finished_at - started
    ttft = first_token_at - started if first_token_at is not None else elapsed
    decode_elapsed = finished_at - first_token_at if first_token_at is not None else 0
    text = "".join(text_parts)
    completion = usage.get("completion_tokens") or 0
    return {
        "prompt": prompt,
        "text": text,
        "elapsed_s": round(elapsed, 4),
        "ttft_s": round(ttft, 4),
        "decode_s": round(decode_elapsed, 4),
        "completion_tokens": completion,
        "decode_tok_s": round(completion / decode_elapsed, 3) if decode_elapsed else 0,
        "end_to_end_tok_s": round(completion / elapsed, 3) if elapsed else 0,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    results = []
    for repeat in range(args.repeats):
        for index, prompt in enumerate(PROMPTS, 1):
            print(f"[HB {int(time.time())}] phase=benchmark-request repeat={repeat+1} index={index}", flush=True)
            item = request(args.base, args.model, prompt, 192)
            item["repeat"] = repeat + 1
            item["index"] = index
            if not item["text"].strip() or item["completion_tokens"] <= 0 or item["decode_s"] <= 0:
                raise RuntimeError(f"empty or uncounted response for case {index}")
            results.append(item)
            print(
                f"[HB {int(time.time())}] phase=benchmark-ok repeat={repeat+1} "
                f"index={index} decode_tok_s={item['decode_tok_s']} "
                f"end_to_end_tok_s={item['end_to_end_tok_s']}",
                flush=True,
            )
    decode_speeds = [r["decode_tok_s"] for r in results]
    wall_speeds = [r["end_to_end_tok_s"] for r in results]
    report = {
        "benchmark": "matched-real-text-wall-clock",
        "model": args.model,
        "base": args.base,
        "repeats": args.repeats,
        "cases": len(results),
        "results": results,
        "comparison": {
            "reference_repository": "https://github.com/344303947/dsv41-flash-pp5-170hx",
            "user_target_decode_tok_s": 50,
            "matched_benchmark": False,
            "note": "The target is user-requested, not a measured reference result. A repository performance comparison requires the same prompts, token budget, thinking mode, warmup, concurrency, and wall-clock measurement on both deployments.",
        },
        "summary": {
            "mean_decode_tok_s": round(statistics.mean(decode_speeds), 3),
            "median_decode_tok_s": round(statistics.median(decode_speeds), 3),
            "min_decode_tok_s": round(min(decode_speeds), 3),
            "max_decode_tok_s": round(max(decode_speeds), 3),
            "mean_end_to_end_tok_s": round(statistics.mean(wall_speeds), 3),
            "median_end_to_end_tok_s": round(statistics.median(wall_speeds), 3),
        },
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"[HB] phase=benchmark-result cases={len(results)} "
        f"mean_decode_tok_s={report['summary']['mean_decode_tok_s']} "
        f"mean_end_to_end_tok_s={report['summary']['mean_end_to_end_tok_s']} "
        f"report={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
