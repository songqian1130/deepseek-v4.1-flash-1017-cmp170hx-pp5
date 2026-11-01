#!/usr/bin/env python3
"""Small correctness-only smoke test for an isolated PP5 endpoint.

This intentionally does not measure throughput, so it cannot be confused with
an investigation of the previous PP5 speed results.
"""
import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = [
    "请用一句话回答：2 加 3 等于几？",
    "In one short sentence, what does a dictionary store?",
    "Write one Python expression that returns the square of x.",
]
EXPECTED = [
    ("Chinese arithmetic", lambda text: "5" in text or "五" in text),
    ("English dictionary", lambda text: "key" in text.lower() and "value" in text.lower()),
    ("Python square", lambda text: "x" in text and ("**" in text or "pow" in text or "x * x" in text)),
]
BAD = re.compile(r"[\ufffd\ue000-\uf8ff]|(.)\1{9,}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="OpenAI-compatible API base, including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start = time.monotonic()
    stop = threading.Event()

    def heartbeats() -> None:
        while not stop.wait(20):
            print(f"[HB {int(time.monotonic()-start)}s] phase=smoke-test", flush=True)

    thread = threading.Thread(target=heartbeats, daemon=True)
    thread.start()
    results = []
    errors = []
    try:
        for index, prompt in enumerate(PROMPTS, 1):
            print(f"[HB {int(time.monotonic()-start)}s] phase=request-start index={index}", flush=True)
            request_body = json.dumps(
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0,
                    "stream": False,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            request = urllib.request.Request(
                args.base.rstrip("/") + "/chat/completions",
                data=request_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=1200) as response:
                    payload = json.loads(response.read())
                choices = payload.get("choices") or []
                message = choices[0].get("message") if choices else None
                if not isinstance(message, dict):
                    raise ValueError("response contains no assistant message")
                text = "\n".join(
                    part for part in (
                        message.get("reasoning_content") or message.get("reasoning") or "",
                        message.get("content") or "",
                    ) if isinstance(part, str) and part
                )
                bad = BAD.search(text)
                expected_name, expected_match = EXPECTED[index - 1]
                if not text.strip():
                    raise ValueError("assistant response is empty")
                if bad:
                    raise ValueError("response contains a replacement/private-use/repeated-character pattern")
                if not expected_match(text):
                    raise ValueError(f"response failed semantic check: {expected_name}")
                results.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "text": text,
                        "usage": payload.get("usage", {}),
                    }
                )
                print(f"[HB {int(time.monotonic()-start)}s] phase=request-ok index={index}", flush=True)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                detail = str(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        detail += ": " + exc.read().decode("utf-8", "replace")[:1000]
                    except Exception:
                        pass
                errors.append({"index": index, "error": detail})
                print(f"[HB {int(time.monotonic()-start)}s] phase=request-failed index={index} error={detail[:300]}", flush=True)
                break
    finally:
        stop.set()
        thread.join(timeout=2)

    report = {
        "ok": not errors and len(results) == len(PROMPTS),
        "base": args.base,
        "model": args.model,
        "cases_total": len(PROMPTS),
        "cases_passed": len(results),
        "errors": errors,
        "results": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[HB] phase=smoke-result passed={len(results)}/{len(PROMPTS)} report={output}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
