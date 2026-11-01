"""Bounded final state, adapter identity and service-health acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--image", required=True)
    args = p.parse_args()
    stopped = threading.Event()
    start = time.monotonic()
    def beat():
        while not stopped.wait(10):
            print(f"[HB {time.monotonic()-start:.0f}s] phase=r6-final-acceptance", flush=True)
    threading.Thread(target=beat, daemon=True).start()
    def command(argv):
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=25)
    def http(path):
        with urllib.request.urlopen("http://127.0.0.1:9010" + path, timeout=10) as r:
            return r.status, r.read().decode()
    try:
        live = json.loads(command(["docker", "inspect", args.container]))[0]
        image = json.loads(command(["docker", "image", "inspect", args.image]))[0]
        assert live["State"]["Running"] and not live["State"]["OOMKilled"]
        assert http("/health")[0] == 200
        models = json.loads(http("/v1/models")[1])
        assert any(x["id"] == "deepseek-v4.1-flash-1017" for x in models["data"])
        metrics = http("/metrics")[1]
        counts = re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*?\})?\s+([0-9.eE+-]+)', metrics, re.M)
        assert counts and all(float(x) == 0 for x in counts)
        helper = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/disk_engram_mmap.py"
        live_hash = command(["docker", "exec", args.container, "sha256sum", helper]).split()[0]
        baked_hash = command(["docker", "run", "--rm", "--network", "host", "--entrypoint", "sha256sum", args.image, helper]).split()[0]
        manifest = json.loads((args.out.parent / "r6-alignment-manifest.json").read_text())
        expected = manifest["approved_adapter_revision"]["r6_sha256"]
        assert live_hash == baked_hash == expected
        logs = command(["docker", "logs", args.container])
        severe = [line for line in logs.splitlines() if re.search(r'Traceback \(most recent|CUDA out of memory|torch\.OutOfMemoryError|EngineCore.*failed', line, re.I)]
        assert not severe, severe
        assert "Stopping profiler..." in logs
        remnants = []
        for path in Path("/proc").glob("[0-9]*/comm"):
            try:
                if path.read_text().strip() in ("bpftrace", "py-spy"):
                    remnants.append(str(path.parent))
            except (OSError, PermissionError):
                pass
        assert not remnants, remnants
        for name in ("deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r4", "deepseek-v41-pp5-alex-engram-mmap-20261031-r5-v1-profile"):
            assert not json.loads(command(["docker", "inspect", name]))[0]["State"]["Running"]
        args.out.with_name("final-container-inspect.json").write_text(json.dumps([live], indent=2))
        args.out.with_name("final-server.log").write_text(logs)
        result = {"ok": True, "health_status": 200, "container": args.container,
                  "live_base_image_id": live["Image"], "reproducible_image": args.image,
                  "reproducible_image_id": image["Id"], "helper_sha256": live_hash,
                  "live_and_baked_helper_identical": True, "profile_stop_logged": True,
                  "sampler_processes_remaining": remnants, "active_requests": counts,
                  "rollback_r4_r5_stopped_and_retained": True, "severe_log_errors": severe,
                  "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        print("R6_FINAL_ACCEPTANCE_DONE", flush=True)
    finally:
        stopped.set()


if __name__ == "__main__":
    main()
