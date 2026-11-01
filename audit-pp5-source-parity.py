"""Compare immutable Git source with an installed PP5 image/container, without imports."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

sys.path.insert(0, str(Path(tempfile.gettempdir()) / "pp5-source-audit-deps"))
import paramiko


SUFFIXES = {".py", ".pyi", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".c"}


def digest(data: bytes) -> dict:
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "lf_sha256": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-out", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--user", default="xpz")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    phase = ["reference-git-objects"]
    done = threading.Event()

    def heartbeat():
        while not done.wait(10):
            print(f"[HB {int(time.monotonic()-started)}s] phase={phase[0]}", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    client = None
    try:
        reference = {}
        reference_data = {}
        proc = subprocess.Popen(
            ["git", "-C", str(args.repo), "archive", args.commit, "vllm"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            for entry in archive:
                if not entry.isfile() or Path(entry.name).suffix not in SUFFIXES:
                    continue
                data = archive.extractfile(entry).read()
                reference[entry.name] = digest(data)
                reference_data[entry.name] = data
        if proc.wait(timeout=30):
            raise RuntimeError(proc.stderr.read().decode("utf-8", errors="replace"))
        (args.out / "reference-manifest.json").write_text(
            json.dumps({"commit": args.commit, "files": reference}, indent=2), encoding="utf-8")

        phase[0] = "runtime-source-manifest"
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.connect(args.host, username=args.user,
                       password=os.environ["PP5_SSH_PASSWORD"],
                       look_for_keys=False, allow_agent=False,
                       timeout=12, auth_timeout=12, banner_timeout=12)

        def remote(command: str, seconds=75) -> str:
            stdin, stdout, stderr = client.exec_command(command, timeout=seconds)
            stdin.close()
            output = stdout.read().decode("utf-8", errors="replace")
            errors = stderr.read().decode("utf-8", errors="replace")
            status = stdout.channel.recv_exit_status()
            if status:
                raise RuntimeError(f"Remote exit {status}: {errors[-3000:]} {output[-1000:]}")
            return output

        package = "/usr/local/lib/python3.12/dist-packages/vllm"
        remote("mkdir -p " + shlex.quote(args.remote_out))
        state = json.loads(remote("docker inspect " + shlex.quote(args.container)))[0]
        image_id = state["Image"]
        running = state["State"]["Running"]
        prefix = ("timeout --foreground 70 docker exec -i " + shlex.quote(args.container)
                  if running else
                  "timeout --foreground 70 docker run --rm -i --network host --entrypoint python3 "
                  + shlex.quote(image_id))
        prefix += " python3 -" if running else " -"
        scan_script = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); files={}; binaries=[]
suffixes={".py",".pyi",".cu",".cuh",".cpp",".cc",".h",".hpp",".c"}
for p in sorted(root.rglob("*")):
    if not p.is_file(): continue
    name="vllm/"+p.relative_to(root).as_posix()
    if p.suffix in suffixes:
        data=p.read_bytes()
        files[name]={"size":len(data),"sha256":hashlib.sha256(data).hexdigest(),"lf_sha256":hashlib.sha256(data.replace(b"\\r\\n",b"\\n")).hexdigest()}
    elif p.suffix==".so": binaries.append({"path":name,"size":p.stat().st_size})
print(json.dumps({"root":str(root),"python":sys.version,"files":files,"binary_inventory":binaries}))
'''
        sftp = client.open_sftp()
        scan_path = args.remote_out + "/scan-runtime.py"
        with sftp.file(scan_path, "wb") as f:
            f.write(scan_script.encode("utf-8"))
        runtime_path = args.remote_out + "/runtime-manifest.json"
        remote(prefix + " " + shlex.quote(package) + " < " + shlex.quote(scan_path)
               + " > " + shlex.quote(runtime_path))
        sftp.get(runtime_path, str(args.out / "runtime-manifest.json"))
        runtime = json.loads((args.out / "runtime-manifest.json").read_text(encoding="utf-8"))
        installed = runtime["files"]
        common = set(reference) & set(installed)
        changed = sorted(p for p in common if reference[p]["lf_sha256"] != installed[p]["lf_sha256"])
        missing = sorted(set(reference) - set(installed))
        extra = sorted(set(installed) - set(reference))
        line_endings_only = sorted(p for p in common
                                  if reference[p]["sha256"] != installed[p]["sha256"]
                                  and reference[p]["lf_sha256"] == installed[p]["lf_sha256"])
        phase[0] = "capture-changed-source"
        export_names = changed + extra
        exporter = ("import pathlib,sys,tarfile\nroot=pathlib.Path(" + repr(package) + ")\n"
                    + "names=" + repr(export_names) + "\n"
                    + "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:\n"
                    + " for name in names:\n"
                    + "  p=root/name.removeprefix('vllm/')\n"
                    + "  t.add(p,arcname=name,recursive=False)\n")
        export_path = args.remote_out + "/export-changed.py"
        with sftp.file(export_path, "wb") as f:
            f.write(exporter.encode("utf-8"))
        bundle_path = args.remote_out + "/runtime-changed.tar.gz"
        remote(prefix + " < " + shlex.quote(export_path) + " > " + shlex.quote(bundle_path))
        sftp.get(bundle_path, str(args.out / "runtime-changed.tar.gz"))
        with tarfile.open(args.out / "runtime-changed.tar.gz") as archive:
            for entry in archive:
                if not entry.isfile() or entry.name not in export_names:
                    raise RuntimeError("Unexpected path in runtime source bundle")
                data = archive.extractfile(entry).read()
                path = args.out / "runtime" / entry.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        diffs = []
        change_details = []
        for name in changed:
            old = reference_data[name].decode("utf-8", errors="replace")
            new = (args.out / "runtime" / name).read_text(encoding="utf-8", errors="replace")
            diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                             fromfile="reference/"+name, tofile="runtime/"+name))
            diffs.append(diff)
            path = args.out / "reference" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(reference_data[name])
            change_details.append({"path": name, "reference_size":len(reference_data[name]),
                                   "runtime_size":installed[name]["size"],
                                   "diff_lines":len(diff.splitlines())})
        (args.out / "all-source-differences.patch").write_text("\n".join(diffs), encoding="utf-8")
        report = {
            "reference_commit":args.commit, "container":args.container,
            "runtime_image_id":image_id,
            "runtime_source_mode":"running-container" if running else "immutable-image-of-stopped-container",
            "reference_source_files":len(reference), "runtime_source_files":len(installed),
            "identical_source_files":len(common)-len(changed),
            "changed":change_details,"missing":missing,"extra":extra,
            "line_endings_only":line_endings_only,
            "binary_inventory":runtime["binary_inventory"],
            "native_binary_source_provenance":"not established by installed Python/source comparison",
            "review_status":"pending difference classification and correction",
        }
        (args.out / "parity-report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        sftp.close()
        print(json.dumps({k: v for k, v in report.items() if k not in ("extra", "binary_inventory")}
                         | {"extra_count": len(extra), "binary_count": len(runtime["binary_inventory"])},
                         indent=2), flush=True)
        print("SOURCE_PARITY_AUDIT_DONE", flush=True)
    finally:
        if client is not None:
            client.close()
        done.set()


if __name__ == "__main__":
    main()
