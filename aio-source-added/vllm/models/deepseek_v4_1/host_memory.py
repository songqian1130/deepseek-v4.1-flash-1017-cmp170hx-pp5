# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host memory headroom guards for bounded Engram offloads."""

from pathlib import Path

GiB = 1024**3


def host_available_bytes() -> int:
    info = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return int(info["MemAvailable"].split()[0]) * 1024


def check_host_headroom(additional_bytes=0):
    """Stop an allocation that would push the host close to exhausting RAM."""
    if host_available_bytes() < 64 * GiB + additional_bytes:
        raise RuntimeError(
            "DeepSeek Engram host allocation stopped before exhausting RAM: "
            "64 GiB of available memory must remain"
        )
    entry = next(
        (
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if entry is None:
        return
    path = Path("/sys/fs/cgroup") / entry.lstrip("/")
    while path != Path("/sys/fs"):
        limit_path = path / "memory.max"
        if limit_path.is_file():
            limit = limit_path.read_text().strip()
            used = int((path / "memory.current").read_text())
            if limit != "max" and used + additional_bytes + 8 * GiB > int(limit):
                raise RuntimeError(
                    "DeepSeek Engram host allocation exceeds cgroup headroom"
                )
        path = path.parent
