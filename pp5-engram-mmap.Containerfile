ARG BASE_IMAGE=local/deepseek-v41-official-pp5:20261031
FROM ${BASE_IMAGE}

COPY engram_mmap.py /tmp/engram_mmap.py
COPY disk_engram_mmap.py /tmp/disk_engram_mmap.py
COPY install-pp5-engram-mmap.py /opt/install-pp5-engram-mmap.py
COPY verify-real-engram-mmap.py /opt/verify-real-engram-mmap.py

RUN python3 /opt/install-pp5-engram-mmap.py

LABEL local.deepseek.engram="alexbrandes-mmap-fp8-row-gather"
LABEL local.deepseek.pp5-source="b3d5e18081285f75eb55e1e7a1c45e209d612e35"
