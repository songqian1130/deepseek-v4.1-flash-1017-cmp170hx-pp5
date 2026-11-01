FROM local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r8-perf
COPY disk_engram_mmap_async_ids.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/disk_engram_mmap.py
LABEL local.deepseek.engram="ssd-mmap-async-hash-index-candidate"
