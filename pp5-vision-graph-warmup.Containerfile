ARG BASE_IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261031-r1
FROM ${BASE_IMAGE}

COPY pp5-capture-input-ids.patch /tmp/pp5-capture-input-ids.patch
COPY verify-pp5-capture-input-ids.py /opt/verify-pp5-capture-input-ids.py

RUN cd /usr/local/lib/python3.12/dist-packages \
    && python3 /opt/verify-pp5-capture-input-ids.py --source vllm/v1/worker/gpu/cudagraph_utils.py --expect-broken \
    && patch --dry-run --batch --forward --fuzz=0 -p1 < /tmp/pp5-capture-input-ids.patch \
    && patch --batch --forward --fuzz=0 -p1 < /tmp/pp5-capture-input-ids.patch \
    && python3 -m py_compile vllm/v1/worker/gpu/cudagraph_utils.py \
    && python3 /opt/verify-pp5-capture-input-ids.py --source vllm/v1/worker/gpu/cudagraph_utils.py \
    && rm /tmp/pp5-capture-input-ids.patch

LABEL local.deepseek.engram="alexbrandes-mmap-fp8-row-gather"
LABEL local.deepseek.graph-warmup="pp-capture-preserve-input-ids"
LABEL local.deepseek.pp5-source="b3d5e18081285f75eb55e1e7a1c45e209d612e35"
