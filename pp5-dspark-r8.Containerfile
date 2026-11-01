ARG BASE_IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r8-perf
FROM ${BASE_IMAGE}

COPY dspark-construction-offload.patch /tmp/dspark-construction-offload.patch

RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch --batch --forward --fuzz=0 -p1 < /tmp/dspark-construction-offload.patch \
    && python3 -m py_compile \
      vllm/model_executor/models/qwen3_dspark.py \
      vllm/models/deepseek_v4_1/nvidia/dspark.py \
      vllm/v1/worker/gpu/spec_decode/dspark/utils.py \
    && rm /tmp/dspark-construction-offload.patch

LABEL local.deepseek.engram="ssd-mmap-mandatory-host-break-full-capable"
LABEL local.deepseek.dspark="independent-offloader-uva-head"
LABEL local.deepseek.pp5-source="b3d5e18081285f75eb55e1e7a1c45e209d612e35"
