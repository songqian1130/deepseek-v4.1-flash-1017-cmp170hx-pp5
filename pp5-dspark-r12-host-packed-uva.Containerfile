ARG BASE_IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r11-dspark-construction-uva
FROM ${BASE_IMAGE}

COPY patch-dspark-host-packed-uva.py /tmp/patch-dspark-host-packed-uva.py

RUN cd /usr/local/lib/python3.12/dist-packages \
    && python3 /tmp/patch-dspark-host-packed-uva.py \
    && python3 -m py_compile \
      vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py \
      vllm/v1/worker/gpu/spec_decode/dspark/utils.py \
      vllm/model_executor/layers/quantization/mxfp4.py \
      vllm/models/deepseek_v4_1/nvidia/dspark.py \
      vllm/model_executor/models/qwen3_dspark.py \
    && rm /tmp/patch-dspark-host-packed-uva.py

LABEL local.deepseek.engram="ssd-mmap-mandatory-host-break-full-capable"
LABEL local.deepseek.dspark="construction-uva-host-packed-output"
LABEL local.deepseek.pp5-source="b3d5e18081285f75eb55e1e7a1c45e209d612e35"
