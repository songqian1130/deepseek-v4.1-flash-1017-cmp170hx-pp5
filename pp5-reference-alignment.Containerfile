ARG BASE_IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261031-r3-pp-capture-ids
FROM ${BASE_IMAGE}
COPY aligned/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY alignment-manifest.json reference-engram.py verify-pp5-reference-alignment.py /opt/source-alignment/
RUN python3 /opt/source-alignment/verify-pp5-reference-alignment.py \
      --package /usr/local/lib/python3.12/dist-packages/vllm \
      --manifest /opt/source-alignment/alignment-manifest.json \
      --reference-engram /opt/source-alignment/reference-engram.py \
      --out /opt/source-alignment/verified.json
LABEL local.deepseek.pp5-source="b3d5e18081285f75eb55e1e7a1c45e209d612e35"
LABEL local.deepseek.source-parity="2468-reference-identical-1-ssd-engram-adapter"
LABEL local.deepseek.graph-warmup="reference-requires-raw-input-tokens"
