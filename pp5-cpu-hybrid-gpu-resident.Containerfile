ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PKG_DIR=/usr/local/lib/python3.12/dist-packages
COPY hybrid.py ${PKG_DIR}/vllm/models/deepseek_v4_1/hybrid.py
COPY hybrid_runtime.py ${PKG_DIR}/vllm/models/deepseek_v4_1/hybrid_runtime.py
RUN python3 -m py_compile ${PKG_DIR}/vllm/models/deepseek_v4_1/hybrid.py ${PKG_DIR}/vllm/models/deepseek_v4_1/hybrid_runtime.py
