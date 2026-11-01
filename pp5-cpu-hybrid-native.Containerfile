ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PKG_DIR=/usr/local/lib/python3.12/dist-packages
COPY libdsv41_cuda.so ${PKG_DIR}/vllm/libdsv41_cuda.so
RUN python3 -c 'import ctypes,pathlib,vllm; p=pathlib.Path(vllm.__file__).parent/"libdsv41_cuda.so"; lib=ctypes.CDLL(str(p)); assert lib.dsv41_engram_ssd_version()==1; print("Engram SSD bridge version",lib.dsv41_engram_ssd_version())'
