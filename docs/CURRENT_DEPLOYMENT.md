# Current production deployment — 2026-11-02

当前保留 **AIO SSD Engram + lazy receive + shared mapping**。本次收尾只更新文档和启动入口，不重启服务、不做新性能实验。

## 唯一正式启动入口

服务器目录 `/run/media/xpz/models/deploy-llm`：

```bash
bash /run/media/xpz/models/deploy-llm/start-deepseek-official.sh
# 兼容入口，直接转调上面同一个脚本：
bash /run/media/xpz/models/deploy-llm/start-deepseek-official-lan.sh
```

两个文件与仓库根目录同名文件保持一致。入口复用已验收容器，运行中只做 health/models 检查；已停止则启动；容器丢失时明确报错，不偷偷回退旧方案。1200 秒就绪期限并输出心跳。

systemd user `deepseek-v41-official.service` 的 `lan-pp5.conf` 清空旧 ExecStart/ExecStop，调用正式入口，Type=oneshot、RemainAfterExit=yes、TimeoutStartSec=1300。它是启动/就绪入口，不是自动故障重启守护。8081 记录网关不变。

## 已验收配置

| 项目 | 当前值 |
|---|---|
| 容器 | `deepseek-v41-pp5-aio-9010-20261102` |
| 镜像 tag | `local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102` |
| 本地 image ID | `sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1` |
| API / 模型名 | `http://192.168.3.44:9010/v1` / `deepseek-v4.1-flash-1017` |
| 权重 | `/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official`，原版量化 |
| 并行 / 专家 | PP5/TP1，`8,8,8,8,8`，专家全部 GPU 常驻 |
| Engram | `storage=ssd`，O_DIRECT + Linux AIO + CUDA side stream，总行缓存 8 GiB |
| 上下文 / batch / sequences | `1048576` / `2048` / `8` |
| 显存利用率 / KV | `0.97` / `fp8_ds_mla` |
| 图 | `FULL_AND_PIECEWISE`，capture `[1,2,4,8]`，host breaks 开启 |
| lazy receive / shared mapping | `VLLM_PP5_LAZY_RECEIVE=1` / `VLLM_PP5_SHARED_MAPPING=1` |
| DSpark | 关闭；源码快照有相关分支不代表运行时启用 |
| NCCL | Ring / Simple，P2P_DISABLE=0，IB_DISABLE=1，P2P_LEVEL=SYS |
| 容器资源 | memory=112 GiB，memory-swap=112 GiB，shm=16 GiB，init，memlock=-1，nofile=1048576，stop-timeout=600 |

Engram config 中保留 `cpu_offload=true` 与 `storage=ssd` 配合，这是 Engram 配置，不是 MoE 专家 offload。1M 是配置上限，尚未完成真实 1M 请求验收。四并发约113 tok/s是合计吞吐。

## 源码、构建与恢复

- `aio-source-tracked.patch`：服务器当前源码相对 PP5 上游 `d63af5a472dc76b12d7a73d50a5af142844c15d1` 的 tracked diff，包含 lazy receive 等本地修改。
- `aio-source-added/`：git diff 不包含的新增 Engram Python/C++ 文件，必须一起恢复。
- `aio.Containerfile`：保存服务器原构建文件，开头 PP6 注释属于来源遗留描述；当前实际运行是 PP5。构建上下文需把恢复后的 `vllm/` 放在根目录，把 `aio-source-added/csrc/cpu/dsv41/engram_ssd.cpp` 放到 `native/engram_ssd.cpp`。BASE_IMAGE 必须使用兼容的本机 SM80/CUDA 基础镜像。
- `build-aiultrayang-pp5-lazy-overlay.sh`：当时从已有 AIO 镜像叠加 gpu_worker 的构建记录，依赖其写明的本机路径/旧 lazy patch，不是干净机器的一键安装器。
- 源码恢复顺序：检出上述上游 revision，`git apply --check` 后应用 tracked patch，再把 `aio-source-added/` 的内容复制到源码根目录。补丁适用性已单独检查；本次未重新构建镜像或重载模型，不能声称从零重建已经验收。
- `run-aiultrayang-pp5-matched-ab.sh` 保存候选完整 docker 参数；`clone-aio-9010.py` 从候选复制已验收配置，仅改变容器身份和端口。它依赖原候选仍存在。`promote-aio-9010.sh` 是已执行的单次晋升记录，不要对已晋升容器重复运行。
- `run-aio-stability-aba.sh` 及旧 r10/r23 启动脚本属于历史实验，会切换服务；它们不作为当前正式启动入口，也不应为了名字一致改写历史实验参数。

原始正式对照及晋升结果在 `benchmarks/2026-11/aio-stability/`；完整遥测和日志留在服务器 `aio-stability-20261102/` 与本地交付证据包。

## 手动回退（仅在明确需要时执行）

当前 r10/trap31 容器 `deepseek-v41-pp5-trap31-p2p-20261102` 保留、停止。回退会中断服务，需先确认没有活跃请求；不要同时运行两套占用同组 GPU 的容器。

1. 有限等待停止 AIO：`timeout --kill-after=15 650 docker stop --time 600 deepseek-v41-pp5-aio-9010-20261102`。
2. 将正式入口的 CONTAINER 改为上述 r10 名称，并同步到仓库/本地副本；LAN 别名继续转调正式入口。旧入口备份在服务器 `aio-stability-20261102/reports/start-deepseek-official.before.sh`。
3. 执行正式入口等待就绪，再检查 health/models、必要的语义请求。只启动 r10 而不更新正式入口会造成后续启动选择错误。

当前无需回退。AIO 没有做 lazy receive 关闭对照，因此没有“不需要 lazy receive”的结论。
