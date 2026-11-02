# AIO SSD Engram 热态四并发与 CPU 对照

日期：2026-11-02。测量顺序 A1（当前 r10）→B（AIO候选）→A2（重新启动的 r10）。

## 结论

本轮没有复现候选此前的十秒级停顿，不能继续将其表述为已证实的热态不稳定。候选在本轮短请求四并发中更快，同时整个推理容器的每 token CPU 时间下降约 21–23%。用户已要求确认更快后采用新方案，因此进行 9010 晋升；最终上线验收另附于文末。

## 固定协议

每个 A/B block 先预热 4 轮，再测 12 轮重复提示、12 轮变化提示及 10 轮自然 EOS；每轮 4 并发。固定长度两组均 ignore_eos=true、192 tokens/request；自然 EOS 组 max_tokens=192。temperature=0，seed=123。合计 456 请求，其中正式测量 408 请求，候选正式测量 136 请求。所有慢轮保留，没有事后删除异常轮。

固定长度用于稳定并发度；自然 EOS 会随请求提前结束而进入尾部低并发，聚合值也会变化。decode 采用 (总 completion_tokens−4)/(最后 DONE−最早文本输出)；端到端包含 TTFT。SSE 间隔是输出 chunk 间隔，不等同于服务器每个 token 的内核耗时。

## 吞吐与 CPU

| 工作负载 | A1 聚合decode中位 | B 聚合decode中位 | A2 聚合decode中位 | B相对两次A |
|---|---:|---:|---:|---:|
| 重复提示、固定192输出 | 107.414 | 113.041 | 107.606 | +5.1%至+5.2% |
| 变化提示、固定192输出 | 105.244 | 115.505 | 105.821 | +9.2%至+9.7% |
| 自然EOS | 92.643 | 99.325 | 93.996 | +5.7%至+7.2% |

单位 tok/s，均为四请求合计。固定长度重复/变化提示的端到端吞吐 A1 为96.431/94.506，B为103.675/102.947，A2为95.745/94.676。变化提示组B中位提升高于均值提升，均值为112.625，对A1/A2的104.713/105.222约+7.0%至+7.6%。

| 工作负载 | A1 CPU ms/token | B CPU ms/token | A2 CPU ms/token | B降幅 |
|---|---:|---:|---:|---:|
| 重复提示 | 41.176 | 32.032 | 41.498 | 22.2%–22.8% |
| 变化提示 | 42.011 | 32.824 | 41.940 | 21.7%–21.9% |
| 自然EOS | 45.755 | 36.119 | 45.691 | 20.9%–21.1% |

CPU 来自 cgroup v2 cpu.stat，包含容器内API、engine、workers及辅助线程的 user/system 累计时间，覆盖 prefill 与 decode，按生成 token 数归一。不是单 token 墙钟延迟，不是整台主机全部内核/中断开销，也不是 Engram 函数单独的 profile。

## 稳定性

- B重复提示12轮112.403–113.784 tok/s，变异系数0.399%；A1/A2为2.960%/4.042%。
- B变化提示12轮105.514–116.275 tok/s，变异系数3.897%；A1/A2约5.175%/5.182%。
- B自然EOS10轮98.905–113.595 tok/s；单请求decode均值28.332，对A1/A2的26.809/27.062。
- B正式测量没有大于1秒的SSE间隔；重复提示最大0.112秒，变化提示最大0.550秒，自然EOS最大0.072秒。A1/A2变化提示最大0.825/0.835秒。
- 三段测试均无 cgroup CPU 限流、memory limit/OOM 事件，所有请求完整收到 DONE 与 usage。
- 样本覆盖约4分钟候选请求负载，不代表长上下文、更多并发、运行数天或所有新shape均已验收。

## 旧第3轮为何慢

Docker日志把旧第3轮定位到06:19:32–06:19:58 UTC。06:19:36.700及36.973分别报告 _sparse_attn_decode_partial_kernel 与 _sparse_attn_decode_reduce_kernel 推理期 Triton JIT，06:19:46窗口generation=0。与约25秒异常轮高度重合，强烈支持首次shape编译停顿解释，而非已证明的SSD热态不稳定。

本轮候选仍在首次变化提示时报告这两个kernel的JIT，但使用已有编译缓存时未复现十秒级停顿。没有清理编译缓存或SSD page cache；这次目标是热态部署表现。新shape仍可能首次触发编译。旧日志原文保存在previous-round3-jit.log。

## 是否节省CPU工作

实测支持容器总CPU/token下降。源码上，旧mmap路径每步需要同步取回GPU IDs、NumPy unique/compact、线程池gather、H2D及event同步；新路径把IDs异步复制到pinned buffer，通过CUDA host callback调用C++行缓存、排序和Linux AIO，再在side stream搬运/解码，以event衔接消费。

新路径仍有CPU memcpy、缓存维护、排序与io_submit/io_getevents，不是硬盘直达GPU。物理读盘量也没有普遍减少：重复提示B约486MiB，对A1约419MiB、A2约524MiB；变化提示B约1173MiB，对A1/A2约1064/1094MiB。不能将CPU收益归结为一律减少读盘。

## 因果与正确性边界

候选基于d63af5a，A1/A2基于b3d5e180/trap31；本轮是两套部署A-B-A，不是相同源码仅替换后端的单变量实验。可以确认本机该候选更快、更省容器CPU，不能把全部差异都归因于AIO代码。此前串行9组输出与token一致、smoke3/3为旧轮证据；本轮固定长度强制忽略EOS用于性能，不宣称新增逐token数值一致性证明。晋升需另做语义smoke与9010性能复验。

## 文件

正式原始 JSON 和分析已提交至仓库 `benchmarks/2026-11/aio-stability/`。完整遥测另存本地 performance-hotspots-20261102/r42-aio-stability/ 及 Artifact Hub 证据包。分析脚本analyze-aio-stability.py；基准benchmark-aio-stability.py；编排run-aio-stability-aba.sh。旧异常轮 JIT 摘录见同目录 previous-round3-jit.txt。

晋升入口promote-aio-9010.sh，包含测量门槛、镜像/Env/资源逐项验证、启动deadline、smoke、9010性能复验、默认入口更新与失败回退。用户2026-11-02明确授权在候选更快时保持候选在线。

## 9010 晋升验收

- 新容器 deepseek-v41-pp5-aio-9010-20261102 已 running，r10/trap31 exited，二者 OOM=false。候选镜像、环境变量值、GPU顺序、挂载和资源逐项核对，只变更服务身份和18023→9010端口。
- 新端口语义smoke 3/3；4轮预热后6轮固定长度四并发113–115 tok/s，CPU约32ms/token，原始数据promoted-9010.json。
- health/models=200，max_model_len=1,048,576，running/waiting=0。正式启动脚本已改为新容器，已实际执行该入口并通过health/models验证。
- 用户明确授权候选更快则保持候选在线；因此没有再次切回r10。回退容器和旧入口备份仍保留。

## 收尾与启动一致性

正式入口 start-deepseek-official.sh 已统一为 AIO 容器；start-deepseek-official-lan.sh 改为转调正式入口，两者均实际执行通过 health/models 检查，未重启运行中容器。systemd user 的有效 ExecStart 指向同一个正式入口。当前 lazy receive 与 shared mapping 均为 1；未做 AIO lazy receive 关闭对照。旧 r10/r23 和 A/B 编排保留为历史实验，不作为生产入口。详见 CURRENT_DEPLOYMENT.md。
