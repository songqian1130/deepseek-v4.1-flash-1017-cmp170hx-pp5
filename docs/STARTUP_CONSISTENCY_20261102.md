# AIO 文档与启动入口收尾核验

日期：2026-11-02。只做文档、脚本同步和只读核验，没有重启模型或重新跑性能测试。

- 正式脚本、LAN 转调脚本在工作副本、Git checkout 与服务器三处 SHA256 相同。正式：`e59feca4e5fba89bde3094fd6bce901f6645bd90745b734f2ef9a43d9fc81f40`；LAN：`acc6784e41d163946d8ed4b6167968aa05df999b15e6f83bdc113648559d8b9e`。
- 两个入口在服务器实际执行均输出 ready，通过 health/models；容器 StartedAt 仍为 `2026-11-02T13:58:55.325773822Z`。服务器两个入口均有执行权限。
- systemd user 有效 drop-in 的 ExecStart 调用正式入口，旧 ExecStart/ExecStop 已清空。网关 8081 保持不变。
- 实际运行容器 `deepseek-v41-pp5-aio-9010-20261102`，image ID `sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1`；r10/trap31 为 exited、OOM=false。
- Docker 实时 Env/Cmd 核对：PP5/TP1、8,8,8,8,8、1M/2048/8、SSD 8GiB cache、lazy receive=1、shared mapping=1、无 speculative config。
- 六个新增 shell 脚本经 Linux `bash -n` 通过；五个 Python 文件 compile 通过；正式原始 JSON 均可解析，分析脚本重新计算成功，两项实验/晋升退出码均为 0。
- tracked source patch 对固定上游 d63af5a472dc76b12d7a73d50a5af142844c15d1 的七个原始文件 `git apply --check` 通过；检查使用临时目录，没有修改服务器源码。新增的三个源码文件另外保存。
- README、ENGINEERING_NOTES、RESULTS 已改为当前 AIO 状态；旧 r10/r23 数据继续保留并标明历史范围。未执行从零镜像重建，未增加 lazy receive on/off 数值一致性结论。

Git 提交和远端 SHA 对齐在提交后核验；Artifact Hub 新交付记录与 Git SHA 在最终回复给出。现有独立修改 profile-pp5-runner-ab.py 不纳入本次提交。
