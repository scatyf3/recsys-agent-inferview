---
title: 代码执行沙箱：隔离强度、冷启动、预热池与安全
status: draft
tags: [sandbox, firecracker, gvisor, microvm, cold-start, snapshot, prompt-injection]
difficulty: 3
order: 9
related: [/agent/orchestration, /agent/agentic-rl, /agent/system-design, /agent/function-calling]
agent: [ag-x-sandbox, ag-x-cold, ag-x-sec]
---

# 代码执行沙箱：隔离强度、冷启动、预热池与安全

> 模型写的代码是不可信代码。沙箱要在「隔离够强」「起得够快」「一台机器放得够多」之间取舍，还要挡住沙箱本身挡不住的 prompt injection

## 一句话结论

三种隔离从弱到强：容器共享宿主内核，攻击面是整个 syscall 接口（一个典型 Ubuntu 安装要用到 224 个 syscall）；gVisor 在用户态重新实现内核，syscall 不直接到宿主，代价是每次 syscall 变贵（Ant 测的 getpid：原生 62 ns，gVisor KVM 平台 830 ns）；Firecracker microVM 每个沙箱一个独立内核，启动 ≤125 ms、每个 VM 的 VMM 内存开销约 3 MB（QEMU 约 131 MB），单机每秒最多创建 150 个。冷启动的大头通常不是 VM 启动，而是拉镜像和装依赖（Slacker 测得拉镜像占容器启动时间的 76%），所以要用预热池和快照：预热池大小按 Little 定律 $\lambda T_r$ 加 Poisson 波动的余量算（每秒 20 个请求、补货 3 s，均值 60，99.9% 分位 85）；快照恢复跳过启动和初始化（E2B 恢复约 1 s，暂停约每 GiB 内存 4 s）。安全上沙箱只挡逃逸和资源滥用，挡不住 prompt injection：私有数据、不可信内容、对外通信三者同时具备（lethal trifecta）就可能泄露数据，要靠网络出口白名单、密钥不进沙箱、最小权限来切断其中一条。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $\lambda$ | 新沙箱请求的到达率（个/s） | 20 |
| $T_r$ | 补货时间：从开始准备一个新沙箱到它可用（s） | 3（冷启动）、0.3（快照恢复），假设 |
| $P$ | 预热池大小：提前起好的空闲沙箱个数 | 待求 |
| $m$ | 每个沙箱分配的内存 | 2 GiB |
| $o$ | 每个沙箱的虚拟化层内存开销（不含分给客户机的内存） | Firecracker 约 3 MB，QEMU 约 131 MB |
| $M$ | 一台宿主机可分给沙箱的内存 | 512 GiB（假设） |
| $n_{\text{sys}}$ | 一个任务里发生的 syscall 次数 | $10^6$（假设） |

MB = $10^6$ 字节，GiB = $2^{30}$ 字节；下面估算时把 1 GiB 近似成 1074 MB。

### 1. 三种隔离：边界在哪、代价是什么

| | 容器（runc） | gVisor | microVM（Firecracker） |
|---|---|---|---|
| 内核 | 和宿主共享 | 用户态的 Sentry 实现 Linux 接口 | 每个沙箱一个客户机内核 |
| 隔离边界 | namespace、cgroup、seccomp-bpf、chroot | Sentry 拦截 syscall，不直接透传给宿主 | 硬件虚拟化（KVM）+ 精简的 VMM |
| 主要代价 | 内核漏洞即逃逸 | 每次 syscall 变贵，部分 syscall 不支持 | 每个 VM 一份内核和 VMM 进程 |
| 典型产品（2026-10-13 查） | Daytona 默认 | Modal Sandboxes 默认 | E2B、Vercel Sandbox、AWS Lambda |

**容器的攻击面**。Firecracker 论文 §2.1 引 Tsai et al.：一个简单的 Linux 程序只需要 15 个 syscall，但一个典型的 Ubuntu 15.04 安装要用到 224 个 syscall 和 52 个 ioctl 才能正常工作。容器的安全靠 seccomp 限制 syscall，可限得越紧兼容性越差；而模型写的代码要 `pip install`、跑编译器、起数据库，什么都可能用到。共享内核意味着任何一个可达 syscall 里的内核漏洞都是逃逸。

**gVisor**（[架构文档](https://gvisor.dev/docs/architecture_guide/intro/)）：一个用 Go 写的「应用内核」。沙箱里程序的 syscall 被 Sentry 截获、在用户态处理；Sentry 自己再调宿主 syscall 时，和应用的 syscall 不是一一对应的。截获方式叫平台：2023 年年中起默认是 Systrap（用 seccomp 的 `SECCOMP_RET_TRAP` 把 syscall 变成信号），裸金属上还可以用 KVM 平台（[Platforms](https://gvisor.dev/docs/architecture_guide/platforms/)）。代价在 syscall 上，Ant 集团 2021 年测的 getpid（Systrap 之前，[gVisor 博客](https://gvisor.dev/blog/2021/12/02/running-gvisor-in-production-at-scale-in-ant/)，作者自测）：

| | 每次 getpid（ns） |
|---|---|
| 原生 | 62 |
| 原生 + KPTI | 236 |
| gVisor KVM 平台 | 830 |
| gVisor ptrace 平台 | 6249 |

算一笔：任务里有 $n_{\text{sys}} = 10^6$ 次 syscall，KVM 平台比原生多 $10^6 \times (830 - 62)\ \text{ns} = 0.77$ s。纯计算不受影响（gVisor 性能指南：内存访问和 CPU 执行没有额外开销），syscall 密集、小 IO 多的负载（大量小文件、频繁网络往返）才明显变慢。Ant 报告 70% 的应用开销 < 1%、25% < 3%。兼容性：amd64 的 352 个 syscall 里实现了 290 个（[兼容性表](https://gvisor.dev/docs/user_guide/compatibility/linux/amd64/)，2026-10-13 查）。

**Firecracker**（[NSDI'20 论文](https://www.usenix.org/system/files/nsdi20-paper-agache.pdf)，作者自测）：

1. 目标数字（§1）：每个 VM 内存开销 < 5 MB，启动到应用代码 < 125 ms，每台宿主每秒最多创建 150 个 microVM。
2. 内存开销（§5.2 图 7，VMM 进程占用减去分给客户机的内存）：Firecracker 约 3 MB，Cloud Hypervisor 约 13 MB，QEMU 约 131 MB，和客户机内存大小无关。
3. 并发启动（§5.1 图 6）：1000 个 VM、每次并发 50 个，预配置好的 Firecracker 启动时间 p99 为 146 ms。
4. 攻击面（§3.4）：VMM 跑在 jailer 里（chroot、namespace、降权、seccomp），2020 年的 seccomp 白名单是 24 个 syscall（带参数过滤）加 30 个 ioctl；QEMU 超过 140 万行代码，最多要用 270 个 syscall。
5. [SPECIFICATION.md](https://github.com/firecracker-microvm/firecracker/blob/main/SPECIFICATION.md)（2026-10-13 查）：从 InstanceStart API 调用到客户机 init ≤ 125 ms，VMM 内存开销 ≤ 5 MiB，客户机 CPU 性能 > 裸金属的 95%。

**开销对密度的影响**：一台机器能放 $\lfloor M / (m + o) \rfloor$ 个沙箱。

1. 沙箱 2 GiB（约 2147 MB）、$M = 512$ GiB：Firecracker $512 \times 1074 / 2150 \approx 255$ 个，QEMU $549{,}888 / 2278 \approx 241$ 个，差 6%。
2. 沙箱 128 MB（Lambda 最小规格）：Firecracker $549{,}888 / 131 \approx 4197$ 个，QEMU $549{,}888 / 259 \approx 2123$ 个，差一半。

所以开销只在小沙箱时要紧：agent 的沙箱一般给 1–4 GiB，VMM 开销不是瓶颈，瓶颈在内存本身和冷启动。

### 2. 冷启动拆开看

一个 coding agent 的沙箱从零到可用，要依次做：

| 阶段 | 量级 | 出处 |
|---|---|---|
| 调度、创建 | K8s 无状态 pod（不含拉镜像和 init 容器）启动 p99 ≤ 5 s | [K8s SIG-Scalability SLO](https://github.com/kubernetes/community/blob/master/sig-scalability/slos/pod_startup_latency.md) |
| 拉镜像 | 占容器启动时间 76%，但只有 6.4% 的数据会被读到 | [Slacker, FAST'16](https://www.usenix.org/system/files/conference/fast16/fast16-papers-harter.pdf)（作者自测） |
| 启动 VM 内核 | ≤ 125 ms | Firecracker 规格 |
| 克隆仓库、装依赖 | 秒到分钟，取决于仓库 | — |

后两步才是 agent 场景里真正的大头：一个中等 Python 仓库 `pip install` 一遍几十秒很常见。两个办法：

**快照恢复**：把「起好、克隆好、装好依赖」的 VM 的内存和磁盘存下来，用时恢复，跳过前面所有阶段。

1. Firecracker 的快照恢复本身只要几毫秒，但内存是按需加载的（mmap 或 userfaultfd），第一次访问每一页都要缺页处理：FaaSnap 测一个 hello-world 函数从快照恢复后执行要 200 ms 以上，而热 VM 只要 4 ms（[FaaSnap, EuroSys'22](https://www.sysnet.ucsd.edu/~voelker/pubs/faasnap-eurosys22.pdf) §2.4，作者自测）。REAP 记录下一次调用会访问哪些页、恢复时一次性预取，冷启动平均缩短 3.7 倍（[arXiv 2101.09355](https://arxiv.org/abs/2101.09355)，作者自测）。
2. E2B（[persistence 文档](https://docs.e2b.dev/sandbox/persistence)，2026-10-13 查）：每个沙箱是一个 Firecracker microVM，「创建」就是恢复快照；暂停（存内存和文件系统）约每 GiB 内存 4 s，恢复约 1 s；暂停的沙箱无限期保留。
3. AWS Lambda SnapStart：初始化后的 Firecracker 内存和磁盘快照，冷启动从几秒降到最好情况下亚秒（[文档](https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html)）。
4. 坑：同一份快照恢复多次，快照里的随机数种子、唯一 id、token 会在所有副本里一样（Firecracker 快照文档和 SnapStart 文档都专门警告）。恢复后要重新播种随机数、重新生成 id。

**预热池**：提前起好一批，见下一节。两者常一起用：池子里放的就是从快照恢复好的沙箱。

### 3. 预热池该放多少个

请求以速率 $\lambda$ 到达，每取走一个就开始补一个，补好要 $T_r$。由 Little 定律，同时「正在补」的沙箱平均有 $\lambda T_r$ 个，这段时间里来的请求只能靠池子里现成的接住。到达近似 Poisson 时，一个补货周期里的请求数均值 $\mu = \lambda T_r$、标准差 $\sqrt{\mu}$，要让请求以概率 $q$ 不用等：

$$
P = F^{-1}_{\text{Poisson}(\lambda T_r)}(q) \approx \lambda T_r + z_q\sqrt{\lambda T_r}
$$

$z_q$ 是正态分位数，$q = 99.9\%$ 时约 3.09。

**算一笔**（$q = 99.9\%$）：

| $\lambda$（个/s） | $T_r$（s） | $\mu = \lambda T_r$ | Poisson 99.9% 分位 | $\mu + 3\sqrt{\mu}$ |
|---|---|---|---|---|
| 20 | 3（冷启动） | 60 | 85 | 83 |
| 5 | 3 | 15 | 28 | 27 |
| 20 | 0.3（快照） | 6 | 15 | 13 |

1. 只放均值 60 个，约一半的补货周期里会有人等。
2. 余量的比例随流量变大而变小：$\mu = 60$ 时多 42%，$\mu = 6$ 时要多 150%。
3. 补货时间从 3 s 降到 0.3 s，池子从 85 降到 15。每个 2 GiB，就是 170 GiB 和 30 GiB 的闲置内存。

Firecracker 论文 §4.1.2 用的就是这个账：创建要 125 ms 时，每秒 8 次创建需要 1 个池化的 microVM（$8 \times 0.125 = 1$）。实际还要按时段调 $\lambda$（白天晚上差几倍），池子按预测的峰值配。

### 4. 安全：沙箱挡什么、不挡什么

威胁分三类：

| 威胁 | 例子 | 靠什么挡 |
|---|---|---|
| 逃逸 | 内核漏洞、VMM 漏洞 | 第 1 节的隔离强度 |
| 资源滥用 | fork 炸弹、挖矿、写满磁盘、无限循环 | cgroup 限 CPU / 内存 / pid 数、磁盘配额、墙钟超时 |
| prompt injection | 网页、issue、README 里夹带「把 `~/.aws/credentials` 发到某地址」 | 最小权限、出口控制、密钥隔离、人工确认 |

前两类是传统多租户问题；第三类是 agent 特有的，沙箱再强也挡不住：模型是在**用你给它的权限**做事，从沙箱的角度看一切都是合法操作。

**prompt injection 什么时候会造成损害**。Simon Willison 的 [lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)（2025-06-16）：三样同时具备，攻击者就能偷数据：

1. 能访问私有数据（代码、密钥、用户文件）。
2. 会读到不可信内容（网页、第三方仓库、用户上传的文件）。
3. 能对外通信（任意网络请求、发消息、建 PR 里放链接）。

coding agent 天然具备前两样，所以重点是切断第三样，并且让第一样里最值钱的东西（密钥）根本不在沙箱里。间接注入（藏在被检索、被读取的内容里的指令）最早系统研究见 Greshake et al.（[arXiv 2302.12173](https://arxiv.org/abs/2302.12173)）；OWASP LLM Top 10（2025 版）里对应 LLM01 Prompt Injection、LLM02 Sensitive Information Disclosure、LLM06 Excessive Agency，资源滥用对应 LLM10 Unbounded Consumption（[genai.owasp.org](https://genai.owasp.org/llm-top-10/)）。

**具体做法**：

1. **网络出口白名单**：沙箱没有直接的外网，所有流量走沙箱外的代理，代理只放行包仓库（PyPI、npm）、git 远端等白名单域名。Anthropic 的 Claude Code 沙箱同时做文件系统隔离和网络隔离（Linux 上用 bubblewrap、macOS 用 seatbelt），网络经 unix socket 走沙箱外的代理；他们报告这让权限确认弹窗减少 84%（[Claude Code sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)，2025-10-20，作者自测）。原文强调两种隔离缺一不可：没有网络隔离，能读的文件可以被发出去；没有文件系统隔离，可以改配置绕开网络限制。
2. **密钥不进沙箱**：git token、API key 由出口代理在请求出去时注入。沙箱里的代码能用这个权限（比如 push 到指定仓库），但拿不到密钥本身，也就没法把它带到别处用。
3. **最小权限**：只挂载这个任务的仓库；token 只授权这一个仓库、只给需要的操作。
4. **高风险操作人工确认**：推送到主分支、删除资源、对外发消息。
5. **设计层面**：CaMeL（[arXiv 2503.18813](https://arxiv.org/abs/2503.18813)）让一个只看可信输入的模型写出控制流，读不可信数据的模型不能改控制流；在 AgentDojo 上以可证明安全完成 77% 的任务，不设防的系统是 84%（作者自测）。安全有代价，但不大。

### 5. 产品参数（2026-10-13 查，会变）

| | 隔离 | 最长连续运行 | 其他 |
|---|---|---|---|
| [E2B](https://docs.e2b.dev/sandbox) | Firecracker microVM | Pro 24 h，Hobby 1 h | 暂停约 4 s/GiB、恢复约 1 s；Pro 100 个并发 |
| [Modal Sandboxes](https://modal.com/docs/guide/sandboxes) | gVisor（可选 VM 运行时） | 默认 5 分钟，最多 24 h | 文件系统快照用于跑超过 24 h 的任务 |
| [Vercel Sandbox](https://vercel.com/docs/sandbox) | Firecracker microVM | Pro 24 h，Hobby 45 分钟 | 默认超时 5 分钟 |
| [Daytona](https://www.daytona.io/docs/en/sandboxes/) | 默认容器，可选 VM | — | 宣称创建 < 90 ms（未给测量方法） |

几家的上限都在 24 h 附近：更长的任务要靠暂停和恢复，或者把状态存成文件系统快照、换一个新沙箱接着跑。

## 面试追问

::: details Q：跑模型生成的代码，为什么普通 Docker 容器不够？
容器和宿主共享内核，隔离靠 namespace、cgroup 和 seccomp。模型写的代码可能用到任意 syscall（装包、编译、起服务），seccomp 很难收紧；任何一个可达 syscall 的内核漏洞都是逃逸。多租户场景至少上 gVisor（syscall 在用户态处理）或 microVM（每个沙箱一个内核，边界是硬件虚拟化）。
:::

::: details Q：gVisor 和 Firecracker 怎么选？
gVisor 不需要嵌套虚拟化，能跑在普通云主机上，启动快、密度高，代价是 syscall 密集的负载变慢、少数 syscall 不支持。Firecracker 兼容性最好（真的 Linux 内核），隔离边界是 KVM，要裸金属或支持嵌套虚拟化的机器。agent 场景里两者都常见：Modal 用 gVisor，E2B、Vercel 用 Firecracker。
:::

::: details Q：Firecracker 启动只要 125 ms，为什么用户还要等好几秒？
125 ms 只是内核启动。前面有调度和拉镜像（Slacker 测得拉镜像占容器启动的 76%），后面有克隆仓库、装依赖，常常几十秒。解法是快照：把装好依赖的 VM 存下来直接恢复；再配预热池，请求来时池子里已经有恢复好的。
:::

::: details Q：快照恢复后第一次请求为什么还是慢？
恢复时内存是按需映射的，第一次访问每一页都要缺页处理、从快照文件读进来。FaaSnap 测 hello-world 从快照恢复后要 200 ms 以上，热 VM 只要 4 ms。解法是记录工作集、恢复时预取（REAP），或者恢复后先跑一遍预热。
:::

::: details Q：预热池放多少个？
补货期间（$T_r$）来的请求只能靠池子接住，平均 $\lambda T_r$ 个（Little 定律）；到达有波动，按 Poisson 分位数配，约 $\lambda T_r + 3\sqrt{\lambda T_r}$。20 个/s、3 s 补货要约 85 个；快照把补货降到 0.3 s，只要约 15 个。
:::

::: details Q：沙箱隔离做得再好，agent 还会有什么安全问题？
prompt injection。模型读到的网页、issue、文件里夹带指令，它会用你给的权限执行，沙箱看来都是合法操作。私有数据、不可信内容、对外通信三者同时具备就能泄露数据。coding agent 前两样避不开，所以切第三样：网络出口走白名单代理；密钥不进沙箱，由代理注入；token 最小权限；危险操作人工确认。
:::

::: details Q：从同一个快照恢复出 1000 个沙箱，有什么隐患？
快照里的状态被复制了 1000 份：随机数生成器的种子、初始化时生成的唯一 id、缓存的 token 都一样。会导致加密随机数可预测、id 冲突。恢复后要重新播种随机数、重新生成 id 和凭据（Firecracker 和 Lambda SnapStart 文档都有专门警告）。
:::

## 手撕

预热池大小：给定到达率、补货时间和目标命中率，用 Poisson 分位数算池子大小，和均值加 3 倍标准差的近似对照：

```python
from math import exp, sqrt

def poisson_quantile(mu, q):
    k, p, cdf = 0, exp(-mu), exp(-mu)
    while cdf < q:
        k += 1
        p *= mu / k          # P(X = k) = P(X = k-1) * mu / k
        cdf += p
    return k

def warm_pool(lam, t_refill, q=0.999):
    mu = lam * t_refill                  # Little 定律：补货期间平均来多少个请求
    return poisson_quantile(mu, q), mu + 3 * sqrt(mu)

for lam, tr in [(20, 3), (5, 3), (20, 0.3)]:
    print(lam, tr, warm_pool(lam, tr))
# 20 3   (85, 83.2)
# 5 3    (28, 26.6)
# 20 0.3 (15, 13.3)
```

常见变体：按小时给不同的 $\lambda$，池子跟着预测调整；池子里的沙箱按镜像类型分组（Python、Node 各一个池），每组单独算；把闲置内存 $P \times m$ 换算成成本，和「让 1% 的请求多等 $T_r$」比。

## 参考

- [Firecracker: Lightweight Virtualization for Serverless Applications (NSDI'20)](https://www.usenix.org/system/files/nsdi20-paper-agache.pdf)：§1 目标数字、§2.1 syscall 攻击面、§3.4 jailer 与 seccomp、§4.1.2 预热池、§5 启动和内存开销
- [Firecracker SPECIFICATION.md](https://github.com/firecracker-microvm/firecracker/blob/main/SPECIFICATION.md)、[快照文档](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md)：规格保证、快照按需加载与唯一性警告
- [gVisor 架构](https://gvisor.dev/docs/architecture_guide/intro/)、[平台](https://gvisor.dev/docs/architecture_guide/platforms/)、[性能指南](https://gvisor.dev/docs/architecture_guide/performance/)、[Ant 生产实践](https://gvisor.dev/blog/2021/12/02/running-gvisor-in-production-at-scale-in-ant/)：Sentry、Systrap、getpid 开销
- [Slacker (FAST'16)](https://www.usenix.org/system/files/conference/fast16/fast16-papers-harter.pdf)：拉镜像占启动时间 76%
- [Kubernetes pod 启动延迟 SLO](https://github.com/kubernetes/community/blob/master/sig-scalability/slos/pod_startup_latency.md)
- [FaaSnap (EuroSys'22)](https://www.sysnet.ucsd.edu/~voelker/pubs/faasnap-eurosys22.pdf)、[REAP (ASPLOS'21)](https://arxiv.org/abs/2101.09355)：快照恢复与缺页
- [AWS Lambda SnapStart](https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html)
- [E2B 文档](https://docs.e2b.dev/sandbox/persistence)、[Modal Sandboxes](https://modal.com/docs/guide/sandboxes)、[Vercel Sandbox](https://vercel.com/docs/sandbox)、[Daytona](https://www.daytona.io/docs/en/sandboxes/)：产品参数（2026-10-13 查）
- [Simon Willison：The lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)
- [Greshake et al.：Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)
- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/)
- [CaMeL：Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813)
- [Anthropic：Claude Code sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)（2025-10-20）：文件系统 + 网络隔离、代理、84%
