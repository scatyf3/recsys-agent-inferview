---
title: 系统设计：设计一个 coding agent 平台
status: draft
tags: [system-design, coding-agent, sandbox, durable-execution, gateway, multi-tenant]
difficulty: 4
order: 14
related: [/agent/sandbox, /agent/orchestration, /agent/llm-gateway, /agent/eval-observability, /agent/prompt-caching]
section: ag-overview
agent: [5]
---

# 系统设计：设计一个 coding agent 平台

> 多用户提交编码任务，每个任务一个沙箱，跑几分钟到一小时，可以中途审批、可以从故障里恢复

## 一句话结论

按这个顺序走：**澄清需求 → 用 Little 定律估并发（并发任务数 = 到达率 × 平均时长）→ 由并发推出沙箱、模型调用、token 和钱 → 画架构 → 讲恢复、等待、安全、多租户四个深入点 → 权衡**。例子：每天 10 万个任务、峰值是平均的 3 倍、每个任务平均 20 分钟，峰值同时有约 4200 个任务、4200 个沙箱、每秒约 175 次模型调用。钱几乎全在模型上：按每个任务 \$1.49（50 轮、带 prompt cache）是每天约 \$15 万，沙箱只有约 \$3300。所以架构的重点是：任务用持久执行跑（进程挂了能续、副作用不重复），沙箱用 microVM 加快照和预热池（快、隔离强、等人时能暂停），所有模型调用过网关（限流、预算、缓存命中率），每一步都有 trace。

## 推导

### 第一步：把需求问清楚

要问出来的几件事：

1. **规模**：每天多少任务、峰谷比、一个任务平均跑多久、多少轮。
2. **任务形态**：给一个仓库和一句需求，产出是 PR 还是直接改分支？用户是盯着看（要流式进度）还是提交了就走？
3. **交互**：中途要不要人审批（比如推送前确认）？审批可能几小时后才来。
4. **可靠性**：任务跑到一半机器挂了，要求从断点续上，而且不能建两个 PR。
5. **安全**：用户的私有仓库和密钥；agent 会读不可信内容（issue、依赖的 README）。
6. **多租户**：配额、预算、公平性。

下面按这组假设算：每天 10 万个任务，峰值是平均的 3 倍，每个任务平均 20 分钟、50 轮，每个沙箱 2 vCPU、4 GiB。

### 第二步：量级估算

**并发**（Little 定律 $L = \lambda W$）：

1. 平均到达率 $100{,}000 / 86{,}400 = 1.16$ 个/s，峰值约 3.5 个/s。
2. 峰值同时在跑的任务 $3.5 \times 1200\ \text{s} = 4200$ 个，也就是 4200 个沙箱。

**沙箱资源**：

1. $4200 \times 2 = 8400$ vCPU，$4200 \times 4 = 16.4$ TiB 内存。
2. agent 的沙箱大部分时间在等模型（[评测与可观测](/agent/eval-observability) 第 5 节的例子里模型占 60%），CPU 可以超分；内存不能超分太多。按宿主机 96 vCPU / 384 GiB，内存决定至少约 44 台，CPU 不超分要 88 台。
3. 预热池（[沙箱](/agent/sandbox) 第 3 节）：峰值 3.5 个/s，从快照恢复约 1 s（E2B 的恢复时间），99.9% 分位只要 11 个；如果是冷启动加克隆仓库约 30 s，就要 138 个。快照省掉的是池子里 100 多个空闲沙箱的内存。

**模型调用**：

1. 每个任务 50 轮、20 分钟，平均每 24 s 一次调用；4200 个并发任务就是每秒约 175 次调用。
2. 每次调用新增约 2k 未缓存输入：每分钟 $175 \times 2\text{k} \times 60 = 21$M 未缓存输入 token；输出按每次 300，每分钟 3.15M。
3. 算上缓存命中的前缀，每分钟经过 prefill 的输入约 6 亿 token（每次平均 59k = 2.95M / 50）。在缓存命中不计入 ITPM 的计费规则下（[LLM 网关](/agent/llm-gateway) 第 1 节），上游额度按 21M 规划；自建推理就要按 6 亿 token/分钟的 prefill 规划，并且要 session 亲和路由才能命中前缀缓存。

**钱**：

1. 模型：每个任务 \$1.49（[编排](/agent/orchestration) 第 1 节，Sonnet 4.5 价格，2026-10-13），每天 \$149,000。不用缓存是 \$9.08 一个，每天 \$908,000。
2. 沙箱：每个任务 1/3 小时，按 \$0.10 / 沙箱小时（假设），每天约 \$3,300。
3. 结论：模型比沙箱贵约 45 倍。缓存命中率掉 10 个点，比沙箱整体翻倍还贵；优化的优先级是缓存、轮数、模型选择，然后才是沙箱密度。

**存储**：

1. 事件历史和对话：只存增量（每个任务约 110k 新 token，约 0.4 MB），每天约 44 GB。
2. 工作区：每轮把工作树的改动提交到一个隐藏分支（git 只存差异，几 KB 到几 MB）。
3. 完整沙箱快照（内存 + 磁盘）：只在暂停时打，几 GB 一个，设过期时间。

### 第三步：架构

```
客户端（Web / IDE / CLI）
   │  HTTPS 提交任务；SSE / WebSocket 推进度
   ▼
[API 网关]        认证、租户配额、每用户并发上限
   ▼
[任务服务]        建任务、查状态、审批接口 ──> 任务库
   ▼
[编排：持久执行]   每个任务一个 workflow，worker 跑 agent loop
   │      │                      事件历史（只存引用和增量）
   │      └──> [对象存储]        上下文、工作区快照、产物
   │
   ├──> [LLM 网关]  限流（token bucket）、预算、路由 / 降级、计量 ──> 模型提供方 / 自建推理
   │
   └──> [沙箱管理]  调度到宿主机、预热池、快照 / 暂停 / 恢复
            ▼
        [沙箱集群]  每个任务一个 microVM
            │  无直接外网
            ▼
        [出口代理]  域名白名单、注入 git token、审计
            ▼
        包仓库、git 远端

旁路：trace（OpenTelemetry GenAI span）、指标、回放回归集
```

### 第四步：深入点

**1. 一个任务的生命周期与恢复**（[编排与持久执行](/agent/orchestration)）

1. 任务 = 一个 Temporal 式的 workflow。每次模型调用、每次工具调用都是一个 activity，结果进事件历史；上下文本体放对象存储，历史里只存引用，避免撞上 50 MB 的历史上限。
2. worker 挂了：另一个 worker 重放 workflow，已完成的 activity 直接取结果，从断点继续。
3. 沙箱宿主机挂了：事件历史还在，但沙箱里的文件没了。新沙箱从仓库原始 commit 起，检出隐藏分支上最后一次提交的工作树；最后一次提交之后的那几步工具调用如果改过文件就重放（所以每轮提交一次，最多丢一轮）。
4. 对外的副作用（建 PR、评论 issue）带幂等键 `hash(task_id, step_id)`，重放不会建两个。
5. 每个任务带预算（token、美元、轮数、墙钟时间），存在 workflow 状态里，恢复后接着累加。

**2. 等人的时候**

1. 需要审批时，workflow 挂起等信号，不占 worker。
2. 沙箱不能一直开着等几小时：打快照、暂停、释放资源。E2B 的数字是暂停约每 GiB 内存 4 s（4 GiB 约 16 s）、恢复约 1 s（2026-10-13 查）。等待预计超过几分钟就暂停。
3. 恢复时 prompt cache 大概率已经过期（5 分钟或 1 小时 TTL），第一次调用要重新写缓存，这是等待的隐性成本。

**3. 沙箱和安全**（[沙箱](/agent/sandbox)）

1. 隔离：多租户、跑不可信代码，用 microVM（Firecracker）或 gVisor，不用普通容器。
2. 启动：按语言准备基础快照（Python、Node……），热门仓库再做「已克隆、已装依赖」的快照；池子按第二步的公式配。
3. prompt injection：私有代码、不可信内容（issue、依赖文档）、对外通信三者俱全。切断对外通信：沙箱无直接外网，只经出口代理访问白名单域名；git token 由代理注入，沙箱里拿不到；token 只授权这一个仓库；推送到保护分支要人确认。
4. 资源滥用：cgroup 限 CPU / 内存 / 进程数，磁盘配额，墙钟超时。

**4. 多租户与成本**（[LLM 网关](/agent/llm-gateway)）

1. 每个租户一个 token bucket（请求数和 token 数各一个），加保底和共享池；每个用户的并发任务数上限。
2. 网关按 usage 逐项记账（未缓存输入、缓存写、缓存读、输出），按租户和任务汇总；任务超预算由编排层停下。
3. 自建推理时，同一任务的调用路由到同一个推理实例，才能命中前缀 KV（session 亲和）。
4. 子任务（读文档、总结日志）用便宜模型；同一任务内不随便换模型，换了缓存就失效。

**5. 可观测与发布**（[评测与可观测](/agent/eval-observability)）

1. 每个任务一棵 span 树：`invoke_agent` 下挂 `chat` 和 `execute_tool`，带 token 和耗时；用户看到的进度流也从这里来。
2. 发布前用回归集重放，比较成功率（配对比较）、轮数、成本、延迟。
3. workflow 代码改动要版本化，正在跑的老任务按老逻辑重放；worker 新旧版本并存、逐步切流（Anthropic 说的 rainbow deployment）。

### 第五步：权衡

| 选择 | A | B | 怎么选 |
|---|---|---|---|
| 隔离 | microVM：兼容性好、隔离强，要裸金属或嵌套虚拟化 | gVisor：普通云主机可用、密度高，syscall 密集负载慢 | 多租户不可信代码两者都行；需要 Docker-in-Docker、特殊内核特性选 microVM |
| 等待时的沙箱 | 一直开着：恢复零延迟 | 快照暂停：省资源，恢复 1 s 左右，缓存可能过期 | 预计等待超过几分钟就暂停 |
| 恢复粒度 | 每轮提交工作树：便宜，最多丢一轮 | 定期整机快照：能恢复进程状态，贵 | 默认前者，长时间运行的服务类任务加后者 |
| 单 / 多 agent | 单 agent + 上下文压缩：改同一个仓库时决策一致 | 多 agent：并行快，token 多、易冲突 | coding 主线单 agent，调研、批量分析用 sub-agent |
| 模型来源 | API：省运维、按量付费、受上游限流 | 自建：可控、可做 session 亲和，要自己管容量 | 先 API，规模到每天几十万美元再评估自建 |

## 面试追问

::: details Q：峰值要准备多少个沙箱？怎么算？
Little 定律：并发 = 到达率 × 平均时长。峰值 3.5 个任务/s、平均 20 分钟，同时约 4200 个任务、4200 个沙箱。再加一个预热池接住补货期间的请求：快照恢复 1 s 时 99.9% 分位只要约 11 个，冷启动 30 s 要约 138 个。
:::

::: details Q：这个平台的钱主要花在哪？
模型。每个任务 50 轮、带缓存约 \$1.49，每天 10 万个就是 \$14.9 万；沙箱每个任务 20 分钟，约 \$0.03，每天约 \$3300。所以优化顺序是 prompt cache 命中率、减少轮数、按子任务选模型，然后才是沙箱密度。
:::

::: details Q：任务跑到一半，跑 worker 的机器挂了，怎么恢复？
任务是持久执行的 workflow，每次模型调用和工具调用的结果都在事件历史里。另一个 worker 重放 workflow，已完成的步骤直接取记录、不重新执行，从断点继续。对外副作用带幂等键，不会重复。如果挂的是沙箱宿主机，新沙箱从隐藏分支上最后一次提交恢复工作树，最多重放一轮的文件改动。
:::

::: details Q：用户要审批，可能几小时后才回来，资源怎么处理？
workflow 挂起等信号，不占 worker；沙箱打快照、暂停、释放（E2B 暂停约每 GiB 4 s，恢复约 1 s）。恢复后第一次模型调用多半没有缓存（TTL 过了），要重新写缓存，这部分成本要算进去。
:::

::: details Q：agent 读到一个恶意 issue，让它把仓库的密钥发到外面，怎么防？
沙箱挡不住这个，因为模型是在用合法权限做事。要切断对外通信这一条：沙箱没有直接外网，只经出口代理访问白名单域名；密钥不进沙箱，由代理在请求出去时注入；token 只授权这个仓库；推送到保护分支、对外发消息要人确认。
:::

::: details Q：怎么保证改了 prompt 或换了模型不会变差？
维护一个回归集（线上任务的输入、仓库快照、判定方式），每次改动重放，配对比较成功率，同时看轮数、成本、延迟；通过后灰度放量，盯线上指标。workflow 代码改动要版本化，老任务按老逻辑跑完。
:::

## 参考

- [Anthropic：How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：断点续跑、rainbow deployment、token 倍数
- [Temporal Cloud limits](https://docs.temporal.io/cloud/limits)、[Workflow Definition](https://docs.temporal.io/workflow-definition)：事件历史上限、确定性重放
- [E2B persistence](https://docs.e2b.dev/sandbox/persistence)（2026-10-13 查）：暂停 / 恢复时间
- [Firecracker (NSDI'20)](https://www.usenix.org/system/files/nsdi20-paper-agache.pdf)：microVM 开销、预热池
- [Anthropic：Rate limits](https://platform.claude.com/docs/en/api/rate-limits)、[Pricing](https://platform.claude.com/docs/en/about-claude/pricing)（2026-10-13 查）：ITPM 计数、单价
- [Anthropic：Claude Code sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)：文件系统 + 网络隔离、出口代理
- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-spans.md)：span 结构
- 推理侧的系统设计（容量、路由、扩缩容）见 sister site 的 [设计一个 LLM 推理服务](https://scatyf3.github.io/ai-infra-inferview/basics/system-design-llm-serving)
