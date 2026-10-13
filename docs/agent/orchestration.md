---
title: 编排与持久执行：多 agent 的 token 账、重试与幂等、断点续跑
status: draft
tags: [orchestration, multi-agent, durable-execution, retry, idempotency, temporal, langgraph]
difficulty: 3
order: 10
related: [/agent/agent-workload, /agent/context-management, /agent/prompt-caching, /agent/sandbox, /agent/llm-gateway]
agent: [ag-o-multi, ag-o-durable]
---

# 编排与持久执行：多 agent 的 token 账、重试与幂等、断点续跑

> 一个任务是几十轮「调模型 → 调工具」。编排层要算清楚每多一轮、每多一个 sub-agent 花多少 token，再保证任务跑到一半挂了能从断点续上、续的时候不把副作用做两次

## 一句话结论

单个上下文跑 $N$ 轮，每轮都要重发整段历史，总输入 token 是 $NS + \Delta N(N-1)/2$，随轮数平方增长：$S = 10$k、$\Delta = 2$k、$N = 50$ 时约 295 万。把同样 100 轮的活拆给 5 个各跑 20 轮的 sub-agent，每个子上下文都短，总输入从 1090 万降到约 300 万，墙钟时间从 100 轮降到约 25 轮。但多 agent 通常会做更多事：Anthropic 的研究系统里 agent 约用聊天 4 倍的 token，多 agent 约 15 倍（作者自测）。要拆成可以并行、互不依赖的子问题；改同一份代码这类步步相关的任务，sub-agent 互相看不到对方的决定，容易冲突。可靠性上，50 步、每步失败率 2% 的任务只有 36% 能一次跑完；每步重试 2 次后是 99.96%。前提是重试安全：有副作用的工具调用带幂等键；整个任务用持久执行（Temporal 的事件历史重放、LangGraph 的 checkpoint）记下每一步的结果，进程挂了在别的机器上重放，已完成的步骤直接返回记录，不重新执行。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $N$ | 一个上下文里的轮数（一轮 = 一次模型调用 + 它发起的工具调用） | 50 |
| $S$ | 每轮都带的固定前缀：system prompt + 工具定义 + 任务描述（token） | 10,000 |
| $\Delta$ | 每轮新增的 token：模型输出 + 工具结果 | 2,000 |
| $m$ | sub-agent 个数 | 5 |
| $t$ | 一轮的墙钟时间（模型生成 + 工具执行） | 10 s（假设） |
| $p$ | 一步失败的概率（超时、429、工具报错），各步独立 | 2%（假设） |
| $r$ | 每步最多重试几次 | 2 |
| $I_{\text{in}}$ | 一个任务的总输入 token | 待求 |

单位：k = $10^3$ token，M = $10^6$ token；价格单位是美元 / 百万 token（\$/MTok）。

### 1. agent loop 的 token 账：输入随轮数平方增长

最基本的循环：

```
while True:
    resp = llm(context)                 # 第 i 轮：发整段 context
    if resp 是最终答案 or 超出预算: break
    context += resp + run_tools(resp)   # 追加模型输出和工具结果
```

第 $i$ 轮发出去的输入是 $S + (i-1)\Delta$，$N$ 轮加起来：

$$
I_{\text{in}} = \sum_{i=1}^{N}\big(S + (i-1)\Delta\big) = NS + \Delta\frac{N(N-1)}{2}
$$

代入 $S = 10$k、$\Delta = 2$k、$N = 50$：

1. 固定前缀：$50 \times 10\text{k} = 0.5$M。
2. 增长部分：$2\text{k} \times 50 \times 49 / 2 = 2.45$M。
3. 合计 **2.95M**。输出假设每轮 300 token，共 15k，和输入比可以忽略。

**prompt cache 把它打到几分之一**。第 $i$ 轮的输入里，前 $S + (i-2)\Delta$ 个 token 上一轮刚发过，可以命中缓存，只有新增的 $\Delta$ 要写缓存。按 Claude Sonnet 4.5 的价格（2026-10-13 查 [Anthropic 定价页](https://platform.claude.com/docs/en/about-claude/pricing)：输入 \$3、5 分钟缓存写入 \$3.75、缓存命中 \$0.30、输出 \$15，每 MTok）：

| | 计算 | 美元 |
|---|---|---|
| 不用缓存 | $2.95 \times 3 + 0.015 \times 15$ | 9.08 |
| 用缓存：写入 | 第 1 轮写 $S$，之后每轮写 $\Delta$：$10\text{k} + 50 \times 2\text{k} = 0.11$M，$\times 3.75$ | 0.41 |
| 用缓存：命中 | $2.95 - 0.11 = 2.84$M，$\times 0.30$ | 0.85 |
| 用缓存：输出 | $0.015 \times 15$ | 0.23 |
| 用缓存合计 | | **1.49** |

便宜约 6 倍。前提是每轮只在末尾追加、前缀一个字节都不改；缓存的细节见 [Prompt caching](/agent/prompt-caching)，上下文太长时的压缩见 [上下文管理](/agent/context-management)。

**停止条件**必须写死，否则一个卡住的任务会一直烧钱：

1. 模型返回最终答案。
2. 最大轮数（比如 100）。
3. token 或美元预算：每轮结束累加，超了就停。
4. 墙钟超时。
5. 连续 $k$ 轮做同一个工具调用、拿到同样的结果（死循环检测）。

### 2. 多 agent：fan-out 的 token 和时间

**结构**：lead agent 拆任务，派给 $m$ 个 sub-agent；每个 sub-agent 在自己的上下文里跑完整的 agent loop，最后只把结论（几千 token）交回 lead。

$$
I_{\text{in}}^{\text{multi}} = I_{\text{lead}} + \sum_{j=1}^{m} I_j, \qquad
T^{\text{multi}} \approx T_{\text{lead}} + \max_j T_j
$$

token 是各上下文之和，时间只取最慢的那个 sub-agent（它们并行）。

**算一笔：同样 100 轮的活，一个上下文做 vs 拆给 5 个**（$S = 10$k、$\Delta = 2$k、$t = 10$ s）：

1. 单 agent，100 轮：$100 \times 10\text{k} + 2\text{k} \times 100 \times 99/2 = 1\text{M} + 9.9\text{M} =$ **10.9M**；时间 $100 \times 10 = 1000$ s。
2. 5 个 sub-agent 各 20 轮：每个 $20 \times 10\text{k} + 2\text{k} \times 20 \times 19 / 2 = 0.2\text{M} + 0.38\text{M} = 0.58$M，5 个共 2.9M。
3. lead 跑 5 轮（拆任务、收结果、汇总），$5 \times 10\text{k} + 2\text{k} \times 10 = 0.07$M。
4. 合计约 **2.97M**，是单 agent 的 27%；时间 $(5 + 20) \times 10 = 250$ s。

拆开之后每个子上下文都短，平方项小了 5 倍。所以「多 agent 一定更费 token」不准确：同样的工作量，拆开反而省；费是因为多 agent 通常会去做更多的工作。

**Anthropic 的数字**（[How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)，2025-06-13，作者自测）：

1. agent 约用聊天 4 倍的 token，多 agent 系统约 15 倍。
2. Opus 4 做 lead、Sonnet 4 做 sub-agent，在内部研究评测上比单 agent 的 Opus 4 高 90.2%。
3. BrowseComp 上 token 用量一项就解释了 80% 的成绩方差；加上工具调用次数和模型选择，三项解释 95%。
4. 按问题复杂度分配：简单事实查找 1 个 agent、3–10 次工具调用；直接比较 2–4 个 sub-agent、每个 10–15 次；复杂研究 10 个以上。
5. lead 并行起 3–5 个 sub-agent、sub-agent 并行调 3 个以上工具，复杂查询的研究时间最多缩短 90%。

**什么时候不该拆**：

1. 子任务之间要共享上下文。Anthropic 原文说需要所有 agent 共享同一上下文的领域目前不适合，大多数编码任务能真正并行的部分比研究少。
2. Cognition 的 [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents)（2025-06-12）的论点：每个动作都隐含决定，两个 sub-agent 各自做了互相冲突的决定（比如一个按 A 风格写组件、一个按 B 风格），合起来就是坏结果；所以要么共享完整轨迹，要么干脆用单线程 agent，长任务靠一个专门的模型压缩历史。
3. 经验法则：广度优先、子问题互相独立（调研、搜索、批量处理文件）就拆；深度优先、步步依赖（改同一个代码库、调一个 bug）就单 agent 加上下文压缩。

### 3. 失败与重试：步数一多，「偶尔失败」变成「经常失败」

一个任务 $N$ 步，每步独立以概率 $p$ 失败，不重试时整个任务一次跑完的概率：

$$
P_{\text{ok}} = (1-p)^N
$$

每步最多重试 $r$ 次（共 $r+1$ 次机会），一步最终失败的概率变成 $p^{r+1}$：

$$
P_{\text{ok}} = \big(1 - p^{r+1}\big)^N
$$

代入 $p = 2\%$、$N = 50$：

1. 不重试：$0.98^{50} = 0.364$，**64%** 的任务半路失败。
2. 每步重试 2 次：$p^3 = 8\times10^{-6}$，$(1 - 8\times10^{-6})^{50} \approx$ **0.9996**。

前提是失败相互独立。上游整体宕机时，重试 100 次也是同一个结果，这时要靠熔断和降级（见 [LLM 网关](/agent/llm-gateway)）。

**退避**。Temporal 的 Activity 默认重试策略（[Retry Policies](https://docs.temporal.io/encyclopedia/retry-policies)，2026-10-13 查）：初始间隔 1 s、系数 2.0、最大间隔为初始的 100 倍、不限次数；第 $k$ 次重试前等

$$
\text{interval}_k = \min\big(1\text{ s} \times 2^{k},\ 100\text{ s}\big)
$$

即 1、2、4、8、16、32、64、100、100…… 秒。再加**抖动**（jitter）：在 $[0, \text{interval}_k]$ 里均匀随机取等待时间（AWS 架构博客 [Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) 里叫 Full Jitter）。不加抖动时，同一时刻被 429 的几千个任务会在同一时刻一起重试，再被一起拒绝。

**重试放大**。每一层都重试时，最底层看到的请求数是各层次数的乘积：5 层、每层最多 3 次，最坏 $3^5 = 243$ 倍。AWS Builders' Library 的 [Timeouts, retries, and backoff with jitter](https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter) 用的也是 243 倍这个例子。所以只在一层重试（通常是离副作用最近的那层），其余层快速失败。

### 4. 幂等：重试安全的前提

重试的语义是至少一次（at-least-once）：超时不代表没执行，可能请求已经生效、只是响应丢了。工具按副作用分三类：

| 类别 | 例子 | 重试 |
|---|---|---|
| 只读 | 读文件、搜索、`git diff` | 随便重试 |
| 天然幂等 | 把文件内容写成 X、`PUT` 一个资源 | 随便重试，结果一样 |
| 非幂等 | 发邮件、建 PR、扣费、`POST` 一条评论 | 要幂等键 |

**幂等键**：调用方给每个逻辑操作生成一个唯一 id，重试时带同一个 id；服务端第一次执行时把 id 和结果存下来，再看到同一个 id 就直接返回存的结果。Stripe 的做法（[Idempotent requests](https://docs.stripe.com/api/idempotent_requests)，2026-10-13 查）：

1. 存的是第一次请求开始执行后的状态码和响应体，同一个键的重试返回存下来的响应，包括 500。
2. 参数和原请求不一致就报错，防止拿旧键发新请求。
3. 键至少保留 24 小时后才可能被清理。

agent 里的键要能在重放时重新算出来，所以不用随机数，用 `hash(task_id, step_id)`：同一个任务的同一步，无论第几次执行，键都一样。

**LLM 调用本身不幂等**：同样的输入，重新采样得到的输出不同。所以 agent 的断点续跑不能「重新跑一遍前面的步骤」，而要把每一步的结果记下来，重放时直接用记录。这就是持久执行。

### 5. 持久执行：事件历史重放 vs checkpoint

**Temporal 的模型**（[Workflow Definition](https://docs.temporal.io/workflow-definition)，2026-10-13 查）：

1. 编排逻辑写成 Workflow，必须是确定性的：同样的历史输入，走同样的分支。
2. 有副作用或不确定的事（LLM 调用、工具调用、读数据库、取当前时间）都放进 Activity。
3. 每个 Activity 的调度、开始、完成和返回值都追加进 Event History，持久化在 Temporal 服务里。
4. worker 挂了，另一个 worker 拿到这个 Workflow，从头执行 Workflow 代码；遇到已经有完成记录的 Activity，直接返回记录里的结果，不重新调用。执行到历史的末尾，就回到了挂掉时的位置。
5. 重放时代码发出的命令要和历史逐条比对，对不上就报 non-determinism error。所以改编排代码要做版本控制：正在跑的老任务按老逻辑重放。

**事件历史有上限**（[Temporal Cloud limits](https://docs.temporal.io/cloud/limits)，2026-10-13 查）：一个 Workflow 的 Event History 最多 51,200 个事件或 50 MB，到 10,240 个事件或 10 MB 时告警；单个请求的 payload 最多 2 MB。

算一笔：如果把整段上下文当成 LLM Activity 的参数，第 $i$ 轮的 payload 就是 $S + (i-1)\Delta$ 个 token。按每 token 约 4 字节（假设，英文文本的粗略值）：

1. 第 50 轮的上下文 $10\text{k} + 49 \times 2\text{k} = 108$k token，约 432 KB。
2. 50 轮的 payload 加起来就是第 1 节的 2.95M token，约 11.8 MB，已经过了 10 MB 的告警线；再跑几十轮就碰 50 MB。

做法：

1. 上下文存在外部（对象存储或数据库），按内容哈希寻址；Activity 的参数和返回值只传引用和新增的那部分。
2. 历史太长时用 Continue-As-New：以当前状态为输入开一个新的 Workflow 执行，历史清零。

**LangGraph 的模型**（[Persistence / Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)，2026-10-13 查）：

1. 图的每个 super-step（同一批并行执行的节点）结束时，checkpointer 把整个图状态存一份快照，按 `thread_id` 组织。
2. 恢复时从最新的 checkpoint 接着跑。一个 super-step 里已经成功的节点，它们的写入作为 pending writes 存下来，恢复时不重跑。
3. 持久化时机三档：`exit`（只在图执行结束时存）、`async`（下一步执行时异步存，崩溃时可能丢最近的 checkpoint）、`sync`（下一步开始前同步存完）。
4. `interrupt()`（等人确认）要 checkpointer 和 thread id；恢复时**整个节点从头重新执行**，所以 interrupt 之前的副作用必须幂等。

| | 事件历史重放（Temporal） | checkpoint（LangGraph） |
|---|---|---|
| 存什么 | 每个 Activity 的结果，追加写 | 每个 super-step 后的整个状态 |
| 恢复 | 从头重放 Workflow 代码，已完成的直接取结果 | 加载最新快照，从下一步继续 |
| 对代码的要求 | Workflow 必须确定性，改代码要做版本 | 节点粒度要合适，节点内的副作用要幂等 |
| 恢复粒度 | Activity | super-step（节点做到一半挂了要重跑整个节点） |

### 6. 长任务还有哪些状态不在历史里

持久执行只保证编排层的状态。一个 coding agent 的任务还有：

1. **沙箱里的文件系统和进程**：装好的依赖、改了一半的代码、跑着的 dev server。这些不在事件历史里。原机器挂了，要么从沙箱快照恢复，要么在新沙箱里重放会改文件系统的步骤（所以这些步骤也要可重放），见 [代码执行沙箱](/agent/sandbox)。
2. **预算计数器**：已花的 token 和美元，存在编排状态里，恢复后接着累加，不能清零。
3. **外部世界**：已经建的 PR、已经发的消息，靠幂等键防重复。
4. **等人的时间**：等人审批可能几小时，期间不该占着 worker 和沙箱。Workflow 挂起不耗资源；沙箱暂停或打快照后释放。
5. **发布新版本**：Anthropic 用 rainbow deployment，新旧版本同时在跑，流量逐渐切过去，避免打断正在运行的 agent（同上博客）。

## 面试追问

::: details Q：一个 agent 跑 50 轮，为什么输入 token 远大于 50 × 每轮的量？
每轮都要把整段历史重发一遍，第 $i$ 轮输入是 $S + (i-1)\Delta$，总量 $NS + \Delta N(N-1)/2$，随轮数平方增长。$S = 10$k、$\Delta = 2$k、$N = 50$ 时是 2.95M，其中 2.45M 来自不断增长的历史。prompt cache 让重复前缀按约十分之一计价，按 Sonnet 4.5 的价格从 \$9.08 降到约 \$1.49。
:::

::: details Q：多 agent 一定比单 agent 费 token 吗？
不一定。token 是各个上下文之和，每个上下文内部是平方增长。同样 100 轮的活拆给 5 个各 20 轮的 sub-agent，总输入从 10.9M 降到约 3M，因为每个子上下文都短。实际中多 agent 更费，是因为它会做更多的事（更广的搜索、更多的工具调用）：Anthropic 报告多 agent 约是聊天的 15 倍 token，单 agent 约 4 倍。
:::

::: details Q：什么任务适合多 agent，什么不适合？
适合：能拆成互不依赖的子问题、以广度为主，比如调研、并行搜索、批量处理。不适合：步步依赖、要共享上下文，比如改同一个代码库。sub-agent 之间看不到彼此的决定，会做出风格或接口互相冲突的改动（Cognition 的论点）。后者用单 agent 加上下文压缩。
:::

::: details Q：工具调用超时了，能直接重试吗？
看副作用。只读和天然幂等的直接重试；非幂等的（发消息、建 PR、扣费）要带幂等键，键取 `hash(task_id, step_id)`，重放时能算出同一个键；服务端存下第一次的结果，同键请求直接返回。超时不代表没执行，可能已经生效只是响应丢了，所以不能假设「没成功」。
:::

::: details Q：为什么 agent 的断点续跑不能「从头再跑一遍」？
LLM 采样不确定，重跑前面的步骤会得到不同的输出，走到不同的分支，前面已经做过的副作用也会重复。持久执行把每一步的结果记下来，恢复时重放编排代码、直接取记录，已完成的步骤不重新执行。Temporal 要求编排代码本身是确定性的，LLM 和工具调用都放在 Activity 里。
:::

::: details Q：用 Temporal 跑 agent，有什么坑？
1. 事件历史有上限（51,200 个事件或 50 MB），把整段上下文当参数传，几十轮就爆；上下文放外部存储，只传引用，太长了 Continue-As-New。
2. Workflow 代码必须确定性：不能在里面直接调 LLM、取时间、生成随机数。
3. 改了编排逻辑，正在跑的老任务重放会报 non-determinism，要做版本。
4. 沙箱状态不在历史里，要单独快照或可重放。
:::

::: details Q：重试为什么要加 jitter？为什么只在一层重试？
不加 jitter，同时被 429 的大量请求会在同一时刻重试，再一起被拒，形成周期性的尖峰。只在一层重试是因为每层都重试会让底层的负载按各层次数相乘放大：5 层各 3 次就是 243 倍，上游本来就过载时会被彻底压垮。
:::

## 手撕

带幂等键、指数退避和 full jitter 的重试。工具端用一个字典模拟「存下第一次的结果」：

```python
import hashlib, random, time

def call_with_retry(fn, key, max_attempts=4, base=1.0, cap=30.0, sleep=time.sleep):
    for attempt in range(max_attempts):
        try:
            return fn(idempotency_key=key)
        except Exception:
            if attempt == max_attempts - 1:
                raise
            sleep(random.uniform(0, min(cap, base * 2 ** attempt)))   # full jitter

done = {}                                   # 服务端：幂等键 -> 第一次的结果
def create_pr(idempotency_key):
    if idempotency_key in done:
        return done[idempotency_key]        # 重试：直接返回，不再建一次
    if random.random() < 0.5:
        raise TimeoutError                  # 模拟超时（可能已执行，也可能没有）
    done[idempotency_key] = 'ok'
    return 'ok'

random.seed(0)
key = hashlib.sha256(b'task-42:step-7').hexdigest()   # 重放时能重新算出同一个键
print(call_with_retry(create_pr, key, sleep=lambda s: None))   # ok

p = 0.02
print((1 - p) ** 50, (1 - p ** 3) ** 50)    # 0.364  0.9996：不重试 / 每步重试 2 次
```

常见变体：区分可重试错误（429、超时、5xx）和不可重试错误（400、权限不足）；尊重上游返回的 `retry-after`；加一个全局的重试预算（重试请求不超过总请求的 10%），防止重试放大。

## 参考

- [Anthropic：How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)（2025-06-13）：4× / 15× token、90.2%、80% 方差、按复杂度分配 sub-agent、rainbow deployment、断点续跑
- [Cognition：Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents)（2025-06-12）：共享上下文、隐含决定冲突
- [Temporal：Retry Policies](https://docs.temporal.io/encyclopedia/retry-policies)：默认重试参数
- [Temporal：Workflow Definition](https://docs.temporal.io/workflow-definition)：确定性约束、重放
- [Temporal Cloud limits](https://docs.temporal.io/cloud/limits)：事件历史 51,200 事件 / 50 MB、payload 2 MB
- [LangGraph：Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)、[Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)：super-step checkpoint、durability 三档、interrupt 重跑整个节点
- [Stripe：Idempotent requests](https://docs.stripe.com/api/idempotent_requests)：幂等键语义
- [AWS Builders' Library：Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
- [AWS Builders' Library：Timeouts, retries, and backoff with jitter](https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter)：重试放大、只在一层重试
- [AWS 架构博客：Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)：Full Jitter
- [Anthropic 定价](https://platform.claude.com/docs/en/about-claude/pricing)（2026-10-13 查）：Sonnet 4.5 单价
