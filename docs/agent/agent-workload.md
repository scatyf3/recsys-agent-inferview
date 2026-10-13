---
title: Agent 负载画像：轮数、token、复用率，钱和时间花在哪
status: draft
tags: [agent, workload, token-accounting, prefix-reuse, cost]
difficulty: 2
order: 1
related: [/agent/prefix-reuse, /agent/prompt-caching, /agent/context-management, /agent/agent-latency]
section: ag-overview
agent: [ag-o-loop, ag-c-budget]
---

# Agent 负载画像：轮数、token、复用率，钱和时间花在哪

> 把一个 agent 任务拆成「每轮输入多少、输出多少、隔多久」，就能看出它和聊天负载差在哪，以及推理系统该往哪使劲

## 一句话结论

一个 agent 任务是几十轮「调模型 → 执行工具 → 把结果追加进上下文」的循环：Manus 报告一个任务平均约 50 次工具调用，输入 : 输出约 100 : 1；Continuum 测的 SWE-bench 轨迹平均 10.9 轮、工具调用平均 0.9 s。每轮的输入是上一轮的全部输入再加几百到几千个新 token，所以总输入随轮数平方增长，而且 96% 以上可以从前缀缓存里读。本页的例子（50 轮、每轮追加 1.5K token）：总输入 234 万 token、总输出 1.5 万，不开缓存时输入占账单的 97%，开了 prompt caching 账单降到约 1/6；时间却主要花在解码上（约 74%），其次是工具（约 19%），prefill 只占 7%。结论是**钱看输入和缓存命中率，时间看输出长度和轮数**。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $T$ | 一个任务的轮数（调用模型的次数） | 50 |
| $S$ | 固定前缀：系统提示 + 工具定义 + 用户的任务描述（token） | 10,000 |
| $o$ | 每轮模型输出（token，含工具调用本身） | 300 |
| $r$ | 每轮工具结果（token） | 1,200 |
| $a$ | 每轮追加到上下文的 token，$a = o + r$ | 1,500 |
| $L_i$ | 第 $i$ 轮的输入长度（token） | 见下 |
| $p_{\text{in}}, p_{\text{out}}$ | 每百万 token 的输入、输出价格 | 3、15 USD（假设，输出是输入的 5 倍） |
| $t_{\text{ttft}}$ | 每轮首 token 延迟（前缀命中时） | 0.6 s（假设） |
| $t_{\text{tpot}}$ | 每个输出 token 的解码时间 | 20 ms（假设） |
| $t_{\text{tool}}$ | 每次工具执行时间 | 1.5 s（假设） |

K = $10^3$，M = $10^6$。例子里的 $T$ 取 Manus 报告的量级，$o, r$ 是假设，用来演示怎么算。

### 1. 一个任务的 token 流水

第 $i$ 轮的输入是固定前缀加上前 $i-1$ 轮追加的所有内容：

$$
L_i = S + (i-1)\,a
$$

总输入是对 $T$ 轮求和，总输出是每轮输出相加：

$$
I_{\text{total}} = \sum_{i=1}^{T} L_i = T S + a\,\frac{T(T-1)}{2},\qquad O_{\text{total}} = T o
$$

第二项是轮数的平方，所以轮数翻倍，总输入接近翻四倍。代入例子：

1. $TS = 50 \times 10\text{K} = 500\text{K}$。
2. $a \cdot T(T-1)/2 = 1.5\text{K} \times 1225 = 1{,}837.5\text{K}$。
3. $I_{\text{total}} = 2{,}337.5\text{K} \approx 234$ 万 token；$O_{\text{total}} = 50 \times 300 = 15\text{K}$。
4. 输入 : 输出 $= 2337.5 / 15 \approx 156 : 1$。Manus 报告的实测是约 100 : 1，同一个量级。
5. 最后一轮的上下文 $L_{50} = 10\text{K} + 49 \times 1.5\text{K} = 83.5\text{K}$。

**前缀复用率**：第 $i$ 轮的输入里，前 $L_{i-1}$ 个 token 和上一轮完全相同，只有 $a$ 个是新的（第 1 轮的 $S$ 全是新的）。整个任务里第一次出现的 token 只有 $S + (T-1)a = 83.5\text{K}$，其余都是重复发送：

$$
h = 1 - \frac{S + (T-1)a}{I_{\text{total}}} = 1 - \frac{83.5\text{K}}{2337.5\text{K}} = 96.4\%
$$

这个 $h$ 是理论上限：只有缓存一直没被淘汰、前缀一个字节都没改时才能达到。推理引擎侧的 KV 复用见 [多轮 agent 的 KV 复用](/agent/prefix-reuse)，API 侧的计费见 [Prompt caching 的账](/agent/prompt-caching)。

### 2. 和聊天负载对比

聊天的例子（假设）：5 轮，系统提示 1K，每轮用户 100 token、回答 400 token。

| | 聊天（假设） | Agent（本页例子） |
|---|---|---|
| 轮数 | 5 | 50 |
| 每轮追加 | 用户 100 + 回答 400 | 工具结果 1,200 + 模型输出 300 |
| 总输入 | 10.5K | 2,337.5K |
| 总输出 | 2K | 15K |
| 输入 : 输出 | 约 5 : 1 | 约 156 : 1 |
| 两轮之间隔多久 | 人在读、在打字，几十秒到几分钟 | 工具执行，均值约 1–2 s（Continuum 表 1） |
| 前缀复用率上限 | 70% | 96% |

聊天的总输入：$5 \times 1.1\text{K} + 500 \times (0+1+2+3+4) = 5.5\text{K} + 5\text{K} = 10.5\text{K}$；第一次出现的 token 是 $1.1\text{K} + 4 \times 0.5\text{K} = 3.1\text{K}$，复用率上限 $1 - 3.1/10.5 = 70\%$。

三个差别决定了 agent 推理系统的设计重点：

1. **输入远多于输出**：prefill 的量和账单都由输入决定，前缀缓存是第一优先级。
2. **长前缀、只追加**：几乎每轮都能命中前缀，前提是前缀不变、请求回到存着 KV 的机器上（[session 亲和路由](/agent/prefix-reuse)）。
3. **轮间隔短**：工具执行 Continuum 测得 SWE-bench 均值 925 ms、BFCL 均值 1,923 ms，比人的思考时间短得多，所以「这一轮结束后 KV 先别扔」是划算的；但标准差很大（SWE-bench 3,550 ms），少数慢调用拖住长尾。

几个公开的量级（都是作者自测）：

1. Manus：一个典型任务约 50 次工具调用，输入 : 输出约 100 : 1。
2. Continuum（表 1）：SWE-bench 平均 10.9 轮、每个任务约 7 万 token；BFCL v4 平均 6.3 轮、约 9.3 万 token。
3. Anthropic 多 agent 研究系统：单 agent 用的 token 约是聊天的 4 倍，多 agent 约 15 倍；在 BrowseComp 上，token 用量一项就解释了表现差异的 80%。

### 3. 钱花在哪

不开缓存时，账单 $= p_{\text{in}} I_{\text{total}} + p_{\text{out}} O_{\text{total}}$：

1. 输入：$2.3375\text{M} \times 3 = 7.01$ USD。
2. 输出：$0.015\text{M} \times 15 = 0.225$ USD。
3. 合计 7.24 USD，输入占 97%。输出单价是输入的 5 倍也扭转不了，因为量差了 156 倍。

开 prompt caching（缓存读取 0.1 倍、写入 1.25 倍，Anthropic 5 分钟缓存的倍数，见 [Prompt caching 的账](/agent/prompt-caching)），每轮读上一轮的前缀、写新追加的 $a$：

1. 写入 $83.5\text{K}$ token：$0.0835 \times 3 \times 1.25 = 0.313$ USD。
2. 读取 $2{,}337.5\text{K} - 83.5\text{K} = 2{,}254\text{K}$：$2.254 \times 3 \times 0.1 = 0.676$ USD。
3. 输出不变 0.225 USD，合计 1.21 USD，约原来的 1/6。

所以 agent 的成本优化，第一件事是**让缓存命中**，第二件事才是少调几轮、把上下文压短（[上下文管理](/agent/context-management)）。

### 4. 时间花在哪

每轮的时间是首 token、解码、工具执行三段（排队另算）：

$$
t_{\text{task}} = \sum_{i=1}^{T}\left(t_{\text{ttft},i} + o_i\, t_{\text{tpot}} + t_{\text{tool},i}\right)
$$

代入例子：每轮 $0.6 + 300 \times 0.02 + 1.5 = 8.1$ s，50 轮 405 s，约 6.8 分钟。

| 部分 | 每轮 | 50 轮 | 占比 |
|---|---|---|---|
| TTFT | 0.6 s | 30 s | 7% |
| 解码 300 token | 6.0 s | 300 s | 74% |
| 工具 | 1.5 s | 75 s | 19% |

钱和时间的大头不在同一处：**钱花在输入（prefill），时间花在输出（decode）**。300 个输出 token 的解码时间是 0.6 s TTFT 的 10 倍。带长思考的模型每轮输出几千 token 时，解码占比还会更高。缓存失效时 TTFT 会从 0.6 s 跳到好几秒（长 prefill 的算法见 [Agent 延迟优化](/agent/agent-latency)）。

### 5. 对推理系统的含义

| 负载特点 | 系统上的对策 | 见 |
|---|---|---|
| 96% 的输入是重复前缀 | 引擎侧 prefix cache，API 侧 prompt caching | [KV 复用](/agent/prefix-reuse)、[Prompt caching](/agent/prompt-caching) |
| 工具执行期间请求「暂停」 | KV 留在 HBM 一段时间（TTL），或换出到 CPU | [KV 复用](/agent/prefix-reuse) |
| 同一任务多次请求 | session 亲和路由 | [KV 复用](/agent/prefix-reuse) |
| 上下文随轮数线性增长 | token 预算、截断、压缩、sub-agent | [上下文管理](/agent/context-management) |
| 时间大头在解码和工具 | 并行工具调用、投机执行、缩短输出 | [Agent 延迟优化](/agent/agent-latency) |

## 面试追问

::: details Q：为什么说 agent 的总输入随轮数平方增长？
第 $i$ 轮的输入 $L_i = S + (i-1)a$ 随轮数线性增长，每轮都要把它整个发一次，总量是等差数列求和 $TS + aT(T-1)/2$，第二项是 $T^2$ 量级。$T = 50$、$a = 1.5\text{K}$ 时这一项是 184 万，是固定前缀那一项（50 万）的 3.7 倍。
:::

::: details Q：agent 负载的输入输出比为什么这么高？
每轮输出只有一个工具调用加几句话（几百 token），输入却是整个历史；而且工具结果（文件内容、搜索结果、命令输出）作为输入进来，往往比模型自己写的长得多。本页例子算出 156 : 1，Manus 实测约 100 : 1。聊天每轮回答几百 token、历史短，大约是个位数比一。
:::

::: details Q：一个 agent 任务的钱和时间分别花在哪？
钱花在输入：不开缓存时输入占账单 97%（本页例子），所以缓存命中率是最重要的成本指标。时间花在解码和工具：每轮 300 个输出 token 按 20 ms/token 就是 6 s，远大于命中缓存时 0.6 s 的 TTFT。优化成本先看缓存，优化时延先看输出长度、轮数和工具。
:::

::: details Q：为什么 agent 场景下「请求结束后把 KV 留一会儿」比聊天更值？
两轮间隔短：工具执行均值约 1–2 s（Continuum 测的 SWE-bench、BFCL），而聊天要等人读完再打字，几十秒到几分钟。KV 留几秒钟大概率能等到下一轮，命中就省掉整段上下文的重算；聊天里留几分钟的显存代价大，命中还不一定。
:::

::: details Q：只看 token 总数评估 agent 成本有什么问题？
同样的 token 数，命中缓存的按 0.1 倍计费，写入的按 1.25 倍，没命中的按 1 倍，价格能差 12 倍。要分开统计 cache read、cache write、uncached input、output 四类。比如本页例子总输入 234 万，不开缓存 7.01 USD，全命中只要 0.99 USD（输入部分）。
:::

## 手撕

按本页的记号算一个任务的 token 和账单，比较开、关缓存：

```python
def task_cost(T=50, S=10_000, o=300, r=1_200, p_in=3.0, p_out=15.0,
              read=0.1, write=1.25, cache=True):
    a = o + r
    inp = [S + (i - 1) * a for i in range(1, T + 1)]     # L_i
    I, O = sum(inp), T * o
    if not cache:
        usd = (I * p_in + O * p_out) / 1e6
        return I, O, usd
    new = S + (T - 1) * a                                # 第一次出现的 token，要写入缓存
    usd = ((I - new) * p_in * read + new * p_in * write + O * p_out) / 1e6
    return I, O, usd

print(task_cost(cache=False))   # (2337500, 15000, 7.2375)
print(task_cost(cache=True))    # (2337500, 15000, 1.2143...)
```

改 `T` 看平方增长：`T=100` 时总输入 $100 \times 10\text{K} + 1.5\text{K} \times 4950 = 842.5$ 万，是 `T=50` 的 3.6 倍。

## 参考

- [Manus: Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)：KV-cache 命中率是最重要的指标、输入 : 输出约 100 : 1、每任务约 50 次工具调用
- [Continuum (arXiv 2511.02230)](https://arxiv.org/abs/2511.02230)：表 1 的 SWE-bench / BFCL 轮数、工具时长、每任务 token 数
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：agent 约 4 倍、多 agent 约 15 倍于聊天的 token，token 用量解释 80% 的表现差异
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)：agent 是「LLM 根据环境反馈在循环里使用工具」
- [Anthropic prompt caching 文档](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)：缓存读写的价格倍数（2026-10-13 读取）
- 姊妹站 [指标与 Benchmark：TTFT / TPOT](https://scatyf3.github.io/ai-infra-inferview/inference/metrics-benchmark)：TTFT、TPOT 的定义
