---
title: Agent 延迟优化：端到端时延怎么拆、长 prefill、并行与投机执行、流式早启动
status: draft
tags: [latency, ttft, prefill, speculative-execution, streaming, agent]
difficulty: 3
order: 7
related: [/agent/agent-workload, /agent/prefix-reuse, /agent/tool-protocols, /agent/function-calling, /agent/llm-gateway]
agent: [ag-m-ttft]
---

# Agent 延迟优化：端到端时延怎么拆、长 prefill、并行与投机执行、流式早启动

> 一个任务几十轮，每轮的首 token、解码、工具都被放大几十倍；先把时间拆到每一项，再看每个手段砍的是哪一项

## 一句话结论

任务时延 $= \sum_i (\text{排队} + \text{TTFT}_i + o_i \cdot \text{TPOT} + t_{\text{tool},i})$。[负载画像](/agent/agent-workload) 的例子里 50 轮共 405 s，解码占 74%、工具 19%、TTFT 7%。TTFT 小是因为前缀命中：70B 模型在 8 张 H100 上，83.5K 的上下文全部重算约 6.6 s，命中后只算新追加的 1.5K 约 0.17 s（假设 MFU 40%），差 39 倍，所以缓存失效首先是时延事故。砍时间的顺序：**少输出**（每轮少说 150 token，50 轮省 150 s）、**少轮数**（并行工具调用、把多步写成代码）、**让工具和生成重叠**（流式解析到一个调用闭合就开跑；投机执行，Speculative Actions 下一步预测准确率最高 55%、时延最多降 20%，PASTE 平均任务完成时间降 43.5%）、**保住缓存命中**（亲和路由、TTL）。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $T$ | 轮数 | 50 |
| $o$ | 每轮输出 token | 300 |
| $t_{\text{tpot}}$ | 每个输出 token 的时间 | 20 ms（假设） |
| $t_{\text{ttft}}$ | 首 token 延迟 | 0.6 s（命中时，假设） |
| $t_{\text{tool}}$ | 工具执行时间 | 1.5 s（假设） |
| $N$ | 模型参数量 | $70.6\times10^9$（Llama-3.1-70B） |
| $\ell, d$ | 层数、hidden 维 | 80、8192 |
| $L$ | 已有上下文长度（token） | 83.5K |
| $x$ | 这一轮要新算的 token 数 | 1.5K（命中）或 83.5K（失效） |
| $F$ | 集群有效算力 | $8 \times 989 \times 40\% = 3.17$ PFLOP/s（假设 MFU 40%） |
| $p$ | 投机执行时下一个工具调用猜对的概率 | 0.5（假设） |

### 1. 端到端时延怎么拆

$$
t_{\text{task}} = \sum_{i=1}^{T}\Bigl(t_{\text{queue},i} + t_{\text{ttft},i} + o_i\,t_{\text{tpot}} + t_{\text{tool},i}\Bigr) + t_{\text{harness}}
$$

$t_{\text{harness}}$ 是应用自己的开销（拼上下文、校验、网络往返），通常不大但要测。例子（不计排队和 harness）：

| 部分 | 每轮 | 50 轮 | 占比 |
|---|---|---|---|
| TTFT | 0.6 s | 30 s | 7% |
| 解码 300 token | 6.0 s | 300 s | 74% |
| 工具 | 1.5 s | 75 s | 19% |
| 合计 | 8.1 s | 405 s | |

先测再优化：给每一轮打 trace（[评测与可观测](/agent/eval-observability)），按这四项分别统计 p50 / p99，看大头在哪。带长思考的模型 $o$ 是几千，解码占比更高；跑测试、编译的 coding agent，工具占比更高。

### 2. TTFT 与长 prefill

TTFT 主要是这一轮要新算的 $x$ 个 token 的 prefill。FLOPs 分两部分：线性层每 token $2N$；attention 里新 token 要和已有的 $L$ 个 token 做 $QK^\top$ 和 $PV$，每层每 token 约 $4Ld$。缓存全失效时 $x = L$，causal 下平均只看一半，attention 是 $2\ell L^2 d$。

$$
t_{\text{prefill}} \approx \frac{2N x + 4 \ell\, x L\, d}{F}\ \ (\text{命中}),\qquad
t_{\text{prefill}} \approx \frac{2N L + 2 \ell L^2 d}{F}\ \ (\text{全失效})
$$

代入 Llama-3.1-70B、8 张 H100（$F = 3.17\times10^{15}$）、$L = 83.5\text{K}$：

| | 线性层 | attention | 合计 | 时间 |
|---|---|---|---|---|
| 全失效，$x = 83.5\text{K}$ | $1.18\times10^{16}$ | $9.14\times10^{15}$ | $2.09\times10^{16}$ | 6.6 s |
| 命中，$x = 1.5\text{K}$ | $2.12\times10^{14}$ | $3.28\times10^{14}$ | $5.40\times10^{14}$ | 0.17 s |

1. 全失效的 attention：$2 \times 80 \times (8.35\times10^4)^2 \times 8192 = 9.14\times10^{15}$，已经和线性层同一量级，上下文再长它就是大头。
2. 命中时 attention：$4 \times 80 \times 1500 \times 83500 \times 8192 = 3.28\times10^{14}$，新 token 虽少，但每个都要看完整个上下文。
3. 差 39 倍。一个 50 轮的任务中途失效几次，就多出几十秒。

所以保 TTFT 的手段都是保命中：前缀不变、请求回到同一个副本、工具执行期间 KV 别被淘汰（[多轮 agent 的 KV 复用](/agent/prefix-reuse)）。另一个问题是干扰：一条长 prefill 会占住 GPU，同一批里其他请求的解码被拖慢，引擎侧用 chunked prefill 或 PD 分离来隔离，见姊妹站 [Continuous Batching、Chunked Prefill 与 PD 分离](https://scatyf3.github.io/ai-infra-inferview/inference/batching-scheduling)。

### 3. 解码：大头，先砍输出

解码时间 $= o \cdot t_{\text{tpot}}$，每轮都有、乘以轮数。

1. **少说**：很多 agent 每轮会复述计划、解释要做什么。每轮少 150 token：$150 \times 0.02 \times 50 = 150$ s，任务从 405 s 降到 255 s。提示词里要求直接调用工具、不复述，或者限制非思考部分的长度。
2. **思考长度**：带思考的模型按步骤难度给思考预算，简单步骤用小预算或换小模型（模型路由见 [LLM 网关](/agent/llm-gateway)）。
3. **投机解码**：agent 的输出里有大量可以从上下文抄的内容（改代码时复制原文、工具调用里重复文件路径），用上下文做草稿的投机解码接受率高，见姊妹站 [Speculative Decoding](https://scatyf3.github.io/ai-infra-inferview/inference/speculative-decoding)。
4. **TPOT 本身**：取决于 batch 大小和 KV 长度，长上下文的 decode 每步要读完整个 KV，见姊妹站 [Prefill vs Decode 与 Roofline](https://scatyf3.github.io/ai-infra-inferview/inference/prefill-decode-roofline)。

### 4. 少轮数：并行工具调用、写成代码

每少一轮，就少一次 TTFT、一次解码、一次上下文读取。

1. **并行工具调用**：$K$ 个互不依赖的调用一轮发出，等待从 $\sum t_k + (K-1)t_{\text{turn}}$ 变成 $\max t_k$，[MCP 与工具协议](/agent/tool-protocols) 里的例子是 11.7 s 变 3 s。
2. **把多步写成代码**：一段代码里编排 20 多次工具调用，省掉 19 次以上模型推理（Anthropic programmatic tool calling，作者自测）。
3. **sub-agent 并行**：几个 sub-agent 同时探索不同方向，Anthropic 的研究系统把复杂查询的时间最多缩短 90%（同时 token 变多，见 [上下文管理](/agent/context-management)）。

### 5. 让工具和生成重叠

**5.1 流式早启动**。不要等整个回复生成完再执行工具：流式解析输出，某个调用的 JSON 一闭合就开跑。一轮发出 $K$ 个调用、每个 $c$ 个 token 时，第 $k$ 个调用可以提前 $(K-k)\,c\,t_{\text{tpot}}$ 开始：

例子：3 个调用各 100 token，按输出顺序工具耗时 3、2、1.5 s。

1. 等全部输出完（6 s）再一起跑：$6 + \max(3, 2, 1.5) = 9$ s。
2. 闭合即启动：三个调用在 2、4、6 s 闭合，分别在 5、6、7.5 s 跑完，这一轮 7.5 s。
3. 所以让慢的工具先输出：可以在提示里要求先调用耗时长的工具，或者由应用重排。

**5.2 投机执行工具**。在模型还在生成时，用一个更快的预测器（小模型、历史模式）猜下一个工具调用并提前执行；模型真正发出的调用和猜的一致就直接用结果，不一致就丢掉。和投机解码同一个思路，只是投机的对象从 token 变成了动作。

每轮的期望节省：猜对的概率 $p$，猜对时工具和生成重叠，省下二者中较短的那段：

$$
\Delta t = p \cdot \min\left(t_{\text{tool}},\ t_{\text{gen}}\right)
$$

例子：$t_{\text{gen}} = 0.6 + 300 \times 0.02 = 6.6$ s，$t_{\text{tool}} = 1.5$ s，$p = 0.5$：每轮省 0.75 s，占 8.1 s 的 9%。工具越慢、越可预测，省得越多。

公开结果（作者自测）：

1. Speculative Actions（Columbia）：下一步动作预测准确率最高 55%，端到端时延最多降 20%。
2. PASTE：利用 agent 反复出现的工具调用模式和参数传递关系预测下一次调用，平均任务完成时间降 43.5%，工具时延降到 1/1.8。

限制：只能投机**只读或可撤销**的调用（查询、读文件、搜索），写操作、发消息、付款这类有副作用的不能提前做，或者要在隔离的环境快照上做；猜错的调用要付工具本身的成本（API 费用、限流配额）。

### 6. 手段汇总

| 手段 | 砍哪一项 | 本页例子的量级 |
|---|---|---|
| 保缓存命中（稳定前缀、亲和路由、TTL） | TTFT | 失效一次 6.6 s，命中 0.17 s |
| 每轮少输出 150 token | 解码 | 405 → 255 s |
| 并行工具调用 | 轮数、工具等待 | 一步 11.7 → 3 s |
| 写成代码执行 | 轮数 | 20 次调用省 19 次推理 |
| 流式早启动 | 工具等待 | 一轮 9 → 7.5 s |
| 投机执行工具 | 工具等待 | 每轮期望省 $p \cdot \min(t_{\text{tool}}, t_{\text{gen}})$ |
| 小模型 / 少思考处理简单步骤 | 解码、TTFT | 看路由命中 |

## 面试追问

::: details Q：agent 任务很慢，你怎么拆？
按轮打 trace，每轮拆成排队、TTFT、解码（输出 token 数 × TPOT）、工具执行、应用开销五项，聚合看占比和 p99。本页例子解码占 74%，那就先砍输出 token 和轮数；如果 TTFT 突然变大，查缓存命中率和路由；工具占比高就并行、早启动、投机执行。
:::

::: details Q：为什么缓存失效对 agent 的 TTFT 影响这么大？
命中时只算新追加的几千 token，失效时要把整个上下文重算一遍，而且 attention 部分随上下文长度平方增长。70B、8 张 H100、83.5K 上下文：全部重算约 6.6 s，命中只要约 0.17 s，差 39 倍。
:::

::: details Q：投机执行工具和投机解码有什么异同？
同：都是用便宜的预测先做、再用真正的结果验证，猜对省时间、猜错丢弃。异：投机解码猜的是 token，验证是一次并行前向，错了没有副作用；投机执行猜的是工具调用，猜错要付工具成本，有副作用的调用（写文件、发请求、付款）不能提前做。期望节省是 $p \cdot \min(t_{\text{tool}}, t_{\text{gen}})$，工具越慢、越可预测越值。
:::

::: details Q：流式输出对 agent 有什么用？
不只是给用户看进度。应用可以边收边解析，一个工具调用的 JSON 闭合就开始执行，不用等整段回复（包括后面的其他调用和说明文字）生成完。一轮发出多个调用时，越早输出的调用提前得越多，所以慢的工具应该先输出。
:::

::: details Q：减少轮数有哪些办法？各自的代价？
1. 并行工具调用：只适用于互不依赖、无冲突副作用的调用。
2. 把多步调用写成代码执行：需要安全的执行环境，代码出错要调试。
3. sub-agent 并行：总 token 变多（Anthropic 多 agent 约 15 倍于聊天）。
4. 更强的模型一次做对：单价高。每少一轮省一次 TTFT、一次解码和一次上下文读取。
:::

## 手撕

一轮输出里有 3 个工具调用，流式解析到一个闭合就开跑，对比等整段输出完再一起跑：

```python
import asyncio, time

TPOT, TOKENS_PER_CALL = 0.02, 100
CALLS = [("run_tests", 3.0), ("read_a", 2.0), ("read_b", 1.5)]   # 按输出顺序，慢的在前

async def tool(name, secs):
    await asyncio.sleep(secs)
    return name

async def stream_calls():
    for name, secs in CALLS:
        await asyncio.sleep(TOKENS_PER_CALL * TPOT)   # 生成这个调用的 token
        yield name, secs                              # 调用的 JSON 闭合

async def one_turn(early_start):
    t0, tasks, buffered = time.perf_counter(), [], []
    async for name, secs in stream_calls():
        if early_start:
            tasks.append(asyncio.create_task(tool(name, secs)))   # 闭合即启动
        else:
            buffered.append((name, secs))
    tasks += [asyncio.create_task(tool(n, s)) for n, s in buffered]  # 输出完再一起跑
    await asyncio.gather(*tasks)
    return round(time.perf_counter() - t0, 1)

print(asyncio.run(one_turn(early_start=False)))   # 9.0
print(asyncio.run(one_turn(early_start=True)))    # 7.5
```

## 参考

- [Speculative Actions (arXiv 2510.04371)](https://arxiv.org/abs/2510.04371)：用快模型预测下一步动作并提前执行，准确率最高 55%、时延最多降 20%
- [PASTE (arXiv 2603.18897)](https://arxiv.org/abs/2603.18897)：按工具调用模式投机执行，平均任务完成时间降 43.5%
- [LLMCompiler (arXiv 2312.04511)](https://arxiv.org/abs/2312.04511)：并行函数调用，最多 3.7 倍时延
- [Anthropic: Introducing advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)：programmatic tool calling 省掉多次推理
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：并行 sub-agent 与工具，时间最多缩短 90%
- [Continuum (arXiv 2511.02230)](https://arxiv.org/abs/2511.02230)：工具调用期间保留 KV 以降低下一轮延迟
- 姊妹站 [Prefill vs Decode 与 Roofline](https://scatyf3.github.io/ai-infra-inferview/inference/prefill-decode-roofline)、[Continuous Batching、Chunked Prefill 与 PD 分离](https://scatyf3.github.io/ai-infra-inferview/inference/batching-scheduling)、[Speculative Decoding](https://scatyf3.github.io/ai-infra-inferview/inference/speculative-decoding)、[指标与 Benchmark](https://scatyf3.github.io/ai-infra-inferview/inference/metrics-benchmark)
