---
title: Function calling 与约束解码：格式、解析、token mask 与失败恢复
status: draft
tags: [function-calling, tool-use, constrained-decoding, xgrammar, structured-output]
difficulty: 3
order: 5
related: [/agent/tool-protocols, /agent/agent-latency, /agent/prompt-caching]
agent: [ag-t-call, ag-m-decode]
---

# Function calling 与约束解码：格式、解析、token mask 与失败恢复

> 模型只会吐 token，「调一个工具」是一段按约定格式写出的文本；这页讲格式长什么样、服务端怎么把它解析出来、怎么从源头保证它合法、以及不合法时怎么办

## 一句话结论

工具调用在模型输出里就是一段文本：开源模型多是 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` 这类标签包 JSON，或专门的特殊 token；API 把它解析成结构化的 `tool_use` / `tool_calls` 块返回。服务端用按模型定制的 parser 从生成文本里抽出调用，自由生成时参数偶尔不合法（vLLM 文档原话）。约束解码从源头解决：每一步按 grammar 算一张 token mask，把不合法 token 的 logit 设成 $-\infty$。朴素做法要对 12.8 万个 token 逐个检查，每步几十毫秒，比一次解码还慢；XGrammar 把 99% 以上的 token 预先按自动机状态算好，运行时只检查约 1,134 个（Llama-3.1、JSON），每个 token 的 mask 不到 40 µs，和 GPU 前向重叠后几乎零开销。约束只保证语法和 schema，保证不了语义：单次调用 1% 的失败率，50 次调用的任务里至少失败一次的概率是 39.5%，所以错误要作为工具结果回给模型，让它自己改。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $V$ | 词表大小（token 数） | 128,256（Llama 3） |
| $c$ | 检查一个 token 能否被 grammar 接受的时间 | 0.5 µs（假设） |
| $n_{\text{dep}}$ | 运行时必须检查的 token 数（context-dependent） | 1,134 → 120（XGrammar 报告） |
| $t_{\text{step}}$ | 一个解码步的 GPU 时间 | 15 ms（假设） |
| $B$ | 同一步里要算 mask 的序列数（batch） | 64（假设） |
| $p$ | 单次工具调用失败（解析或执行出错）的概率 | 1%（假设） |
| $K$ | 一个任务的工具调用次数 | 50 |

µs = $10^{-6}$ s，ms = $10^{-3}$ s。

### 1. 一次调用长什么样

**API 层**（应用看到的）：

1. Anthropic：响应里有 `tool_use` 块（`id`、`name`、`input`），`stop_reason` 是 `"tool_use"`；应用执行后，在下一条 user 消息里放 `tool_result` 块（`tool_use_id`、`content`，出错时 `is_error: true`）。
2. OpenAI：响应里有 `tool_calls`，每个带 `id` 和 `function.name`、`function.arguments`（一个 JSON 字符串）；结果用 `role: "tool"` 的消息带 `tool_call_id` 回填。
3. 工具定义也是输入 token：Anthropic 在有工具时额外注入一段工具使用的系统提示，按模型不同 264–675 token（`tool_choice` 为 auto 时；2026-10-13 读取），再加上每个工具的名字、描述和 JSON schema。

**模型层**（模型真正输出的 token）：chat template 把工具定义渲染进系统提示，模型按训练时见过的格式输出调用。

| 格式 | 例子 | 用的模型（vLLM parser 名） |
|---|---|---|
| 标签包 JSON | `<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>` | Hermes、Qwen2.5（`hermes`） |
| Python 调用 | `[get_weather(city="Paris")]` | Llama 4 等（`pythonic`） |
| 特殊 token 分段 | 专用 token 标出「工具调用区开始 / 结束」，中间是名字和 JSON | DeepSeek-V3、Kimi K2（`deepseek_v3`、`kimi_k2`） |

格式的取舍：JSON 便于校验，但把代码塞进 JSON 字符串要转义换行和引号，模型容易写错。Anthropic 的 Building effective agents 建议工具格式贴近模型在网上自然见过的文本，避免这类格式负担。

### 2. 服务端怎么解析

以 vLLM 为例，有两种模式：

1. **自由生成 + 事后解析**（`tool_choice="auto"`，开 `--enable-auto-tool-choice` 和 `--tool-call-parser`）：模型可以先写一段文字再调用，parser 在生成的文本里找起始标记、抽出 JSON。不做约束，文档原话是「参数偶尔可能格式错误」。流式时 parser 增量解析，把参数一段段作为 delta 推给客户端。
2. **约束生成**（指定了工具名，或 `tool_choice="required"`）：vLLM 用 structured outputs 按工具的参数 schema 约束解码，保证能解析。文档同时提醒：保证的是「可解析」，不是「调用得好」。

OpenAI 的 `strict: true` 是同一思路：保证参数符合 schema，要求每个对象 `additionalProperties: false`、所有字段都列为 required（可选字段用加 `null` 类型表示）。

### 3. 约束解码：token mask 的成本

每个解码步：

1. grammar（JSON schema 编译成的下推自动机，PDA）处在某个状态。
2. 算出这个状态下哪些 token 能被接受，得到长度 $V$ 的 mask。
3. 不合法 token 的 logit 设成 $-\infty$，再采样。
4. 把采到的 token 的字节喂给自动机，推进到下一个状态。

**朴素做法**：对词表里每个 token 都模拟一遍自动机。一个 token 是一串字节，能不能被接受取决于当前状态和栈：

$$
t_{\text{mask}} = V \cdot c = 128{,}256 \times 0.5\ \mu\text{s} \approx 64\ \text{ms}
$$

比一个解码步（15 ms，假设）还慢 4 倍；batch 里 64 条序列都要约束时，每步要算 64 张 mask，CPU 根本跟不上。

**XGrammar 的做法**：把 token 分成两类。

1. **context-independent**：只看自动机当前节点（不看栈）就能判断接受与否。这类 token 对每个节点预先算好，存进 adaptive token mask cache。XGrammar 博客说通常超过 99% 的 token 属于这一类。
2. **context-dependent**：要看栈才能判断（比如 `}` 能不能出现，取决于嵌套到第几层）。只有这些在运行时检查，用持久化栈加速。论文报告 Llama-3.1 的 JSON grammar 只有 1,134 个（不到 1%），再做 context expansion 降到 120 个。

运行时检查量从 128,256 降到约 120，按同样的 $c$：$120 \times 0.5\ \mu\text{s} = 60\ \mu\text{s}$，和论文实测的量级一致：每个 token 的 mask 生成 JSON schema 和 JSON CFG 不到 40 µs，XML 和 Python DSL 不到 200 µs。

其他细节：

1. **存储**：预计算的 mask 按节点自适应存（存接受的或拒绝的那一边，取少的），Llama-3.1 的 JSON grammar 从 160 MB 降到 0.46 MB（论文）。
2. **传输**：mask 是 $V$ 个 bit，$128{,}256 / 8 \approx 16$ KB 一条序列，64 条约 1 MB，传到 GPU 可以忽略。
3. **重叠**：mask 在 CPU 上算，和 GPU 的前向同时进行，前向算完 mask 也好了。
4. **端到端**：XGrammar 博客报告在 H100 上 JSON schema 生成比现有引擎快最多 14 倍、CFG 最多 80 倍；SGLang 集成后 JSON 解码比其他开源方案快最多 10 倍（作者自测）。

**跳跃解码**（SGLang 的 compressed FSM）：grammar 在某处只允许唯一的一串字符时（比如固定的 key `"arguments": `），不用一个 token 一个 token 解码，直接把这串 token 追加上去，在一次前向里算完它们的 KV。

**别把思考也约束死**：mask 只允许语法上合法的 token，如果整个输出都被 JSON 约束，模型就没有地方先推理。做法是让模型先自由输出，遇到工具调用的起始标记再启用 grammar。

### 4. 解析失败与恢复

失败分三类，约束解码只能消灭前两类：

| 类型 | 例子 | 约束解码能防吗 |
|---|---|---|
| 语法 | JSON 不闭合、被 `max_tokens` 截断 | 能（截断除外） |
| schema | 少了必填字段、类型错、工具名不存在 | 能 |
| 语义 | 日期写成过去、文件路径不存在、参数组合不对 | 不能 |

**为什么必须有恢复路径**：单次失败概率 $p$，一个任务 $K$ 次调用，至少失败一次的概率

$$
P = 1 - (1-p)^K
$$

$p = 1\%$、$K = 50$：$1 - 0.99^{50} = 39.5\%$；$p = 2\%$ 时 63.6%。长任务里失败是常态。

**恢复的标准做法**：

1. 应用侧先按 schema 校验参数，不合法就不执行。
2. 把错误作为工具结果回给模型（Anthropic 的 `is_error: true`，MCP 的 `isError: true`），写清楚错在哪：MCP 规范把这类错误叫 tool execution error，意思是模型能据此自己改正后重试；未知工具这类协议错误模型较难自救。
3. 设重试上限（比如同一个调用连续失败 3 次就换策略或终止），否则模型可能反复犯同一个错。
4. `stop_reason` 是 `max_tokens` 时，说明调用被截断了，放宽输出上限重来，而不是当成语法错误回给模型。
5. 每次重试是多一轮：一次 prefill（多半命中缓存）加一次解码，时延算在 [Agent 延迟](/agent/agent-latency) 里。

Manus 的经验是把失败的动作和报错留在上下文里，模型看到之后不容易再犯同样的错。

**怎么评测**：Berkeley Function Calling Leaderboard（BFCL）用 AST 比对判断调用是否正确（v1），v3 加入多轮交互，v4 做整体的 agent 评测，同时报告跑完整个 benchmark 的成本和时延。

## 面试追问

::: details Q：约束解码为什么不能直接对整个词表逐个检查？
一个 token 能否被接受要拿它的每个字节去跑 grammar 的自动机，词表 12.8 万，按每个 0.5 µs 算一步要约 64 ms，比一个解码步还长；batch 里每条序列都要一张 mask。XGrammar 先对每个自动机节点预计算「只看节点就能定」的 token（超过 99%），运行时只查需要看栈的那 1% 以下，每 token 不到 40 µs。
:::

::: details Q：XGrammar 里 context-dependent token 是什么？举个例子。
能否接受取决于下推自动机的栈、而不只是当前节点的 token。比如 `}`：在 JSON 里它是否合法取决于当前嵌套在哪一层对象里，要看栈。相反，在字符串内部的普通字母只看「当前在字符串里」这个节点就能判断，属于 context-independent。
:::

::: details Q：用了约束解码还会出错吗？
会。约束只保证语法和 schema：JSON 闭合、字段齐全、类型正确。语义错误（参数值不对、文件不存在、日期在过去）照样发生；输出被 `max_tokens` 截断时也会不完整。另外约束过严会伤效果：整段输出都被 JSON 约束，模型就没有地方先推理，所以通常只在工具调用的起始标记之后启用 grammar。
:::

::: details Q：工具调用解析失败，服务端应该怎么处理？
不要直接抛异常终止任务。1. 先按 schema 校验，不合法不执行。2. 把具体的错误信息作为工具结果（`is_error`）回给模型，让它下一轮改正。3. 设重试上限，连续失败换策略或终止。4. 被 `max_tokens` 截断的情况单独处理，放宽上限重来。单次 1% 的失败率在 50 次调用的任务里有 39.5% 的概率至少碰到一次，所以这是常规路径。
:::

::: details Q：tool_choice 的 auto 和 required 在服务端实现上有什么区别？
以 vLLM 为例：auto 时模型自由生成，可能先说话再调用，parser 事后从文本里抽调用，不保证参数合法；required 或指定工具名时，服务端按工具参数的 schema 做约束解码，保证能解析出至少一个调用。代价是约束模式下没有自由文本，模型不能先推理。
:::

## 手撕

最小的约束解码：合法输出是一组工具名，每一步只允许「拼上之后仍是某个合法名字前缀」的 token。`build_mask` 就是朴素的 $V$ 次检查，正是 XGrammar 要预计算掉的部分。

```python
import math

def build_mask(vocab, allowed, prefix):
    # token t 合法 <=> prefix + t 仍是某个合法输出的前缀；朴素做法，每步扫整个词表
    return [any(a.startswith(prefix + t) for a in allowed) for t in vocab]

def constrained_greedy(logits, vocab, allowed):
    out = ""
    while out not in allowed:                    # allowed 里没有谁是别人的前缀，到达即结束
        mask = build_mask(vocab, allowed, out)
        masked = [l if ok else -math.inf for l, ok in zip(logits(out), mask)]
        out += vocab[max(range(len(vocab)), key=masked.__getitem__)]
    return out

vocab = ["}", "get", "_", "weather", "time", "search", "sea", "rch"]
score = {"}": 5, "time": 4, "weather": 3.5, "get": 3, "_": 2, "search": 1, "sea": 0.5, "rch": 0.5}
logits = lambda prefix: [score[t] for t in vocab]   # 假装模型最想输出 "}"
allowed = ["get_weather", "get_time", "search"]

print(vocab[max(range(len(vocab)), key=lambda i: logits("")[i])])  # 不约束：}
print(constrained_greedy(logits, vocab, allowed))                    # 约束后：get_time
```

## 参考

- [vLLM tool calling 文档](https://docs.vllm.ai/en/latest/features/tool_calling.html)：auto 模式事后解析、参数可能不合法；named / required 用 structured outputs；各模型的 parser
- [OpenAI function calling 文档](https://developers.openai.com/api/docs/guides/function-calling)：`strict: true` 的要求、工具定义计入输入 token、`parallel_tool_calls`
- [Anthropic tool use 文档](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)：`tool_use` / `tool_result` 往返、工具使用系统提示的 token 数（2026-10-13 读取）
- [XGrammar (arXiv 2411.15100)](https://arxiv.org/abs/2411.15100)：context-independent / dependent token、持久化栈、每 token 不到 40 µs、1,134 → 120、160 MB → 0.46 MB
- [XGrammar 博客](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)：超过 99% 的 token 可预计算、端到端 14 倍 / 80 倍
- [SGLang (arXiv 2312.07104)](https://arxiv.org/abs/2312.07104)、[SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4)：compressed FSM 跳跃解码、集成 XGrammar 后 JSON 解码最多快 10 倍
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)：工具格式贴近自然文本、避免格式负担
- [MCP 规范：Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)：protocol error 与 tool execution error（`isError`）的区分
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)：AST 评测、多轮、agentic 评测、成本与时延
- [Manus: Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)：把错误留在上下文里
