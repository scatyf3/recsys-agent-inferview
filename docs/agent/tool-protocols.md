---
title: MCP 与工具协议：架构、传输、工具定义的 token 账、并行调用
status: draft
tags: [mcp, tool-use, tool-search, parallel-tool-calls, json-rpc]
difficulty: 2
order: 6
related: [/agent/function-calling, /agent/prompt-caching, /agent/context-management, /agent/agent-latency, /agent/sandbox]
agent: [ag-t-mcp, ag-t-parallel]
---

# MCP 与工具协议：架构、传输、工具定义的 token 账、并行调用

> 工具从哪来（协议）、工具定义占多少上下文（token 账）、一轮能调几个（并行）

## 一句话结论

MCP 用 JSON-RPC 把「提供工具的 server」和「用工具的应用」解耦：应用（host）里每个 client 连一个 server，通过 `tools/list` 发现工具、`tools/call` 调用；传输有两种，本地子进程的 stdio 和远程的 Streamable HTTP（一个端点、每条消息一个 POST，回复是 JSON 或 SSE 流）。2026-07-28 版规范去掉了协议层的 session，每个请求自带版本和能力信息，server 可以无状态水平扩展。系统上最大的代价是工具定义的 token：Anthropic 举的例子 5 个 server、58 个工具约 55K token，在 50 轮的任务里即使全命中缓存也要约 34 万个基础价 token，和任务其余输入的花费相当，还占掉 200K 窗口的 27%。办法是按需加载（tool search，前缀里只放约 500 token 的搜索工具，找到的工具追加在后面，不破坏缓存）、用屏蔽代替删工具、把多步调用写成代码。互不依赖的调用要在一轮里并行发出：等待从求和变成取最大值，还省掉中间几轮模型调用，例子里 11.7 s 变 3 s。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $D$ | 工具定义占的 token（名字、描述、schema） | 55K（58 个工具） |
| $T$ | 任务轮数 | 50 |
| $r, w$ | 缓存读取、写入价相对基础输入价的倍数 | 0.1、1.25 |
| $K$ | 一步里要调的工具数 | 3 |
| $t_k$ | 第 $k$ 个工具的执行时间 | 2、1.5、3 s |
| $t_{\text{turn}}$ | 多一轮模型调用的时间（TTFT + 解码） | 2.6 s（假设） |

成本单位「基础价 token」同 [上下文管理](/agent/context-management)：按基础输入价计费的 1 个 token 记 1。

### 1. MCP 的架构

三个角色（规范原文：hosts、clients、servers）：

| 角色 | 是什么 | 例子 |
|---|---|---|
| host | 发起连接的 LLM 应用，管理多个 client，决定给模型看哪些工具、执行前要不要问用户 | Claude Code、IDE 插件 |
| client | host 里的连接器，和一个 server 一对一 | 每个 server 一个 client 实例 |
| server | 提供能力的服务 | GitHub、数据库、浏览器 |

server 提供三种东西：**tools**（模型决定调用的函数）、**resources**（给模型或用户的数据）、**prompts**（给用户用的模板）。消息是 JSON-RPC 2.0：

```text
→ {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
← {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "get_weather",
     "description": "...", "inputSchema": {"type": "object", ...}}]}}
→ {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
   "params": {"name": "get_weather", "arguments": {"location": "Paris"}}}
← {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "..."}], "isError": false}}
```

工具列表变化时 server 发 `notifications/tools/list_changed`。规范建议 server 按确定的顺序返回工具列表，理由是让 client 能缓存列表、提高 LLM 的 prompt cache 命中率（见 [Prompt caching 的账](/agent/prompt-caching)）。

**两种传输**：

| | stdio | Streamable HTTP |
|---|---|---|
| 怎么连 | client 把 server 作为子进程启动 | server 是独立进程，一个 HTTP 端点（如 `/mcp`） |
| 消息 | stdin / stdout 上按行分隔的 JSON-RPC，stderr 打日志 | 每条消息一个 POST；回复是一个 JSON，或一个 SSE 流（中间可以推进度） |
| 适合 | 本地工具：读写文件、跑命令 | 远程、多用户的服务 |
| 要注意 | stdout 只能写合法 MCP 消息 | 鉴权、校验 `Origin` 头防 DNS rebinding、本地只绑 127.0.0.1 |

**版本演进和无状态**：2024-11-05 版远程用 HTTP+SSE（两个端点）；2025-03-26 换成 Streamable HTTP；2025-06-18 版有 `initialize` 握手和 `Mcp-Session-Id` 头，负载均衡要按 session 粘到同一实例。2026-07-28 版去掉了协议层 session：每个请求在 `_meta` 里带协议版本和 client 能力，server 不主动发请求。于是任何实例都能处理任何请求，server 可以像普通无状态服务一样扩容。确实需要状态的工具（购物车、打开的浏览器）返回一个显式的 handle，之后的调用把 handle 当参数传回来。

**两类错误**：协议错误（未知工具、请求格式错）走 JSON-RPC 的 `error`，模型很难自己修；工具执行错误（API 失败、参数值不合法）放在结果里 `isError: true`，应该交给模型让它改正重试（见 [Function calling](/agent/function-calling)）。

**安全**：工具等于任意代码执行，规范要求 host 在调用前取得用户同意，server 声称的工具行为注解默认不可信。隔离执行见 [沙箱](/agent/sandbox)。

### 2. 工具定义占多少 token

每个工具的名字、描述、参数 schema 都会渲染进系统提示，每轮都在输入里。Anthropic 的 advanced tool use 文章给的例子：

| server | 工具数 | token | 平均每个工具 |
|---|---|---|---|
| GitHub | 35 | ~26K | ~740 |
| Slack | 11 | ~21K | ~1,900 |
| Sentry | 5 | ~3K | ~600 |
| Grafana | 5 | ~3K | ~600 |
| Splunk | 2 | ~2K | ~1,000 |
| **合计** | **58** | **~55K** | |

单个 Jira server 约 17K；他们见过优化前工具定义达到 134K 的情况。

**一个任务的账**：$D = 55\text{K}$ 放在前缀里跑 $T = 50$ 轮，第一轮写入、之后每轮读取：

$$
\text{cost} = D\,\bigl(w + (T-1)\,r\bigr) = 55\text{K} \times (1.25 + 49 \times 0.1) = 338\text{K}
$$

按 3 USD / 百万 token 约 1.01 USD。对比 [负载画像](/agent/agent-workload) 的例子，任务其余输入全命中缓存时约 0.99 USD：工具定义让输入账单翻倍。不开缓存是 $55\text{K} \times 50 = 2.75\text{M}$，8.25 USD。另外 55K 占了 200K 窗口的 27.5%。

工具多了也更容易选错：OpenAI 的建议是一轮开始时可用的函数最好少于 20 个（软建议）；Anthropic 报告用工具检索后，MCP 评测的准确率 Opus 4 从 49% 到 74%、Opus 4.5 从 79.5% 到 88.1%（作者自测）。

### 3. 工具多了怎么办

**3.1 按需加载（tool search）**。前缀里只放一个「搜索工具」的工具，其余工具标成延迟加载（Anthropic 的 `defer_loading: true`），模型需要时先搜，搜到的工具定义再进上下文。Anthropic 报告：前缀约 500 token，找到 3–5 个工具约 3K，总共约 8.7K，对比全部加载的约 77K，少 85%。常用的几个工具仍然直接加载。Claude Code 默认开启 tool search。

关键是**它不破坏缓存**：延迟加载的工具一开始根本不在前缀里，搜到以后是追加在后面的，前缀没变。

**3.2 不要每轮换工具**。「每轮按当前步骤检索 top-k 工具放进 prompt」看起来省 token，但工具定义在前缀最前面，换一次整个缓存失效，每轮都按 1.25 倍重写（时间戳的例子在 [Prompt caching](/agent/prompt-caching) 里是 7.01 → 8.77 USD）。历史里还会有对已经消失的工具的调用，模型会困惑。两个合规的做法：

1. 任务开始时选定工具集，整个任务不变；或者只追加（3.1）。
2. 工具全部保留，用解码时屏蔽来限制本轮能选哪些（Manus 的 logits masking，工具名用统一前缀如 `browser_`、`shell_` 方便成组屏蔽），或 API 的 allowed tools 参数。

**3.3 把多步调用写成代码**。让模型写一段代码去调多个工具，中间结果留在执行环境里，只把最终结果放回上下文。Anthropic 的 programmatic tool calling：复杂研究任务平均 token 从 43,588 降到 27,297（少 37%）；一个预算核对的例子把 200KB 的原始数据变成 1KB 的结果；一段代码里编排 20 多次调用，省掉 19 次以上的模型推理（作者自测）。代价是需要一个安全的代码执行环境。

**3.4 少而精**。总是一起调用的函数合并成一个；能由应用填的参数（比如已知的 `order_id`）不要让模型填；描述写清楚但别冗长；工具返回精简的结果。

### 4. 并行工具调用

模型在一轮输出里发出多个调用（Anthropic：多个 `tool_use` 块，结果在下一条消息里一起回填；OpenAI：多个 `tool_calls`）。两家都可以关掉：Anthropic 的 `disable_parallel_tool_use: true`，OpenAI 的 `parallel_tool_calls: false`。

**时延**：$K$ 个互不依赖的调用，串行要等 $K$ 次工具、多跑 $K-1$ 轮模型；并行只等最慢的一个：

$$
t_{\text{serial}} = \sum_{k=1}^{K} t_k + (K-1)\,t_{\text{turn}},\qquad
t_{\text{parallel}} = \max_k t_k
$$

例子：三个读操作 2、1.5、3 s，多一轮模型调用 $t_{\text{turn}} = 0.6 + 100 \times 0.02 = 2.6$ s（TTFT 加 100 个输出 token，假设）：

1. 串行：$6.5 + 2 \times 2.6 = 11.7$ s。
2. 并行：3 s。
3. 钱：省掉的 2 轮每轮都要读一遍上下文，上下文 50K 时省 $2 \times 50\text{K} \times 0.1 = 10\text{K}$ 个基础价 token。

**有依赖时**：LLMCompiler 让模型先输出一张调用的依赖图（后面的调用用占位符引用前面的结果），调度器在依赖满足时立即派发，执行器并行跑。对比逐步调用的 ReAct：时延最多快 3.7 倍、成本最多省 6.7 倍、准确率最多高约 9%（作者自测）。Anthropic 的多 agent 研究系统里，主 agent 并行派出 3–5 个 sub-agent，每个 sub-agent 并行调 3 个以上工具，复杂查询的研究时间最多缩短 90%。

**注意事项**：

1. 有副作用的调用（写同一个文件、改同一条记录）不能并行，或者要定好顺序。
2. 工具服务有限流，一次并发太多会被拒。
3. 每个结果按调用 id 对上；一个失败只回填那一个的错误，其他照常。

## 面试追问

::: details Q：MCP 里 host、client、server 分别是什么？为什么 client 和 server 一对一？
host 是 LLM 应用（比如 Claude Code），负责管理连接、决定把哪些工具给模型、调用前问用户；client 是 host 里的连接器；server 提供工具、资源、提示模板。一个 client 只连一个 server，隔离边界清楚：每个 server 的权限、生命周期、故障互不影响，host 统一做汇总和授权。
:::

::: details Q：stdio 和 Streamable HTTP 怎么选？
stdio 是 client 把 server 当子进程启动、走标准输入输出，适合本地工具（文件、命令），没有网络和鉴权问题。Streamable HTTP 是一个 HTTP 端点、每条消息一个 POST、回复可以是 JSON 或 SSE 流，适合远程、多用户的服务，要做鉴权和 Origin 校验。新版规范去掉了协议层 session，HTTP server 可以无状态扩容。
:::

::: details Q：接了几十个工具，上下文和成本怎么控制？
先算账：58 个工具约 55K token，50 轮任务即使全命中缓存也要 $55\text{K} \times (1.25 + 49 \times 0.1) = 338\text{K}$ 个基础价 token。办法：1. tool search，前缀只放搜索工具，找到的工具追加进来，不破坏缓存；2. 任务开始时选定工具集，中途不换；3. 需要限制时用解码屏蔽或 allowed tools，不删定义；4. 多步调用写成代码执行，中间结果不进上下文。
:::

::: details Q：为什么不每轮按语义检索 top-k 个工具放进 prompt？
工具定义在缓存前缀的最前面，每轮换一次就整个前缀失效，每轮都按写入价（1.25 倍）重付全部上下文，比不开缓存还贵。而且历史里调用过的工具突然不见了，模型会出错。检索可以做，但结果要追加在后面（tool search 的做法），或者只在任务开始时做一次。
:::

::: details Q：并行工具调用能省多少？有什么限制？
时延从 $\sum t_k + (K-1)t_{\text{turn}}$ 变成 $\max t_k$：三个 2、1.5、3 s 的调用，加上省掉的两轮模型调用，11.7 s 变 3 s；省掉的轮次也不用再读一遍上下文。限制：只能并行互不依赖、没有冲突副作用的调用；工具服务有并发上限；有依赖的要先规划成 DAG（LLMCompiler）再按依赖调度。
:::

## 参考

- [MCP 规范（2026-07-28）](https://modelcontextprotocol.io/specification/latest)：host / client / server、JSON-RPC、tools / resources / prompts
- [MCP 规范：Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)：`tools/list`、`tools/call`、确定的工具顺序、两类错误、显式 handle
- [MCP 规范：Transports（2026-07-28）](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)：stdio 与 Streamable HTTP、请求元数据放在 `_meta`、旧版的 session
- [MCP 规范：Transports（2025-06-18）](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)：`Mcp-Session-Id`、取代 HTTP+SSE、Origin 校验
- [Anthropic: Introducing advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)：58 个工具约 55K token、tool search 少 85% token 与准确率提升、programmatic tool calling 少 37% token
- [OpenAI function calling 文档](https://developers.openai.com/api/docs/guides/function-calling)：少于 20 个函数的建议、`parallel_tool_calls`、allowed tools
- [Anthropic tool use 文档](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)：并行 `tool_use`、`disable_parallel_tool_use`
- [LLMCompiler (arXiv 2312.04511)](https://arxiv.org/abs/2312.04511)：planner / task fetching unit / executor，3.7 倍时延、6.7 倍成本
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：并行 sub-agent 和并行工具，研究时间最多缩短 90%
- [Manus: Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)：屏蔽而不是删除工具、统一的工具名前缀
- [Claude Code MCP 文档](https://code.claude.com/docs/en/mcp)：tool search 默认开启
