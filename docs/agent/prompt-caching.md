---
title: Prompt caching 的账：价格倍数、TTL、命中率与上下文排布
status: draft
tags: [prompt-caching, cost, ttl, cache-hit-rate, agent]
difficulty: 2
order: 3
related: [/agent/agent-workload, /agent/prefix-reuse, /agent/context-management, /agent/tool-protocols]
agent: [ag-c-cache]
---

# Prompt caching 的账：价格倍数、TTL、命中率与上下文排布

> API 侧的前缀缓存怎么计费、一个 agent 任务能省多少、哪些写法会让它失效

## 一句话结论

prompt caching 是推理引擎 prefix cache 在账单上的样子：前缀和之前某次请求 token 级相同，这部分就按「缓存读取价」计费。以 2026-10-13 读到的文档为准，Anthropic 缓存读取是基础输入价的 0.1 倍（部分最新模型更低），写入 5 分钟缓存 1.25 倍、1 小时缓存 2 倍，每次命中刷新 TTL；OpenAI 自动缓存 1024 token 以上的前缀，读取同样是 0.1 倍。一轮的有效输入单价是 $h \cdot 0.1 + (1-h)\cdot 1.25$ 倍（$h$ 为命中率），agent 任务 $h$ 能到 96%，输入成本降到约 1/7。5 分钟缓存只要被读一次就回本，1 小时缓存要读两次。缓存失效的代价比不开缓存还贵：开头放个时间戳，每轮都按 1.25 倍重写，本页例子从 7.01 USD 变成 8.77 USD。规则只有一条：**稳定的放前面，变化的追加在后面，前缀一个 token 都不要改**。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $p$ | 基础输入价格（USD / 百万 token） | 3（假设） |
| $w$ | 缓存写入价相对 $p$ 的倍数 | 1.25（5 分钟）、2（1 小时） |
| $r$ | 缓存读取价相对 $p$ 的倍数 | 0.1 |
| $C_{\text{read}}$ | 这一轮从缓存读的 token 数 | |
| $C_{\text{write}}$ | 这一轮写入缓存的 token 数 | |
| $C_{\text{plain}}$ | 既不读也不写的输入 token 数（最后一个缓存断点之后） | |
| $h$ | 命中率，$C_{\text{read}}$ 占总输入的比例 | 96.4% |
| $k$ | 一段缓存写入后被读取的次数 | |
| $L_i$ | 第 $i$ 轮的输入长度 | 见 [负载画像](/agent/agent-workload) |

任务例子沿用 [负载画像](/agent/agent-workload)：50 轮，固定前缀 10K，每轮追加 1.5K，总输入 2,337.5K，第一次出现的 token 83.5K。

### 1. 价格和规则

两家的规则（2026-10-13 读取官方文档；产品数字会变，以倍数为主）：

| | Anthropic | OpenAI |
|---|---|---|
| 怎么开 | 请求里加 `cache_control`（顶层一个字段即自动缓存，断点随对话后移），最多 4 个断点 | 自动，无需改请求 |
| 最短可缓存长度 | 512–4096 token，按模型不同 | 1024 token |
| 读取价 | 0.1 倍（部分最新模型 0.05 或 0.025 倍） | 0.1 倍（个别模型 0.05 倍） |
| 写入价 | 5 分钟 1.25 倍，1 小时 2 倍 | 较早的模型不另收费；GPT-5.6 起 1.25 倍 |
| 存活时间 | 默认 5 分钟，可选 1 小时；每次命中免费刷新，从请求开始时计时 | 内存缓存一般闲置 5–10 分钟后失效，最长 1 小时；可选更长保留（最长 24 小时） |
| 前缀顺序 | `tools` → `system` → `messages` | 隐藏内容、工具定义之后按请求顺序 |
| 用量字段 | `cache_read_input_tokens`、`cache_creation_input_tokens`、`input_tokens` | `cached_tokens` |

两个容易忽略的点：

1. **前缀匹配是 token 级的**：从第一个不同的 token 起，后面全部不命中。Anthropic 文档列了哪些参数改动会让哪一层失效：改工具定义，`tools`、`system`、`messages` 全失效；切换 `tool_choice`、增删图片，`messages` 失效。
2. **缓存在机器上**：OpenAI 按前缀开头的 hash（加可选的 `prompt_cache_key`）路由到机器，同一前缀每分钟超过约 15 个请求就会溢出到别的机器、命中率下降，所以高并发时要按用户把 key 分散开。这和引擎侧的 [session 亲和路由](/agent/prefix-reuse) 是同一个问题。

### 2. 一轮的账

一轮的输入费用（输出另算）：

$$
\text{cost}_{\text{in}} = p\left(r\, C_{\text{read}} + w\, C_{\text{write}} + C_{\text{plain}}\right)
$$

agent 每轮把新追加的部分写入缓存（自动缓存的断点在最后），$C_{\text{plain}} \approx 0$，于是有效单价是基础价的

$$
m(h) = h\, r + (1-h)\, w
$$

| 命中率 $h$ | $m(h)$（$r = 0.1$，$w = 1.25$） |
|---|---|
| 0 | 1.25（比不开缓存还贵 25%） |
| 50% | 0.675 |
| 90% | 0.215 |
| 96.4% | 0.141 |

### 3. 一个任务的账

本页例子，$p = 3$ USD / 百万 token：

1. 不开缓存：$2.3375 \times 3 = 7.01$ USD。
2. 开缓存、全程命中：写入 $83.5\text{K}$（第 1 轮的 10K 加之后每轮 1.5K），$0.0835 \times 3 \times 1.25 = 0.313$；读取 $2{,}254\text{K}$，$2.254 \times 3 \times 0.1 = 0.676$；合计 0.989 USD。
3. 比例 $0.989 / 7.01 = 0.141$，正好是 $m(96.4\%)$。

Manus 用 Claude Sonnet 的价格举过同样的例子：缓存输入 0.30 USD / 百万 token，未缓存 3 USD，差 10 倍，所以他们把 KV-cache 命中率当作生产 agent 最重要的单一指标。

### 4. 什么时候不划算

**盈亏平衡**：一段前缀写入一次、之后被读 $k$ 次，开缓存的总价是 $w + kr$，不开是 $1 + k$（单位：这段前缀按基础价算一次）。开缓存划算当且仅当

$$
w + k r < 1 + k \quad\Leftrightarrow\quad k > \frac{w - 1}{1 - r}
$$

1. 5 分钟缓存：$k > 0.25/0.9 = 0.28$，读一次就回本。
2. 1 小时缓存：$k > 1/0.9 = 1.11$，要读两次。

**TTL 过期**：工具执行超过 TTL（比如跑一遍 8 分钟的测试），下一轮整段前缀要重新写入。每过期一次，多付

$$
\Delta = (w - r)\, L_i\, p
$$

本页例子里假设有 5 轮的工具都超过 5 分钟、这几轮的上下文平均 60K：每次 $1.15 \times 0.06 \times 3 = 0.207$ USD，5 次 1.04 USD，输入成本从 0.989 涨到约 2.03 USD。改用 1 小时缓存：所有写入从 1.25 倍变 2 倍，多付 $83.5\text{K} \times 0.75 \times 3 / 10^6 = 0.188$ USD，但不再过期，合计约 1.18 USD。所以**有长耗时工具的任务用 1 小时缓存**，只有短工具的任务用 5 分钟。

**前缀每轮都变**：把精确到秒的时间戳放在系统提示开头，每轮都是 $h = 0$，全按 1.25 倍写：$2.3375 \times 3 \times 1.25 = 8.77$ USD，比不开缓存的 7.01 还贵。

### 5. 怎么排才能命中

前缀从前到后按「多久变一次」排：

```
[工具定义]  [系统提示]  [任务描述]  [第 1 轮] [工具结果] [第 2 轮] ... [最新一轮]
 整个任务不变 ─────────────────────────────►  只追加，不修改 ──────────────►
```

1. **工具定义和系统提示放最前面，任务中途不改**。要限制模型能用哪些工具，别删工具定义（会让 `tools` 层及之后全部失效），改用解码时屏蔽（Manus 的 logits masking）或 API 的 allowed tools 参数。工具多了怎么办见 [MCP 与工具协议](/agent/tool-protocols)。
2. **不放会变的东西**：时间戳、请求 id、随机示例放到最后的用户消息里，或者干脆不放。
3. **只追加**：不回头修改历史里的工具结果、不重排消息。要删旧内容（压缩）就一次删够多，见 [上下文管理](/agent/context-management)。
4. **序列化确定**：JSON 的 key 顺序、空白、工具列表顺序每次都一样。MCP 规范也建议 server 按确定的顺序返回工具列表，理由就是提高 prompt cache 命中率。
5. **断点放对**：Anthropic 每个断点往前最多查 20 个位置（连续的 `tool_use` 块、连续的 `tool_result` 块各算一个位置），一轮追加的块太多时，上一轮的缓存可能找不到，要多放一个断点。
6. **够长**：短于最短可缓存长度的前缀不会被缓存，也不报错。

### 6. 怎么监控

每次请求的 usage 里有三个数（Anthropic 的字段名）：read = `cache_read_input_tokens`，creation = `cache_creation_input_tokens`，input = `input_tokens`。这一轮的命中率：

$$
h = \frac{\text{read}}{\text{read} + \text{creation} + \text{input}}
$$

按任务聚合看分布：正常的 agent 任务除第 1 轮外 $h$ 应该在 90% 以上。某一轮突然 $h \approx 0$ 且 creation 很大，就是前缀断了或 TTL 过期，去 diff 这一轮和上一轮的请求。

## 面试追问

::: details Q：缓存写入为什么比普通输入还贵？
写入时要把这段前缀的 KV 存下来，占用服务方的显存或内存，存得越久越贵（5 分钟 1.25 倍、1 小时 2 倍）。读取时省掉了 prefill 计算，所以只收 0.1 倍。写入贵这一点决定了：前缀只用一次的请求开缓存反而亏 25%。
:::

::: details Q：5 分钟和 1 小时的缓存怎么选？
算盈亏平衡次数 $k > (w-1)/(1-r)$：5 分钟要读 1 次回本，1 小时要读 2 次。再看两轮之间的间隔：工具都在几秒内返回就用 5 分钟，每次命中会刷新 TTL；有超过 5 分钟的工具（长测试、等人审批），过期一次就要按 1.25 倍重写整段前缀，这时 1 小时更便宜。
:::

::: details Q：为什么任务中途不能增删工具？
工具定义排在缓存前缀的最前面，改了它 `tools`、`system`、`messages` 三层全部失效，下一轮整段上下文按写入价重付。另外历史里已经有对旧工具的调用，工具没了模型会困惑。要限制工具就在解码时屏蔽对应的 token，或者用 allowed tools 这类参数，工具定义本身不动。
:::

::: details Q：命中率突然掉了，怎么排查？
1. 看 usage：`cache_creation_input_tokens` 很大说明前缀断了或过期，它和 `cache_read_input_tokens` 都是 0 说明前缀太短或没开缓存。
2. diff 这一轮和上一轮的完整请求（工具、系统提示、消息），找第一个不同的位置：常见是时间戳、JSON key 顺序、工具列表顺序、改写了历史。
3. 看两轮间隔有没有超过 TTL。
4. 高并发下看是不是同一前缀的请求太多被分到别的机器（OpenAI 约每分钟 15 个以上会溢出）。
:::

::: details Q：API 的 prompt caching 和推理引擎的 prefix cache 是什么关系？
同一件事的两面：引擎把前缀的 KV 存在显存或内存里，命中时省掉 prefill；API 把省下的计算以 0.1 倍读取价让给用户，再对存储收写入费。所以引擎侧的规则（token 级前缀匹配、要路由回同一台机器、会被淘汰）在 API 侧一样成立，见 [多轮 agent 的 KV 复用](/agent/prefix-reuse)。
:::

## 手撕

比较相邻两轮请求，找出前缀从哪里开始不同（用来排查命中率掉了）：

```python
import json

def segments(req):
    # 按缓存前缀的顺序展开：tools -> system -> messages，序列化时 key 排序保证确定
    out = [("tools", json.dumps(t, sort_keys=True)) for t in req["tools"]]
    out.append(("system", req["system"]))
    out += [(f"messages[{i}]", json.dumps(m, sort_keys=True)) for i, m in enumerate(req["messages"])]
    return out

def first_break(prev, cur):
    a, b = segments(prev), segments(cur)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return x[0]                      # 第一个不同的段，从这里往后都不命中
    return None if len(b) >= len(a) else "truncated"

prev = {"tools": [{"name": "grep"}], "system": "You are an agent. 2026-10-13 09:00:01",
        "messages": [{"role": "user", "content": "fix bug"}]}
cur = {**prev, "system": "You are an agent. 2026-10-13 09:00:09",
       "messages": prev["messages"] + [{"role": "assistant", "content": "..."}]}
print(first_break(prev, cur))                # system：时间戳让整个前缀从系统提示起失效
```

## 参考

- [Anthropic prompt caching 文档](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)：读写倍数、TTL、最短长度、4 个断点、20 个位置的回看、失效规则、usage 字段（2026-10-13 读取）
- [OpenAI prompt caching 文档](https://developers.openai.com/api/docs/guides/prompt-caching)：1024 token 起、按前缀 hash 路由、约 15 RPM 溢出、`prompt_cache_key`、保留时长、读写价（2026-10-13 读取）
- [Manus: Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)：KV-cache 命中率是最重要的指标、缓存与未缓存差 10 倍、稳定前缀、logits masking 代替删工具
- [MCP 规范：Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)：工具列表按确定顺序返回以提高 prompt cache 命中率
