---
title: 上下文管理：token 预算、截断、压缩与 sub-agent
status: draft
tags: [context-engineering, compaction, token-budget, sub-agent, prompt-caching]
difficulty: 3
order: 4
related: [/agent/agent-workload, /agent/prompt-caching, /agent/tool-protocols, /agent/orchestration]
agent: [ag-c-budget, ag-c-compact]
---

# 上下文管理：token 预算、截断、压缩与 sub-agent

> 上下文随轮数线性变长，迟早会撞上窗口、拖慢效果、抬高账单；问题是在哪个环节、用什么代价把它变短

## 一句话结论

上下文长度 $L_i = S + (i-1)a$，窗口 200K、预留 32K 输出时，本页例子最多跑 106 轮（假设的数字）。但不用等撞墙：上下文越长模型越容易漏掉信息（Anthropic 称为 context rot），每轮的读取费也越高。从便宜到贵四种手段：**进来时就截断工具结果**（不碰已有前缀，缓存不受影响；Claude Code 对 MCP 工具输出默认上限 25,000 token）；**清掉旧工具结果**（Anthropic context editing 默认输入超过 100K 时触发、保留最近 3 次）；**整体摘要**（compaction）；**交给 sub-agent**，它在自己的上下文里做完，只回传 1–2K token 的结论。后三种都改写了前缀，被改处之后要按缓存写入价重付，所以要一次删得够多：本页例子里在 100K 处摘要成 12K，花约 35K 个「基础价 token」，之后每轮省 8.8K，4 轮回本。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $W$ | 模型上下文窗口（token） | 200K（假设） |
| $O_{\max}$ | 为输出（含思考）预留的 token | 32K（假设） |
| $S$ | 固定前缀：系统提示 + 工具定义 + 任务 | 10K |
| $a$ | 每轮追加（模型输出 + 工具结果） | 1.5K |
| $L_i$ | 第 $i$ 轮的输入长度，$S + (i-1)a$ | |
| $r, w$ | 缓存读取、写入价相对基础输入价的倍数 | 0.1、1.25 |
| $\rho$ | 输出价相对基础输入价的倍数 | 5（假设） |
| $n$ | 压缩之后还要跑的轮数 | |

成本用「基础价 token」计：1 个按基础输入价计费的 token 记 1，缓存读取记 $r$，写入记 $w$，输出记 $\rho$。乘以基础价（例如 3 USD / 百万 token）就是钱。

### 1. 预算怎么分

窗口要装下这一轮的输入和输出：$L_i + O_{\max} \le W$。代入 $L_i = S + (i-1)a$：

$$
T_{\max} = \left\lfloor \frac{W - O_{\max} - S}{a} \right\rfloor + 1
$$

例子：$\lfloor (200 - 32 - 10)/1.5 \rfloor + 1 = \lfloor 105.3 \rfloor + 1 = 106$ 轮。工具结果一旦变长（比如每轮 5K），就只剩 32 轮。

撞墙之前就要管，原因有两个：

1. **效果**：Anthropic 的 context engineering 文章说，随着上下文 token 变多，模型从中准确回忆信息的能力下降，这是渐变而不是悬崖，所有模型都有。所以要把上下文当成有限的「注意力预算」，只放对下一步有用的。
2. **成本**：每轮都要把整个上下文读一遍，第 $i$ 轮的读取费正比于 $L_i$（见 [Prompt caching 的账](/agent/prompt-caching)）。

预算按类别给上限，每一类各有手段：

| 类别 | 增长方式 | 手段 |
|---|---|---|
| 工具定义 | 固定，随接入的工具数 | 按需加载、工具检索（[MCP 与工具协议](/agent/tool-protocols)） |
| 单个工具结果 | 每轮一次，可能很大 | 进来时截断 |
| 历史 | 每轮线性增长 | 清旧工具结果、摘要 |
| 子任务的中间过程 | 一段探索几十轮 | 交给 sub-agent |
| 整个任务 | | 总 token、轮数、墙钟的硬上限 |

### 2. 进来时就截断

一个工具结果在第 $i$ 轮进来、大小 $x$，之后还要跑 $n$ 轮，它的总成本是写一次加读 $n$ 次：

$$
\text{cost}(x) = x\,(w + n r)
$$

例子：第 10 轮一次 `grep` 返回 80K token，后面还有 40 轮。

1. 不截断：$80\text{K} \times (1.25 + 0.1 \times 40) = 420\text{K}$，按 3 USD / 百万约 1.26 USD，比整个任务其余部分的输入费（约 0.99 USD）还多；而且占了 200K 窗口的 40%。
2. 截断到 10K：$10\text{K} \times 5.25 = 52.5\text{K}$，约 0.16 USD。

做法：

1. 每个工具设输出上限。Claude Code 对 MCP 工具输出超过 10,000 token 给警告，默认最多 25,000 token（`MAX_MCP_OUTPUT_TOKENS` 可调）。
2. 截断要可恢复：完整输出写进文件，上下文里只放开头、结尾和文件路径，模型需要时再读。Manus 的说法是压缩要能还原，留着 URL 或路径就能再取回来。
3. 让工具本身支持分页、过滤（`grep` 带行数上限、搜索只返回前 k 条），比事后截断更好。

截断只作用在新追加的内容上，已有前缀不变，**不影响缓存**，是最便宜的一层。

### 3. 清掉旧工具结果

工具结果被用过以后，模型很少需要再看原文。Anthropic 的 context editing（`clear_tool_uses`）：输入超过阈值（默认 100K）时，把最旧的工具结果换成占位符，保留最近的几次（默认 3 次），工具调用本身留着，模型还知道自己做过什么。

代价是**从第一个被清的位置起，前缀变了**，之后的部分要按写入价重付。设第一个被清的位置在 $P$，清掉 $x$ 个 token，清之前上下文长 $L$：

$$
\text{一次性代价} = w\,(L - x - P),\qquad \text{之后每轮省} = r\,x
$$

例子：$L = 100\text{K}$，从 $P = 15\text{K}$ 开始清掉 $x = 60\text{K}$：

1. 一次性：$1.25 \times (100 - 60 - 15)\text{K} = 31.25\text{K}$。
2. 每轮省：$0.1 \times 60\text{K} = 6\text{K}$，约 5.2 轮回本。
3. 如果只清了 $x = 2\text{K}$：一次性 $1.25 \times 83\text{K} = 103.75\text{K}$，每轮只省 0.2K，要 519 轮才回本。

这就是 Anthropic 提供 `clear_at_least` 参数的原因：清不够这么多就干脆不清。原则是**少清几次、每次清一大块**。

### 4. 整体摘要（compaction）

上下文接近阈值时，让模型把历史写成一段摘要，用「固定前缀 + 摘要」替换掉历史，从这里继续。Claude Code 的做法：把历史交给模型总结，保留架构决策、没解决的 bug、实现细节，然后带着摘要和最近访问的 5 个文件继续（Anthropic context engineering 文章）。

代价和收益（$L = 100\text{K}$ 时压缩，摘要 $s = 2\text{K}$，压缩后上下文 $S + s = 12\text{K}$）：

| 项 | 计算 | 基础价 token |
|---|---|---|
| 压缩请求读一遍历史（命中缓存） | $0.1 \times 100\text{K}$ | 10K |
| 输出摘要 | $\rho \times 2\text{K} = 5 \times 2\text{K}$ | 10K |
| 下一轮重写前缀 | $1.25 \times 12\text{K}$ | 15K |
| **一次性合计** | | **35K** |
| 之后每轮省 | $0.1 \times (100 - 12)\text{K}$ | 8.8K |

$35 / 8.8 \approx 4$ 轮回本。另一个代价是时间：摘要 2K token 按每 token 20 ms（假设）要解码 40 s，任务在这期间停住，所以 Anthropic 的 API 也提供在后台压缩、算好再换进来的方式。

摘要会丢信息，丢了的东西要能找回来：和截断一样，关键内容落盘（笔记文件、todo 列表），摘要里留路径。

### 5. 交给 sub-agent

把一段需要大量探索的子任务（搜代码、查资料）交给 sub-agent：它有自己的上下文，做完只回传结论。Anthropic 的说法是 sub-agent 内部可能用掉几万 token，回传的通常是 1,000–2,000 token 的精炼摘要。

例子：一段探索要 10 轮，每轮追加 3K（共 30K），主 agent 之后还要跑 $n = 20$ 轮。

1. **主上下文**：自己做会多出 30K 并一直带着；交给 sub-agent 只多 1.5K。之后每轮少读 28.5K，20 轮省 $0.1 \times 28.5\text{K} \times 20 = 57\text{K}$。
2. **sub-agent 自己的花费**：固定前缀 5K（假设），10 轮输入合计 $10 \times 5\text{K} + 3\text{K} \times 45 = 185\text{K}$；写入 $5\text{K} + 9 \times 3\text{K} = 32\text{K}$，读取 153K，合计 $1.25 \times 32 + 0.1 \times 153 = 55.3\text{K}$。
3. 钱上大致打平，换来的是主上下文干净（效果更好）、几个 sub-agent 能并行（时延更短）。代价是总 token 变多：Anthropic 的多 agent 研究系统用的 token 约是聊天的 15 倍，单 agent 约 4 倍。

sub-agent 的固定前缀也要保持稳定，它自己的缓存才能命中。多 agent 怎么编排见 [编排](/agent/orchestration)。

### 6. 对缓存的影响

| 操作 | 前缀从哪里断 | 额外成本 |
|---|---|---|
| 截断新进来的工具结果 | 不断 | 无 |
| 清旧工具结果 | 第一个被清的位置 | 之后的内容按写入价重付一次 |
| 整体摘要 | 固定前缀之后 | 一次摘要输出 + 重写摘要 |
| sub-agent | 主上下文不断 | sub-agent 自己的前缀单独写入、读取 |
| 改系统提示或工具定义 | 最开头 | 整个上下文重写 |

顺序是：先在入口截断，历史满了先清旧工具结果，还不够再摘要；能预见到的大块探索一开始就交给 sub-agent。

## 面试追问

::: details Q：上下文还远没到窗口上限，为什么也要压缩？
两个原因。效果：上下文越长，模型从中准确找回信息的能力越差（context rot），无关的旧工具结果会干扰判断。成本：每轮都要把整个上下文读一遍，读取费正比于长度；在 100K 处压缩到 12K，之后每轮少读 88K。
:::

::: details Q：清旧工具结果时为什么要「一次清够多」？
清理会从第一个被清的位置起改写前缀，之后的内容按 1.25 倍写入价重付一次，这个代价和清掉多少关系不大。收益是每轮少读 $0.1x$。清 60K 约 5 轮回本，清 2K 要几百轮。所以设一个最小清理量（Anthropic 的 `clear_at_least`），不够就不清。
:::

::: details Q：compaction 的代价有哪些？
1. 钱：压缩请求要读一遍全部历史、输出一段摘要（按输出价），下一轮还要把新前缀写入缓存。本页例子合计约 35K 基础价 token，4 轮回本。
2. 时间：摘要要解码，2K token 约 40 s，任务停住；可以在后台压缩。
3. 信息：摘要会丢细节，关键内容要落盘，摘要里留路径以便找回。
:::

::: details Q：sub-agent 省钱吗？
不一定。主上下文少带了子任务的中间过程，之后每轮读得少；但 sub-agent 自己也要付输入输出，总 token 往往更多（Anthropic 报告多 agent 约是聊天的 15 倍）。它真正的收益是主上下文干净和可以并行。子任务探索量大、主任务后面还很长时才值得。
:::

::: details Q：工具结果太长，截断会丢信息怎么办？
截断要可恢复：完整结果写到文件，上下文里放开头、结尾和路径，模型需要时用读文件的工具按范围取。更好的办法是工具本身支持过滤和分页，让模型先要摘要再要细节。截断在入口做，不改已有前缀，缓存不受影响。
:::

## 手撕

一个上下文压缩器：超预算时先清旧工具结果（保留最近 `keep` 个，清不够 `clear_at_least` 就不清），还不够再摘要；返回前缀从哪里断开。

```python
PLACEHOLDER = 20   # 被清掉的工具结果换成一行占位符，约 20 token

def compact(msgs, budget, keep=3, clear_at_least=5_000, summarize=None):
    """msgs: [{"kind": "fixed" | "text" | "tool_result", "tokens": int, ...}]"""
    total = sum(m["tokens"] for m in msgs)
    if total <= budget:
        return msgs, None                                   # 不动，缓存完整
    tools = [i for i, m in enumerate(msgs) if m["kind"] == "tool_result" and m["tokens"] > PLACEHOLDER]
    old = set(tools[:-keep]) if keep else set(tools)
    freed = sum(msgs[i]["tokens"] - PLACEHOLDER for i in old)
    if freed >= clear_at_least:
        out = [dict(m, tokens=PLACEHOLDER, cleared=True) if i in old else m for i, m in enumerate(msgs)]
        if total - freed <= budget:
            return out, min(old)                            # 前缀从第一个被清的位置断开
        msgs = out
    fixed = [m for m in msgs if m["kind"] == "fixed"]       # 系统提示、工具定义、任务
    summary = summarize(msgs) if summarize else {"kind": "text", "tokens": 2_000}
    return fixed + [summary], len(fixed)                    # 前缀从固定部分之后断开

msgs = [{"kind": "fixed", "tokens": 10_000}] + \
       [{"kind": k, "tokens": t} for _ in range(30) for k, t in (("text", 300), ("tool_result", 3_000))]
new, cut = compact(msgs, budget=80_000)
print(sum(m["tokens"] for m in new), cut)   # 109000 -> 28540，前缀从下标 2（第一个工具结果）断开
```

## 参考

- [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：context rot、注意力预算、Claude Code 的 compaction、清工具结果、sub-agent 回传 1,000–2,000 token
- [Anthropic context editing 文档](https://platform.claude.com/docs/en/build-with-claude/context-editing)：`clear_tool_uses` 的默认阈值 100K、保留 3 次、`clear_at_least`、对缓存的影响（2026-10-13 读取）
- [Anthropic compaction 文档](https://platform.claude.com/docs/en/build-with-claude/compaction)：服务端摘要、按需 / 按阈值 / 后台压缩
- [Claude Code MCP 文档](https://code.claude.com/docs/en/mcp)：MCP 工具输出 10,000 token 警告、默认上限 25,000
- [Manus: Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)：文件系统当上下文、可恢复的压缩、todo.md 复述
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：多 agent 约 15 倍 token
