---
title: LLM 网关：多租户限流、配额、背压、模型路由与成本账
status: draft
tags: [gateway, rate-limit, token-bucket, quota, backpressure, routing, fallback, cost]
difficulty: 3
order: 11
related: [/agent/orchestration, /agent/prompt-caching, /agent/eval-observability, /agent/system-design]
agent: [ag-g-rate, ag-g-model, ag-g-cost]
---

# LLM 网关：多租户限流、配额、背压、模型路由与成本账

> 所有模型调用都过网关。它把上游按 RPM / TPM 给的额度分给各租户，额度不够时让调用方排队或退避，上游出事时切模型，并把每一次调用记到账上

## 一句话结论

上游按 RPM 和 TPM 限流（Anthropic 再细分成输入 ITPM 和输出 OTPM，并明说用 token bucket：额度连续补充，60 RPM 可能按每秒 1 个执行）。网关给每个租户一个 token bucket，速率 $r = \text{TPM}/60$，容量 $b$ 决定允许多大的突发。TPM 在请求进来时不知道会输出多少，只能按「输入 + `max_tokens`」预扣、完成后退还；不退还时，`max_tokens = 8000` 实际只输出 500 的请求会让额度只用到 25%。过载时宁可快速返回 429 + `retry-after`，也不要无限排队：按 Little 定律，排队时间 = 队长 / 吞吐。路由把简单请求发给便宜模型：60% 走 Haiku 4.5（单价是 Sonnet 4.5 的 1/3）能省 40%；RouteLLM 报告在 MT Bench 上保持 GPT-4 95% 的质量、成本降到约 1/3.7（作者自测）。降级换模型会让 prompt cache 失效：100k 前缀的一次调用从 \$0.03 变成 \$0.375。成本账按 usage 字段逐项记：未缓存输入、缓存写、缓存读、输出各一个单价；离线任务走 Batch API 打五折。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| RPM / TPM | 每分钟请求数 / token 数上限 | 1000 / 2M |
| ITPM / OTPM | 每分钟输入 / 输出 token 上限（Anthropic 的拆法） | 2M / 400k |
| $r$ | token bucket 的补充速率（token/s） | $\text{TPM}/60$ |
| $b$ | 桶容量：最多攒多少，决定最大突发 | 待定 |
| $I$ | 一个请求的输入 token | 2,000 |
| $O_{\max}$ | 请求里的 `max_tokens` | 8,000 |
| $O$ | 实际输出 token | 500 |
| $\lambda$ | 请求到达率（个/s） | |
| $W$ | 一个请求在系统里待的时间（s） | |
| $L$ | 系统里同时在处理的请求数 | $L = \lambda W$ |
| $h$ | prompt cache 命中率（命中的输入 token / 总输入 token） | 80% |

价格单位是美元 / 百万 token（\$/MTok），2026-10-13 查 [Anthropic 定价页](https://platform.claude.com/docs/en/about-claude/pricing)：

| 模型 | 输入 | 5 分钟缓存写 | 缓存命中 | 输出 |
|---|---|---|---|---|
| Claude Sonnet 4.5 | 3 | 3.75 | 0.30 | 15 |
| Claude Haiku 4.5 | 1 | 1.25 | 0.10 | 5 |

缓存写是输入价的 1.25 倍，命中是 0.1 倍（部分新模型命中价更低，以定价页为准）。

### 1. 上游怎么限：RPM、TPM，以及各家的细节

几家的规则不一样，网关要按上游的规则记账（2026-10-13 查）：

1. **OpenAI**（[Rate limits](https://developers.openai.com/api/docs/guides/rate-limits)）：RPM、RPD、TPM、TPD 等，限额按组织和项目算，不按用户。一个请求计入 TPM 的量取 `max_tokens` 和按字符数估的 token 数里较大的。失败的请求也计入每分钟限额。
2. **Anthropic**（[Rate limits](https://platform.claude.com/docs/en/api/rate-limits)）：
   1. 按模型分别限 RPM、ITPM、OTPM，超了返回 429 和 `retry-after` 头。
   2. 原文：用 token bucket 算法限流，额度连续补充到上限，而不是在固定时间点重置；60 RPM 可能按每秒 1 个执行。
   3. 大多数模型只有未缓存的输入计入 ITPM，缓存命中的不计入：ITPM 2M、命中率 80% 时，每分钟实际能送进去 $2\text{M} / (1 - 0.8) = 10$M 输入 token（文档里的例子）。
   4. `max_tokens` 不计入 OTPM，按实际输出算。

两点直接影响网关设计：限的是**速率**而不是固定窗口里的计数，所以网关自己也该用 token bucket，而不是「每分钟清零的计数器」（后者在窗口交界处会放进两倍的突发）；限额里同时有请求数和 token 数，网关要两样都管，任何一样满了都要拦。

**并发数**。agent 请求一次几秒到几十秒，同时在途的请求数由 Little 定律给出：

$$
L = \lambda W
$$

算一笔：平均每秒 10 个模型调用，每个平均 20 s（长输出），同时在途 $L = 200$ 个。上游或自建推理服务的并发上限要按这个量配。

### 2. Token bucket：速率加突发

每个租户（或每个 API key）一个桶：

$$
\text{tokens}(t) = \min\big(b,\ \text{tokens}(t_0) + r\,(t - t_0)\big)
$$

请求要 $n$ 个 token：桶里够就扣掉、放行；不够就算出还要等 $(n - \text{tokens})/r$ 秒，排队或者把这个数作为 `retry-after` 返回。长期看放行的速率不超过 $r$，短时间最多一次放出 $b$。

**参数怎么定**：

1. $r$ = 分给这个租户的 TPM / 60。租户分到 600k TPM，$r = 10{,}000$ token/s。
2. $b$ 至少要大于单个请求的最大 token 数，否则大请求永远进不去。agent 一次调用可能带 100k 上下文，$b$ 取 10 s 的量 = 100k 正好卡线，取 200k 更稳。
3. $b$ 越大越能吸收突发，但多个租户同时突发时加起来可能超过上游限额，所以各租户的 $b$ 之和要和上游的余量对得上。

请求数也一个桶（RPM / 60 的速率），两个桶都过才放行。

### 3. 预扣与退还：TPM 在请求开始时还不知道

请求进来时只知道输入 $I$，不知道会输出多少，只能先按最坏情况扣，完成后按实际退还：

$$
\text{预扣} = I + O_{\max}, \qquad \text{退还} = O_{\max} - O
$$

**不退还会怎样**：$I = 2$k、$O_{\max} = 8$k、实际 $O = 500$：

1. 每个请求扣 10k，实际只用 2.5k。
2. 额度利用率 $2.5 / 10 = 25\%$：明明只用了四分之一，网关已经开始返回 429。

所以：完成后一定要退还；调用方把 `max_tokens` 设到真正需要的量；按租户的历史输出长度（比如 p95）预扣，而不是一律按 $O_{\max}$。OpenAI 的计数方式本身就包含 `max_tokens`，设得过大会直接吃掉上游额度；Anthropic 的 OTPM 按实际输出算，没有这个问题。

### 4. 多租户：保底、共享、优先级、预算

上游给整个组织一份总额 $G$，网关分给租户：

1. **保底**：每个租户一份保证额度 $g_i$，$\sum g_i \le G$，自己的桶。
2. **共享池**：剩下的 $G - \sum g_i$ 放进一个公共桶，租户用完保底后可以借；借的时候按权重公平排队（weighted fair queuing），防止一个租户把共享池吃光。
3. **优先级**：交互式请求（用户在等）优先于后台任务（批量评测、离线数据生成），后者可以被延后，或者干脆走 Batch API（Anthropic 和 OpenAI 的批处理都是五折）。
4. **预算配额**：按美元的日 / 月上限，独立于速率限制。速率限的是「同一时刻别太猛」，预算限的是「总共别花太多」。一个死循环的 agent 不一定触发速率限制，但会烧光预算，所以预算要按任务和按租户两级都设。

LiteLLM 这类开源网关按 key / user / team 配 `tpm_limit`、`rpm_limit`、`max_parallel_requests`、`max_budget`（[文档](https://docs.litellm.ai/docs/proxy/users)），就是这几样。

### 5. 背压：排队要有上限

上游满了，网关有两个选择：排队，或拒绝。排队的等待时间由 Little 定律给出：

$$
W_{\text{queue}} = \frac{Q}{\mu}
$$

$Q$ 是队长，$\mu$ 是处理速率。

算一笔：上游 ITPM 2M，每个 agent 调用带 50k 未缓存输入，每分钟最多处理 $2\text{M}/50\text{k} = 40$ 个。队里排着 100 个，新来的要等 $100/40 = 2.5$ 分钟。用户等不了这么久，到时候调用方超时、重试，又进一次队，队列只会更长。

做法：

1. **队列有上限**：按调用方的截止时间算，预计等待超过截止时间的请求直接拒绝，返回 429 和 `retry-after`（就是桶算出的等待秒数）。
2. **把压力传回编排层**：编排层看到 429，降低 sub-agent 的并发（fan-out 从 5 降到 2），而不是每个 sub-agent 各自重试（重试放大见 [编排与持久执行](/agent/orchestration) 第 3 节）。
3. **提高等效吞吐**：上面的例子里如果 80% 的输入命中缓存，未缓存部分只有 10k，每分钟能处理 200 个，排队时间降到 30 s。在 Anthropic 的计数规则下，提高缓存命中率直接等于扩容。

### 6. 模型路由与降级

**路由**：按难度把请求发给不同的模型。总单价是加权平均：

$$
c = f \cdot c_{\text{small}} + (1 - f)\, c_{\text{large}}
$$

$f$ 是发给小模型的比例。Haiku 4.5 的输入、输出单价都是 Sonnet 4.5 的 1/3，$f = 60\%$ 时 $c = 0.6 \times \tfrac{1}{3} + 0.4 = 0.6$，省 40%。难点在判断哪些请求能交给小模型：

1. 规则：按任务类型（摘要、分类、格式转换给小模型；规划、写代码给大模型）。agent 里常见的做法是 sub-agent 和工具结果摘要用小模型，lead 用大模型。
2. 学出来的路由器：RouteLLM（[arXiv 2406.18665](https://arxiv.org/abs/2406.18665)）用人类偏好数据训练路由器（矩阵分解、BERT 分类器、因果 LLM 分类器、相似度加权排序），在 GPT-4 和 Mixtral-8x7B 之间选。摘要：成本降到一半以下而不损失质量。表 6：达到 GPT-4 95% 质量时，MT Bench 上成本节省 3.66 倍，MMLU 1.41 倍，GSM8K 1.49 倍（作者自测）。

**降级**：上游出事时的退路，按顺序：

1. 同一模型、同一 provider 重试（带退避）。
2. 同一模型换区域或换 provider（同一个模型常在多个云上有部署）。
3. 换更小的模型，或者缩短 `max_tokens`。
4. 都不行就排队或拒绝。

配合熔断：某个上游在一段时间内失败率过高就暂时摘掉，冷却后再试。LiteLLM 的默认行为是遇到 429、或一分钟内失败超过 50% 时冷却这个部署（[routing 文档](https://docs.litellm.ai/docs/routing)）。

**换模型的隐藏成本：prompt cache 失效**。缓存是按模型存的，长会话中途换模型，整段前缀要重新写缓存。100k token 的上下文，按 Sonnet 4.5：

1. 不换：命中，$0.1\text{M} \times 0.30 = \$0.03$。
2. 换到另一个模型（同价位）：写缓存，$0.1\text{M} \times 3.75 = \$0.375$，贵 12.5 倍，TTFT 也要多一次 100k 的 prefill。

所以路由最好在任务开始时定，一个会话内保持同一个模型；降级只在真的出事时做。另外不同模型的工具调用格式和能力不同，换了模型的 agent 可能在后续轮次表现异常，要单独评测（见 [评测与可观测](/agent/eval-observability)）。

### 7. 成本账

每次调用的响应里都有 usage（未缓存输入、缓存写入、缓存读取、输出 token 数），网关按它逐项记账：

$$
\text{cost} = I_{\text{uncached}}\,p_{\text{in}} + I_{\text{write}}\,p_{\text{write}} + I_{\text{read}}\,p_{\text{read}} + O\,p_{\text{out}}
$$

**算一笔**：[编排](/agent/orchestration) 那页的 50 轮任务（$S = 10$k、每轮新增 2k、输出共 15k），按 Sonnet 4.5：

| 项 | token | 单价 | 美元 |
|---|---|---|---|
| 缓存写 | 0.11M | 3.75 | 0.41 |
| 缓存读 | 2.84M | 0.30 | 0.85 |
| 输出 | 0.015M | 15 | 0.23 |
| 合计 | | | 1.49 |

不用缓存是 \$9.08。网关能看到的优化点：

1. 缓存命中率低于预期，通常是前缀里有会变的东西（时间戳、随机 id、工具定义顺序不稳定），见 [Prompt caching](/agent/prompt-caching)。
2. 每天 1 万个这样的任务就是每天 \$14,900，按租户、按任务类型拆开看谁在花钱。
3. 离线评测和数据生成走 Batch API，五折。
4. 每个任务带预算上限，超了让编排层停下来，而不是等月底看账单。

## 面试追问

::: details Q：为什么网关的限流用 token bucket，不用每分钟清零的计数器？
固定窗口在窗口交界处会放进两倍的突发：上一分钟最后一秒用满，下一分钟第一秒又用满。上游（比如 Anthropic 明说用 token bucket）是按连续速率限的，网关按固定窗口放行，会在交界处把上游打出 429。token bucket 长期速率不超过 $r$，突发不超过 $b$，和上游的行为一致。
:::

::: details Q：TPM 限流时，请求还没跑完怎么知道用了多少 token？
先按输入 + `max_tokens` 预扣，完成后按 usage 退还差额。不退还时，`max_tokens` 8k、实际输出 500 的请求只用了预扣的 25%，额度会被大量浪费。还可以按租户的历史输出分布（比如 p95）预扣，比 `max_tokens` 更准。
:::

::: details Q：上游满了，网关应该排队还是拒绝？
有上限地排队。排队时间 = 队长 / 吞吐，预计等待超过调用方截止时间的直接返回 429 和 `retry-after`。无限排队会让请求超时、调用方重试、队列更长。还要把压力传回编排层，让它减少并发的 sub-agent，而不是每个调用各自重试。
:::

::: details Q：prompt cache 和限流有什么关系？
在 Anthropic 的规则下，缓存命中的输入不计入 ITPM。ITPM 2M、命中率 80% 时，每分钟能送进 10M 输入 token。所以提高命中率等于给上游扩容，同时单价也降到约十分之一。
:::

::: details Q：模型降级有什么代价？
1. 质量下降，agent 后续轮次可能出错，要单独评测降级路径。
2. prompt cache 按模型存，换模型后整段前缀重新写缓存：100k 前缀一次调用从 \$0.03 变成 \$0.375，TTFT 也多一次长 prefill。
3. 不同模型的工具调用格式和行为不同。

所以路由尽量在任务开始时定，一个会话内保持同一个模型。
:::

::: details Q：怎么防一个租户把共享额度吃光？
每个租户一份保底额度和自己的桶；剩下的放进共享池，借用时按权重公平排队；交互式请求优先于后台任务；再加按美元的日 / 月预算上限。速率限制防瞬时过猛，预算防总量失控，两者独立。
:::

## 手撕

带退还的 token bucket。`try_acquire` 返回 0 表示放行，否则返回要等的秒数（填进 `retry-after`）：

```python
import time

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate, self.capacity = rate, capacity      # 每秒补多少、桶最多存多少
        self.tokens, self.t = capacity, time.monotonic()

    def _refill(self):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now

    def try_acquire(self, n):
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return 0.0                                  # 放行
        return (n - self.tokens) / self.rate            # 还要等几秒

    def refund(self, n):
        self._refill()
        self.tokens = min(self.capacity, self.tokens + n)

# 租户 60k TPM -> 每秒 1000；桶容量 5000
b = TokenBucket(rate=1000, capacity=5000)
I, max_tokens = 2000, 2000
print(b.try_acquire(I + max_tokens))    # 0.0：放行，预扣 4000，桶里剩 1000
print(b.try_acquire(I + max_tokens))    # ≈ 3.0：还差 3000，按 1000/s 要等 3 s
b.refund(max_tokens - 500)              # 第一个请求实际只输出 500，退还 1500
print(b.try_acquire(I + max_tokens))    # ≈ 1.5：退还后只差 1500
```

常见变体：请求数和 token 数各一个桶、都过才放行；多个网关实例共享状态时，把桶放进 Redis，用 Lua 脚本原子地做「补充 + 扣减」；用 GCRA（只存一个「理论到达时间」）代替显式的 token 计数，省存储。

## 参考

- [Anthropic：Rate limits](https://platform.claude.com/docs/en/api/rate-limits)（2026-10-13 查）：token bucket、RPM / ITPM / OTPM、缓存命中不计入 ITPM、`max_tokens` 不计入 OTPM
- [Anthropic：Pricing](https://platform.claude.com/docs/en/about-claude/pricing)（2026-10-13 查）：各模型单价、缓存倍率、Batch 五折
- [OpenAI：Rate limits](https://developers.openai.com/api/docs/guides/rate-limits)（2026-10-13 查）：RPM / TPM、`max_tokens` 计入方式
- [RouteLLM (arXiv 2406.18665)](https://arxiv.org/abs/2406.18665)：路由器类型、表 6 成本节省
- [LiteLLM：Routing](https://docs.litellm.ai/docs/routing)、[Reliability](https://docs.litellm.ai/docs/proxy/reliability)、[Budgets](https://docs.litellm.ai/docs/proxy/users)：fallback、冷却、按 key 的限额和预算
- [Token bucket（Wikipedia）](https://en.wikipedia.org/wiki/Token_bucket)、[GCRA](https://en.wikipedia.org/wiki/Generic_cell_rate_algorithm)、[Little's law](https://en.wikipedia.org/wiki/Little%27s_law)
