---
title: 评测与可观测：benchmark、pass@k、评测成本、tracing 与线上回放
status: draft
tags: [eval, pass-at-k, swe-bench, tau-bench, tracing, opentelemetry, llm-as-judge, regression]
difficulty: 3
order: 12
related: [/agent/llm-gateway, /agent/orchestration, /agent/agentic-rl, /agent/agent-latency]
agent: [ag-e-trace, ag-e-bench, ag-e-online]
---

# 评测与可观测：benchmark、pass@k、评测成本、tracing 与线上回放

> agent 一次跑几十步、结果带随机性。离线评测要把「采 n 次、对 c 次」换算成无偏的 pass@k，并知道差几个点才算真的有差；线上要把每一步记成 span，才能回答慢在哪、钱花在哪，改了东西之后拿真实任务回放做回归

## 一句话结论

benchmark 用能自动判定的任务：SWE-bench Verified 是 500 道人工筛过的题（93 名开发者标注，原集 68.3% 的样本被筛掉），跑仓库的测试判 patch；τ-bench 比对对话结束时数据库的状态，并用 pass^k 衡量「k 次是不是都对」。一题采 $n$ 次、对 $c$ 次，pass@k 的无偏估计是 $1 - \binom{n-c}{k}/\binom{n}{k}$，直接代 $1-(1-c/n)^k$ 会系统性偏低（$n=10, c=3, k=5$ 时 0.917 vs 0.832）。500 题、成功率 60% 的标准误约 2.2 个百分点，两个版本差 3 个点在 95% 置信下分不出来。评测本身很贵：HAL 跑 9 个模型 × 9 个 benchmark 共 21,730 条 rollout 花了约 4 万美元。线上按 OpenTelemetry GenAI 约定打点：`invoke_agent` 根 span 下挂 `chat {model}` 和 `execute_tool {tool}`，每个 span 记 token 数和耗时，加起来就是这个任务的成本和延迟构成。改了 prompt、模型或工具，把存下来的线上任务重放一遍，比较成功率、轮数、token、延迟，过门槛再放量。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $n$ | 一道题采样（独立跑）的次数 | 10 |
| $c$ | 其中判为正确的次数 | 3 |
| $k$ | pass@k 的 $k$：允许尝试几次 | 5 |
| $N$ | 题目数 | 500 |
| $p$ | 真实的单次成功率（对题目平均） | 0.6 |
| SE | 成功率估计的标准误 | 待求 |
| $C_{\text{task}}$ | 跑一道题一次的花费（美元） | 1.49（[编排](/agent/orchestration) 那页的 50 轮任务） |

$\binom{a}{b}$ 是组合数，$a < b$ 时为 0。

### 1. 常用 benchmark 和它们怎么判对错

| benchmark | 任务 | 怎么判 | 规模 |
|---|---|---|---|
| [SWE-bench](https://arxiv.org/abs/2310.06770) | 真实 GitHub issue，改代码 | 跑仓库测试：原来失败的要通过（FAIL_TO_PASS），原来通过的不能挂（PASS_TO_PASS） | 2,294 题，12 个 Python 仓库 |
| [SWE-bench Lite](https://www.swebench.com/lite.html) | 同上，挑简单的 | 同上 | 300 题 |
| [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) | 同上，人工筛过 | 同上 | 500 题 |
| [τ-bench](https://arxiv.org/abs/2406.12045) | 客服对话，用工具改数据库，用户由 LLM 模拟 | 结束时数据库状态和标准答案比对 | retail 115 题，airline 50 题 |

**SWE-bench Verified 为什么要筛**（OpenAI，2024-08-13）：93 名 Python 开发者标注了 1,699 个随机样本，38.3% 被标为题目描述不清，61.1% 被标为测试可能把正确解判错；按这两条筛掉 68.3%，留下 500 题。GPT-4o 在 Verified 上 33.2%，在完整 SWE-bench 上 16%（作者自测）。题目本身有问题时，分数的天花板不是 100%，模型之间的差距也会被噪声淹没。

**benchmark 会过时**：2026-02-23 OpenAI 发文说 Verified 已经测不出前沿模型的编码能力：审计 o3 在 64 次运行里不能稳定解出的 138 题，59.4% 有实质问题，加上训练数据污染，他们不再报告 Verified，转向 SWE-bench Pro（[原文](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)）。面试时说清楚：分数接近饱和、题目被污染之后，要换更难、更新的集合。

**跑起来要什么**：SWE-bench 的评测 harness 每道题一个 Docker 镜像，官方要求 x86_64、至少 120 GB 空闲磁盘、16 GB 内存、8 核（[README](https://github.com/SWE-bench/SWE-bench)，2026-10-13 查）。agent 的每次尝试还要一个沙箱，见 [代码执行沙箱](/agent/sandbox)。

**评测环境的漏洞**：2025-09 有人发现 SWE-bench 的镜像里留着「未来」的 git 历史，agent 用 `git log --all` 能直接看到修复提交（[SWE-bench issue #465](https://github.com/SWE-bench/SWE-bench/issues/465)）；维护者统计约万分之五的轨迹命中，v4.1.0 镜像删掉了 base commit 之后的历史。所以评测要看轨迹，不能只看分数。

### 2. pass@k：为什么不能直接代 $1-(1-\hat p)^k$

pass@k 的定义：一道题给 $k$ 次机会，至少一次对的概率，再对题目平均。每题只采 $k$ 次方差太大，Chen et al.（[arXiv 2107.03374](https://arxiv.org/abs/2107.03374) §2.1）的做法是采 $n \ge k$ 次（论文里 $n = 200$、$k \le 100$），数出 $c$，用

$$
\text{pass@}k = \mathbb{E}_{\text{题目}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

含义：从 $n$ 个样本里不放回地抽 $k$ 个，全是错的概率是 $\binom{n-c}{k}/\binom{n}{k}$，1 减去它就是至少一个对。这是无偏的。

**算一笔**：$n = 10$、$c = 3$、$k = 5$。

1. $\binom{7}{5} = 21$，$\binom{10}{5} = 252$，pass@5 $= 1 - 21/252 =$ **0.917**。
2. 直接代 $\hat p = 0.3$：$1 - 0.7^5 =$ **0.832**，低了 8.5 个点。
3. 原因：$1-(1-\hat p)^k$ 是 $\hat p$ 的凹函数，由 Jensen 不等式，它的期望小于 $1-(1-p)^k$，所以系统性偏低（论文附录 A）。

$\binom{n}{k}$ 很大时直接算会溢出，论文给了连乘形式：

$$
\frac{\binom{n-c}{k}}{\binom{n}{k}} = \prod_{i=n-c+1}^{n}\left(1 - \frac{k}{i}\right)
$$

**pass^k（τ-bench）**：反过来问 $k$ 次是不是**全**对，衡量可靠性。客服 agent 要的是每次都对，不是试几次总有一次对。无偏估计是从 $n$ 次里抽 $k$ 次全对的概率：

$$
\text{pass}^k = \mathbb{E}_{\text{题目}}\left[\frac{\binom{c}{k}}{\binom{n}{k}}\right]
$$

$n = 10$、$c = 3$、$k = 2$：$\binom{3}{2}/\binom{10}{2} = 3/45 = 0.067$。$n = 8$、$c = 6$、$k = 4$：$15/70 = 0.214$，单次成功率 75% 的题，连对 4 次只有约五分之一。τ-bench 原文：gpt-4o 在 retail 上 pass^1 = 61.2，pass^8 不到 25%（作者自测）。

### 3. 差几个点才算真的有差

$N$ 道题、每题一次，成功率估计的标准误：

$$
\text{SE} = \sqrt{\frac{p(1-p)}{N}}
$$

$N = 500$、$p = 0.6$：$\sqrt{0.24/500} = 0.0219$，95% 置信区间约 $\pm 1.96 \times 2.19 = \pm 4.3$ 个百分点。

1. 版本 A 61%、版本 B 58%，差 3 个点，落在区间里，不能说 A 更好。
2. 两个版本跑同一批题时，用配对比较更灵敏：只看两边结果不同的题（A 对 B 错、A 错 B 对各多少），做 McNemar 检验；两边都对或都错的题不提供信息。
3. 每题多跑几次（$n > 1$）能压低「同一题跑出不同结果」的那部分方差，但压不低「题目只抽了 500 道」的那部分。想要更窄的区间，最终还是要更多题。

### 4. 评测本身的成本

$$
\text{cost} = N \times n \times C_{\text{task}} + \text{沙箱费用}
$$

**算一笔**：500 题、每题 5 次、每次 \$1.49（50 轮、带缓存、Sonnet 4.5 价格）：$500 \times 5 \times 1.49 = \$3{,}725$。沙箱：每次 20 分钟，按每沙箱小时 \$0.10 算（假设），$2500 \times (1/3) \times 0.1 \approx \$83$。token 是大头。

公开的数字：

1. HAL（[arXiv 2510.11977](https://arxiv.org/abs/2510.11977)）：9 个模型 × 9 个 benchmark 共 21,730 条 rollout，约 \$40,000，日志共 25 亿 token；最便宜的 benchmark 平均每次评测 \$13，Online Mind2Web 平均超过 \$450（作者自测）。
2. HAL 的 SWE-bench Verified Mini（50 题）排行榜上，Claude Sonnet 4.5 一次完整评测 \$463.90，约每题 \$9（[页面](https://hal.cs.princeton.edu/swebench_verified_mini)，2026-10-13 查；页面注明部分成本因 span 重复计数被算成 2 倍，且没算缓存折扣）。

省钱的办法：先在小子集上筛（50 题），有希望的再跑全集；离线评测走 Batch API 五折（见 [LLM 网关](/agent/llm-gateway)）；缓存住共享的前缀（system prompt、工具定义）。

### 5. Tracing：一个任务记成一棵 span 树

按 OpenTelemetry 的 GenAI 语义约定（v1.37.0，状态 Development；约定已迁到 [semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) 仓库）：

```
invoke_agent coder                       根 span：整个任务
├── chat claude-sonnet-4-5               一次模型调用（kind = CLIENT）
│     gen_ai.usage.input_tokens  = 12000
│     gen_ai.usage.output_tokens = 300
├── execute_tool run_tests               一次工具调用（kind = INTERNAL）
├── chat claude-sonnet-4-5
├── invoke_agent researcher              sub-agent，下面再挂自己的 chat / execute_tool
│   └── ...
└── chat claude-sonnet-4-5
```

1. 模型调用的 span 名是 `{gen_ai.operation.name} {gen_ai.request.model}`，比如 `chat gpt-4o`；工具是 `execute_tool {gen_ai.tool.name}`；agent 是 `invoke_agent {gen_ai.agent.name}`（[spans](https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-spans.md)、[agent spans](https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-agent-spans.md)）。
2. 关键属性：`gen_ai.provider.name`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.conversation.id`（把同一会话的多个 trace 串起来）。
3. 消息内容（`gen_ai.input.messages`）是 opt-in，约定建议默认不采：既涉及隐私，体积也大。
4. 指标：`gen_ai.client.token.usage`（按 `gen_ai.token.type` 分输入输出的直方图）、`gen_ai.client.operation.duration`（[metrics](https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-metrics.md)）。
5. 缓存读写的 token 数要单独记（各家 usage 字段里都有），否则按输入总数乘单价会把成本高估好几倍。

**用 trace 回答问题**：

1. 钱花在哪：按 span 把四类 token 乘单价加起来（公式见 [LLM 网关](/agent/llm-gateway) 第 7 节），按任务、租户、模型聚合。
2. 慢在哪：沿根 span 找关键路径。比如 50 轮、每轮模型 6 s、工具 4 s（假设），总 500 s 里 60% 在模型、40% 在工具；工具里如果 `run_tests` 每次 3 s，就是先优化测试（只跑相关测试），而不是换更快的模型。
3. 卡在哪：同一个工具连续调用、结果相同，是死循环的信号。

**体积**：如果每个 `chat` span 都存完整输入，第 1 节那个 50 轮任务就要存 2.95M token，约 12 MB（按每 token 约 4 字节，假设），而且大部分是重复的前缀。只存每轮新增的消息，按 `conversation.id` 重建上下文，降到约 0.4 MB。采样用尾部采样：失败的、慢的、贵的任务全留，正常的抽一部分。

### 6. 线上评估与回放

离线 benchmark 和真实流量不同：用户的任务更杂、仓库更大、描述更含糊。线上要做两件事：

**给线上任务打分**：

1. 硬信号：测试是否通过、PR 是否被合并、用户有没有撤销改动。
2. 用户反馈：点赞点踩、重新提问。
3. LLM 评审：让模型按评分标准给轨迹打分。MT-Bench 的研究里，GPT-4 评审和人类的一致率（不算平局 85%）与人和人之间的一致率（81%）相当，但有位置偏差：同一对回答换个顺序，GPT-4 只有 65% 的情况给出一致的判断（[arXiv 2306.05685](https://arxiv.org/abs/2306.05685)，作者自测）。所以要两种顺序各评一次，并定期抽样和人工标注对齐。

**回放做回归**：把线上任务（输入、仓库快照、判定方式）存成回归集，改了 prompt、模型或工具后重跑。两种回放：

| | 回放记录的工具结果 | 在沙箱里重新执行 |
|---|---|---|
| 做法 | 模型每次调用工具，返回当时记下的结果 | 真的在沙箱里执行工具 |
| 成本 | 只有模型的 token | token + 沙箱 |
| 能测什么 | 改了 prompt 后单步决策变没变 | 端到端成功率 |
| 局限 | 模型一旦走了不同的路，记录里没有对应的工具结果，回放就断了 | 外部依赖（网页、API）会变，结果不完全可复现 |

门禁：对比新旧版本在同一批题上的成功率（配对检验，见第 3 节）、平均轮数、token、延迟、成本，全都不明显变差才放量；放量时先灰度一小部分流量，看线上指标。

## 面试追问

::: details Q：pass@k 为什么不能用 $1-(1-c/n)^k$？
$1-(1-\hat p)^k$ 是 $\hat p$ 的凹函数，由 Jensen 不等式，它的期望小于真实的 $1-(1-p)^k$，系统性偏低。无偏估计是 $1 - \binom{n-c}{k}/\binom{n}{k}$：从 $n$ 个样本里不放回抽 $k$ 个、全错的概率。$n=10, c=3, k=5$ 时无偏值 0.917，直接代是 0.832。
:::

::: details Q：pass@k 和 pass^k 分别衡量什么？
pass@k：$k$ 次里至少一次对，衡量能力上限，适合有验证手段、可以多试挑一个的场景（比如能跑测试选通过的 patch）。pass^k：$k$ 次全对，衡量可靠性，适合每次都要对的场景（客服、自动化流程）。成功率 75% 的题，pass@4 接近 1，pass^4 只有约 0.2。
:::

::: details Q：新版本在 SWE-bench Verified 上高了 3 个点，能上线吗？
先看是不是噪声。500 题、成功率 60% 时标准误约 2.2 个点，95% 区间 ±4.3，差 3 个点不显著。用配对比较（只看两个版本结果不同的题，做 McNemar 检验）更灵敏；同时看成本、轮数、延迟有没有变差；还要看分数是不是来自评测漏洞（比如读 git 历史）。
:::

::: details Q：一个 agent 任务的 trace 应该怎么组织？
OpenTelemetry GenAI 约定：根 span `invoke_agent {agent}`，下面每次模型调用一个 `chat {model}` span、每次工具调用一个 `execute_tool {tool}` span，sub-agent 是嵌套的 `invoke_agent`。每个模型 span 记输入输出 token（缓存读写单独记）、耗时、模型名；用 `gen_ai.conversation.id` 串起同一会话。消息内容默认不采，要采也只存增量。
:::

::: details Q：怎么知道一个 agent 任务慢在哪？
沿 trace 的关键路径把时间分到模型调用和工具调用上，再看每类里的大头。常见结论：工具（跑测试、装依赖）占的时间不比模型少，这时优化工具（只跑相关测试、缓存依赖）比换更快的模型有效；模型侧看 TTFT 是否因为缓存没命中而变长。
:::

::: details Q：LLM 当评审靠得住吗？
总体一致率和人与人之间相当（MT-Bench：GPT-4 与人 85%、人与人 81%，不算平局），但有位置偏差、偏好长回答、偏好自己的输出。做法：两种顺序各评一次，不一致算平局；给明确的评分标准；定期抽样人工标注校准；能用硬信号（测试、状态比对）的地方不用 LLM 评审。
:::

::: details Q：线上回放用记录的工具结果还是真的重新执行？
记录回放便宜、可复现，适合检查改 prompt 后单步决策；但模型一旦走了不同的路，就没有对应的记录可用。重新执行能测端到端成功率，但要沙箱、更贵，外部依赖变了结果也会变。一般两层都做：快速的记录回放做每次提交的检查，重新执行做发版前的门禁。
:::

## 手撕

pass@k 和 pass^k 的无偏估计，用连乘避免组合数溢出：

```python
import numpy as np
from math import comb

def pass_at_k(n, c, k):
    """n 次里 c 次对：1 - C(n-c, k) / C(n, k)"""
    if n - c < k:
        return 1.0                      # 错的不够 k 个，抽 k 个必有对的
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def pass_hat_k(n, c, k):
    """tau-bench 的 pass^k：k 次全对的概率 C(c, k) / C(n, k)"""
    return comb(c, k) / comb(n, k)

print(pass_at_k(10, 3, 1))              # 0.3
print(pass_at_k(10, 3, 5))              # 0.9167
print(1 - (1 - 0.3) ** 5)               # 0.8319：直接代，偏低
print(pass_hat_k(10, 3, 2))             # 0.0667
print(pass_hat_k(8, 6, 4))              # 0.2143

def benchmark_score(results, k):        # results: 每题的 (n, c)，对题目平均
    return float(np.mean([pass_at_k(n, c, k) for n, c in results]))
```

常见变体：给出标准误 $\sqrt{p(1-p)/N}$ 和置信区间；两个版本的配对比较（统计 A 对 B 错、A 错 B 对的题数）；按题目难度分桶报告。

## 参考

- [Chen et al., Evaluating Large Language Models Trained on Code (arXiv 2107.03374)](https://arxiv.org/abs/2107.03374)：§2.1 pass@k 无偏估计、图 3 连乘实现、附录 A 偏差
- [SWE-bench (arXiv 2310.06770)](https://arxiv.org/abs/2310.06770)、[SWE-bench Lite](https://www.swebench.com/lite.html)、[harness README](https://github.com/SWE-bench/SWE-bench)：任务、判定、资源要求
- [OpenAI：Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)（2024-08-13）：标注与筛选数字
- [OpenAI：Why SWE-bench Verified no longer measures frontier coding capabilities](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)（2026-02-23）
- [SWE-bench issue #465](https://github.com/SWE-bench/SWE-bench/issues/465)：git 历史泄露
- [τ-bench (arXiv 2406.12045)](https://arxiv.org/abs/2406.12045)：pass^k 定义和估计、retail / airline 结果
- [HAL (arXiv 2510.11977)](https://arxiv.org/abs/2510.11977)、[HAL SWE-bench Verified Mini 排行](https://hal.cs.princeton.edu/swebench_verified_mini)：评测成本
- [OpenTelemetry GenAI semantic conventions v1.37.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-spans.md)：span 名、属性、指标；[新仓库](https://github.com/open-telemetry/semantic-conventions-genai)
- [Judging LLM-as-a-Judge with MT-Bench (arXiv 2306.05685)](https://arxiv.org/abs/2306.05685)：评审一致率、位置偏差
