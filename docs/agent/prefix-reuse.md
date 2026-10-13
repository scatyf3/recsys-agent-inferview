---
title: 多轮 agent 的 KV 复用：前缀命中、工具等待、session 亲和
status: draft
tags: [prefix-caching, kv-cache, radix-attention, routing, agent]
difficulty: 3
order: 2
related: [/agent/agent-workload, /agent/prompt-caching, /agent/agent-latency]
agent: [ag-m-prefix, ag-m-route]
---

# 多轮 agent 的 KV 复用：前缀命中、工具等待、session 亲和

> 推理引擎这一侧怎么让 agent 的下一轮少算：前缀怎么匹配、工具执行那几秒 KV 放哪、请求该发到哪台机器

## 一句话结论

agent 第 $i$ 轮的输入是第 $i-1$ 轮的输入加输出再加工具结果，前缀完全相同，推理引擎把算好的 KV 按 block 存下来，用链式 block hash（vLLM）或 radix tree（SGLang）匹配，命中的部分不再 prefill。三个系统问题：第一，前缀必须 token 级一致，chat template 改写历史（比如删掉上一轮的 thinking）就会从改写处断开。第二，工具执行期间请求已经结束，KV 是留在 HBM（占显存）、换到 CPU（PCIe 来回）还是丢掉（重算）：8B 模型 5 万 token 的 KV 是 6.55 GB，换出约 0.13 s，重算约 3 s（假设 MFU 50%），工具均值只有 1–2 s，所以短时间留在 HBM 或换出都远比重算划算；InferCept 测得这类重算占模型前向时间的 37–40%，Continuum 按工具时长分布给 KV 设 TTL。第三，KV 只在一台机器上，随机路由到 8 个副本时下一轮只有 1/8 概率回到原机；按前缀做亲和路由能把命中率拉上去（SGLang v0.4 报告 20% → 75%），但要在负载失衡时退回最空闲副本。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $B$ | KV block 大小（token） | 16 |
| $L$ | 这一轮请求的输入长度（token） | 11,500 |
| $\ell$ | 层数 | 32（Llama-3.1-8B） |
| $h_{kv}$ | KV head 数（GQA） | 8 |
| $d_h$ | 每个 head 的维度 | 128 |
| $b$ | 每个元素的字节数 | 2（bf16） |
| $m_{kv}$ | 每个 token 的 KV 字节数 | 128 KiB |
| $N$ | 模型参数量 | $8.03\times10^9$ |
| $t_{\text{tool}}$ | 工具执行时间 | 均值 1–2 s |
| $\text{BW}_{\text{pcie}}$ | GPU 与 CPU 内存之间的有效带宽 | 50 GB/s（假设，PCIe Gen5 x16 单向理论 64 GB/s） |
| $R$ | 推理副本数 | 8 |

GB = $10^9$ 字节，KiB = 1024 字节。

### 1. 命中怎么判：链式 block hash

vLLM 的 automatic prefix caching 把 KV 切成 $B$ 个 token 一块，第 $j$ 块的 hash 依赖它前面的全部内容：

$$
H_j = \text{hash}\left(H_{j-1},\ \text{tokens}[jB : (j+1)B],\ \text{extra}\right)
$$

1. $H_{j-1}$ 是前一块的 hash，所以两条序列第 $j$ 块 hash 相同，就意味着前 $(j+1)B$ 个 token 全部相同。查表时从第 0 块往后找，遇到第一个不命中就停。
2. extra 里放 LoRA id、多模态输入的 hash、租户隔离用的 `cache_salt`。不同租户的 salt 不同，第 0 块 hash 就不同，KV 不会跨租户共享（防止通过 TTFT 猜别人的 prompt）。
3. 只缓存**满块**：没满的最后一块还会被追加写，不能共享。
4. 至少留最后一个 token 去算：全部命中也得跑一次前向拿下一个 token 的 logits，所以最多匹配 $\lfloor (L-1)/B \rfloor$ 块。
5. 请求结束后 block 不立刻清空，而是进入 free 队列（LRU），内容和 hash 还在，被新请求分配走之前都能命中。

**算一笔**（用 [负载画像](/agent/agent-workload) 的例子，$S = 10\text{K}$，$o = 300$，$r = 1{,}200$）：

1. 第 1 轮：输入 10,000，输出 300，整条序列 10,300 token。生成时产生的 KV 也在 block 里，满块有 $\lfloor 10300/16 \rfloor = 643$ 块，即 10,288 token。
2. 第 2 轮：输入 $10{,}000 + 300 + 1{,}200 = 11{,}500$，最多可匹配 $\lfloor 11499/16 \rfloor = 718$ 块；表里有前 643 块，命中 10,288 token。
3. 要算的只有 $11{,}500 - 10{,}288 = 1{,}212$ 个 token，命中率 $10288/11500 = 89.5\%$。轮数越多、上下文越长，这个比例越接近 [负载画像](/agent/agent-workload) 里 96% 的上限。

SGLang 的 RadixAttention 做的是同一件事，数据结构换成 radix tree：边上是 token 序列，叶子按 LRU 淘汰，天然支持多条序列共享前缀。所有 session 都共享同一段系统提示和工具定义（$S$），树里只存一份：10K token × 128 KiB = 1.31 GB，每多一个并发 session 就省这么多。两种结构的细节对比见姊妹站 [KV Cache 与 PagedAttention](https://scatyf3.github.io/ai-infra-inferview/inference/kv-cache-paged-attention)。

**前缀断开的常见原因**（token 级不一致，从那个位置往后全部重算）：

1. 系统提示开头放了时间戳、请求 id。
2. 工具定义顺序不稳定，或 JSON 序列化的 key 顺序不确定（Manus 专门提醒过）。
3. chat template 改写历史：有些模型的模板在下一轮会删掉上一轮的 thinking 内容，于是从上一轮回答开始和生成时的 token 不一样，每轮都从这里断开。
4. 把输出解码成文本再重新分词：同一段文字重新分词可能得到不同的 token。RL rollout 里同样的问题见姊妹站 [Agentic RL](https://scatyf3.github.io/ai-infra-inferview/posttrain/rl-agentic)。

### 2. KV 有多大

每个 token 的 KV 是每层一份 K、一份 V：

$$
m_{kv} = 2\,\ell\,h_{kv}\,d_h\,b
$$

| 模型 | $\ell$ | $h_{kv}$ | $d_h$ | $m_{kv}$ | 5 万 token |
|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | 8 | 128 | 131,072 B = 128 KiB | 6.55 GB |
| Llama-3.1-70B | 80 | 8 | 128 | 327,680 B = 320 KiB | 16.4 GB |

8B 模型放一张 80 GB 的 H100：bf16 权重约 16 GB，假设留给 KV 55 GB，能放 $55\times10^9 / 131072 \approx 42$ 万 token，也就是 8 个 5 万 token 的 session。agent 的上下文长，并发数被显存卡得很死，所以「等工具的 session 还占不占显存」直接影响吞吐。

### 3. 工具等待期间：留、换、丢

请求在发出工具调用后就结束了，下一轮要等工具执行完、应用把结果拼回来才到。这段时间 KV 有三种处理（InferCept 的分类）：

| 做法 | 代价 | 例子（8B，5 万 token，6.55 GB） |
|---|---|---|
| 留在 HBM（preserve） | 占着 $L \cdot m_{kv}$ 的显存 $t_{\text{tool}}$ 秒，别的请求用不了 | 6.55 GB 占 1.5 s |
| 换到 CPU（swap） | 来回各 $L\,m_{kv} / \text{BW}_{\text{pcie}}$，可以和计算重叠 | $6.55/50 = 0.13$ s 一趟 |
| 丢掉（discard） | 下一轮重新 prefill $L$ 个 token | 约 3 s |

重算的时间按 FLOPs 估：线性层每 token $2N$，causal attention 每层约 $2 L^2 d$（$d = 4096$ 是 8B 的 hidden 维）：

1. 线性层：$2 \times 8.03\times10^9 \times 5\times10^4 = 8.0\times10^{14}$。
2. attention：$2 \times 32 \times (5\times10^4)^2 \times 4096 = 6.6\times10^{14}$。
3. 合计 $1.46\times10^{15}$ FLOPs，H100 bf16 峰值 989 TFLOP/s，假设 MFU 50%：$1.46\times10^{15} / 4.9\times10^{14} \approx 3.0$ s。

工具均值只有 1–2 s（Continuum 表 1），而重算要 3 s、还占着 GPU 算力，所以丢掉是最差的。InferCept 统计：把每次工具调用当作请求结束、下一轮从头算，重算占了模型前向总时间的 37–40%；按每次中断的情况在三种做法里选，吞吐提高 1.6–2 倍（作者自测）。

**留多久：TTL**。Continuum 的做法是请求发出工具调用后把 KV 钉在 HBM 里一段时间 $\tau$，到期没回来就淘汰。$\tau$ 取「工具在 $\tau$ 内返回的概率 × 命中省下的时间」减去「占显存的代价」最大的那个值，概率用这个工具历史时长的经验分布估计。工具时长长尾严重（BFCL 里最慢 10% 的 `fetch_url` 占了总等待的 52.5%），固定 TTL 要么对慢工具白占显存，要么对快工具过早淘汰，所以按工具分别统计。报告的平均任务完成时间提升最高 8 倍以上（作者自测）。

**多 agent 工作流**：KVFlow 把多个 agent 的执行顺序建成一张 step graph，估计每个 agent 还有几步会再次运行，按这个而不是 LRU 决定淘汰谁，并在它运行前一步把 KV 从 CPU 预取回 GPU。相对 SGLang（带分层 radix cache）最高 1.83 倍（单个工作流、大 prompt）和 2.19 倍（大量并发工作流）（作者自测）。

**再往下一层：集群级 KV 池**。Mooncake 把 GPU 集群里闲置的 CPU 内存和 SSD 组织成分布式 KV 缓存，prefill 和 decode 分离，KV 通过 RDMA 在机器间传。好处是 KV 不再绑死在某个副本上，下一节的路由矛盾随之变小。报告模拟场景吞吐最高提高 525%，Kimi 线上多处理 75% 的请求（作者自测）。

### 4. session 亲和路由与负载均衡

KV 在某个副本的显存里，下一轮必须回到这个副本才能命中。

**随机路由**：$R$ 个副本，下一轮回到原副本的概率只有 $1/R$。$R = 8$ 时 12.5%，其余 87.5% 的轮次要在新副本上重算整段前缀（上面的例子是 3 s）。

**按 session 固定**：`hash(session_id) mod R`，Manus 的建议也是自部署 vLLM 时用 session id 做路由。问题是 session 的长度和轮数差别很大，一个副本分到几个超长任务就会过载，而迁走一个 session 的代价是在新副本上重算它的上下文。

**cache-aware 路由**（SGLang router）：

1. router 为每个 worker 维护一棵近似 radix tree，记录发给它的请求前缀（存原始文本，省掉分词），不用去问 worker 真实的缓存。
2. 先判断负载是否失衡：最忙和最闲的 worker 负载之差超过绝对阈值、且最忙的超过最闲的若干倍，就算失衡（当前 SGLang Model Gateway 文档的默认值是差 64、倍数 1.5，版本间会变），这时直接发给最闲的。
3. 不失衡时，找前缀匹配最长的 worker；匹配比例超过 `cache_threshold`（当前默认 0.3）就发给它，否则发给缓存空间最多的。
4. SGLang v0.4 报告：命中率 20% → 75%，吞吐 1.9 倍（作者自测）。

这个规则的思路是：**均衡时追求命中，失衡时先保负载**，用一次重算换掉一次排队。迁移的代价可以按第 3 节估：如果 KV 能通过集群 KV 池（Mooncake 这类）以 50 GB/s 拉过来，6.55 GB 只要 0.13 s，迁移就便宜得多，亲和和均衡的矛盾也就小了。

## 面试追问

::: details Q：为什么 vLLM 的 block hash 要把前一块的 hash 带进去？
单看一块的 16 个 token 相同不代表 KV 相同：KV 取决于这个位置之前的所有 token。把 $H_{j-1}$ 带进去，hash 相同就等价于整个前缀相同，查表只要从头往后逐块比，遇到第一个不命中就停。这也意味着前缀中间改一个 token，后面所有块都不命中。
:::

::: details Q：前缀明明没改，命中率却很低，可能是什么原因？
按顺序排查：1. 请求被路由到了别的副本（看路由策略）。2. 显存紧张，block 在两轮之间被 LRU 淘汰了（看 KV 使用率和淘汰数）。3. 前缀其实变了：chat template 删了历史里的 thinking、工具 JSON 的 key 顺序不稳定、开头有时间戳。4. 多租户开了 `cache_salt`，不同租户本来就不共享。可以把两轮实际送进引擎的 token id 序列 diff 一下，看从哪里开始不同。
:::

::: details Q：工具执行期间 KV 是留、换还是丢？怎么定？
看三个时间：工具时长 $t_{\text{tool}}$、换出换入 $2 L m_{kv}/\text{BW}$、重算 prefill 的时间。8B、5 万 token 时分别约 1.5 s、0.26 s、3 s。工具短、显存够就留；显存紧就换到 CPU，传输能和别的请求的计算重叠；工具要跑几分钟、或者上下文很短重算很便宜，才丢。实际系统按工具的历史时长分布设 TTL（Continuum）。
:::

::: details Q：session 亲和路由会带来什么问题？怎么缓解？
负载不均：session 长短、轮数差别大，固定路由让个别副本过载，而迁走 session 要重算上下文。缓解办法是 cache-aware 路由加失衡检测：均衡时按前缀匹配度选副本，失衡（排队数差超过阈值）时发给最闲的。再进一步是把 KV 放进集群级 KV 池，跨机器拉 KV 比重算便宜，迁移代价变小。
:::

::: details Q：多个 session 共享系统提示和工具定义，能省多少显存？
radix tree / block 共享只存一份。8B 模型每 token KV 128 KiB，10K token 的共享前缀是 1.31 GB；100 个并发 session 不共享要 131 GB，共享只要 1.31 GB。前提是这段前缀在所有 session 里 token 级一致，所以系统提示里不要放每个用户不同的内容（放到后面）。
:::

## 手撕

链式 block hash 的前缀匹配：只缓存满块，至少留最后一个 token 去算。

```python
import hashlib

B = 16

def block_hashes(tokens, extra=b""):
    hs, h = [], extra                       # extra: LoRA id / cache_salt 等
    for j in range(len(tokens) // B):       # 只有满块
        blk = tokens[j * B:(j + 1) * B]
        h = hashlib.sha256(h + repr(blk).encode()).digest()   # 带上前一块的 hash
        hs.append(h)
    return hs

class PrefixCache:
    def __init__(self):
        self.table = {}                     # hash -> 物理 block id

    def insert(self, tokens):               # 请求结束时把满块登记进来
        for h in block_hashes(tokens):
            self.table.setdefault(h, len(self.table))

    def match(self, tokens):                # 返回命中的 token 数
        n = 0
        for h in block_hashes(tokens[:len(tokens) - 1]):   # 最后一个 token 必须算
            if h not in self.table:
                break
            n += 1
        return n * B

cache = PrefixCache()
turn1 = list(range(10_300))                 # 第 1 轮输入 10,000 + 输出 300
cache.insert(turn1)
turn2 = turn1 + list(range(10**6, 10**6 + 1_200))   # 追加 1,200 个工具结果 token
print(cache.match(turn2))                   # 10288
turn2_bad = [-1] + turn2[1:]                # 开头改了一个 token（比如时间戳）
print(cache.match(turn2_bad))               # 0
```

## 参考

- [vLLM automatic prefix caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching.html)：链式 block hash、只缓存满块、LRU free 队列、`cache_salt`
- [SGLang (arXiv 2312.07104)](https://arxiv.org/abs/2312.07104)：RadixAttention
- [SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4)：cache-aware load balancer，命中率 20% → 75%、吞吐 1.9 倍
- [SGLang Model Gateway 文档](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)：cache-aware 路由的阈值参数（2026-10-13 读取）
- [InferCept (arXiv 2402.01869)](https://arxiv.org/abs/2402.01869)：工具中断导致的重算占 37–40% 前向时间，preserve / swap / discard
- [Continuum (arXiv 2511.02230)](https://arxiv.org/abs/2511.02230)：工具调用期间按 TTL 钉住 KV，TTL 的选取，工具时长分布
- [KVFlow (arXiv 2507.07400)](https://arxiv.org/abs/2507.07400)：agent step graph 驱动的淘汰与预取
- [Mooncake (arXiv 2407.00079)](https://arxiv.org/abs/2407.00079)：以 KV 为中心的分离式架构、集群级 KV 池
- [Manus: Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)：稳定前缀、确定性序列化、自部署时按 session id 路由
- 姊妹站 [KV Cache 与 PagedAttention](https://scatyf3.github.io/ai-infra-inferview/inference/kv-cache-paged-attention)：block table、prefix caching 生命周期、RadixAttention 对比
