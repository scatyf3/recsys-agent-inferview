---
title: RAG 与记忆：切块、混合检索、rerank、长期记忆的延迟与成本账
status: draft
tags: [rag, embedding, bm25, hybrid-search, rrf, rerank, colbert, memory, memgpt]
difficulty: 3
order: 8
related: [/agent/context-management, /agent/prompt-caching, /agent/agent-latency, /recsys/ann-retrieval]
agent: [ag-r-rag, ag-r-vec, ag-r-mem]
---

# RAG 与记忆：切块、混合检索、rerank、长期记忆的延迟与成本账

> 窗口放不下的东西要在需要时取回来。离线算清楚切多少块、存多大、embedding 花多少钱；在线算清楚两路召回、rerank、拼进 prompt 各花多少时间和 token；长期记忆再多三件事：写什么、怎么取、什么时候忘

## 一句话结论

10 亿 token 的语料按 512 token、重叠 64 切块是约 223 万块；用 1536 维的 text-embedding-3-small，向量 fp32 约 13.7 GB，HNSW 图每个向量再多约 150 字节，embedding 一遍约 \$23。在线先走两路召回：dense（bi-encoder + ANN）擅长同义改写，BM25 擅长错误码、函数名这类精确词，两路按名次用 RRF 融合（$\sum 1/(60 + \text{rank})$）；再用 cross-encoder 对几十到上百个候选精排，它准但每个候选都要过一遍模型（ColBERT 论文里 BERT 精排 1000 个候选要 10.7 s，ColBERT 的 late interaction 只要 61 ms）。Anthropic 的 Contextual Retrieval 把 top-20 检索失败率从 5.7% 降到 1.9%：给每块加上下文、加 BM25、加 rerank 依次贡献。token 账上，拼进 prompt 的 top-k 块在 agent 里会留在上下文、被之后每一轮重发，检索得越早越贵。长期记忆按 MemGPT 的分层：窗口里放工作记忆，外部放全部历史和档案，靠函数调用换入换出；检索时按相关度、新近度、重要度打分（Generative Agents 的新近度每小时乘 0.995，一周后约剩 0.43）。Mem0 报告用抽取后的记忆代替整段历史，token 从约 2.6 万降到约 1800、p95 延迟从 17 s 降到 1.4 s，但准确率比全量上下文低约 6 个点（作者自测）。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $D$ | 语料总 token 数 | $10^9$ |
| $c$ | 块大小（token） | 512 |
| $o$ | 相邻块重叠（token） | 64 |
| $n_{\text{chunk}}$ | 块数 | 待求 |
| $d$ | embedding 维度 | 1536（text-embedding-3-small） |
| $K$ | 召回后送去 rerank 的候选数 | 50–150 |
| $k$ | 最终拼进 prompt 的块数 | 20 |
| $q$ | 查询长度（token） | 32 |
| $k_{\text{rrf}}$ | RRF 的平滑常数 | 60 |
| $k_1, b$ | BM25 的词频饱和参数和长度归一化参数 | 1.2、0.75 |

GB = $10^9$ 字节。价格 2026-10-13 查 OpenAI 模型页：text-embedding-3-small \$0.02 / 百万 token，text-embedding-3-large \$0.13 / 百万 token。

### 1. 离线：切块、embedding、索引的账

**块数**：每块往前走 $c - o$ 个 token：

$$
n_{\text{chunk}} = \left\lceil \frac{D - o}{c - o} \right\rceil
$$

$D = 10^9$、$c = 512$、$o = 64$：$n_{\text{chunk}} = \lceil (10^9 - 64)/448 \rceil \approx 2.23\times10^6$ 块。重叠让实际要 embed 的 token 变成 $n_{\text{chunk}} \cdot c = 1.14\times10^9$，多 14%。

**embedding 费用**：$1.14\times10^9 / 10^6 \times \$0.02 \approx \$23$。用 large 模型是 \$149。一次性的，但每次换 embedding 模型都要全量重算。

**存储**：

1. 向量：$n_{\text{chunk}} \times d \times 4$ 字节 $= 2.23\times10^6 \times 1536 \times 4 \approx 13.7$ GB（fp32）。
2. HNSW 图：每个元素平均 $(M_{\max 0} + m_L M_{\max}) \times 4$ 字节（[HNSW 论文](https://arxiv.org/abs/1603.09320) §4.2.3），$M = 16$ 时 $M_{\max 0} = 2M = 32$、$M_{\max} = 16$、$m_L = 1/\ln 16 = 0.36$，约 151 字节，223 万块共 0.34 GB。论文说典型是每个对象 60–450 字节，比向量本身小得多。
3. 压缩：OpenAI 的 v3 embedding 支持用 `dimensions` 参数截短，发布文里说 large 截到 256 维仍好于 1536 维的 ada-002（[OpenAI，2024-01-25](https://openai.com/index/new-embedding-models-and-api-updates/)）。1536 → 256 维，存储和内积计算都降到 1/6。

**块大小怎么选**：

1. 太大：一块里混了多个主题，embedding 被平均掉，检索不准；拼进 prompt 也更费 token。
2. 太小：一块的语义不完整（代词指向前文、表格被切断），检索到了也用不上。
3. 框架默认值：LlamaIndex `chunk_size` 1024、`chunk_overlap` 20（token）；LangChain 的 `TextSplitter` 默认 4000 / 200（字符），`RecursiveCharacterTextSplitter` 按段落、换行、空格的顺序找切分点（2026-10-13 查文档和源码）。
4. 结构化内容按结构切：代码按函数 / 类，Markdown 按标题，表格整张保留。

### 2. 两路召回：dense + BM25，用 RRF 融合

**dense（bi-encoder）**：查询和文档各自编码，相似度是向量内积（归一化后就是余弦）。文档向量离线算好，在线只编码一次查询、做一次 ANN 搜索，几毫秒到几十毫秒。擅长语义相近、措辞不同的匹配。

**BM25**：词频加长度归一化的打分（[Robertson & Zaragoza 2009](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) §3.4）：

$$
\text{BM25}(q, D) = \sum_{t \in q} \text{idf}(t)\cdot\frac{f(t, D)\,(k_1 + 1)}{f(t, D) + k_1\left(1 - b + b\,\frac{|D|}{\text{avgdl}}\right)}
$$

$f(t, D)$ 是词 $t$ 在文档 $D$ 里出现的次数，$|D|$ 是文档长度，avgdl 是平均长度，idf 让罕见词权重大。Elasticsearch 默认 $k_1 = 1.2$、$b = 0.75$；论文说 $1.2 < k_1 < 2$、$0.5 < b < 0.8$ 在很多场景都不错。

**词频饱和**：文档长度等于平均长度时，分子分母里的长度项是 1，词频那一项是 $f(k_1+1)/(f + k_1)$：

1. $f = 1$：$2.2 / 2.2 = 1$。
2. $f = 2$：$4.4 / 3.2 = 1.375$。
3. $f = 10$：$22 / 11.2 = 1.96$。

出现 10 次只比出现 1 次多不到 2 倍，上限是 $k_1 + 1 = 2.2$。这防止一个词堆砌很多次的文档霸榜。

**为什么要两路**：dense 对错误码 `E1234`、函数名 `parse_header`、版本号这类精确字符串不敏感（它们在 embedding 空间里和相似的串挨得很近）；BM25 对同义改写无能为力（「怎么退款」查不到「申请返还货款」）。agent 的查询里两类都多。

**RRF 融合**（[Cormack et al., SIGIR'09](http://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)）：两路的分数尺度不同（余弦在 0–1，BM25 没有上界），不能直接相加，只用名次：

$$
\text{RRF}(d) = \sum_{r \in \text{各路}} \frac{1}{k_{\text{rrf}} + \text{rank}_r(d)}
$$

算一笔（$k_{\text{rrf}} = 60$）：

| 文档 | dense 名次 | BM25 名次 | RRF |
|---|---|---|---|
| A | 1 | 5 | $1/61 + 1/65 = 0.03178$ |
| B | 3 | 1 | $1/63 + 1/61 = 0.03227$ |
| C | 1（只在一路） | — | $1/61 = 0.01639$ |

B 排第一：两路都靠前胜过只在一路第一。$k_{\text{rrf}} = 60$ 是论文在预实验里定的，之后没再调；论文表 1 的 MAP：$k = 0$ 时 0.2072、60 时 0.2145、500 时 0.2098，说明它不敏感。它的作用是压平头部：没有它，第 1 名（1/1）比第 2 名（1/2）大一倍，一路的排序抖动就能决定结果。Elasticsearch 的 RRF 默认 `rank_constant` 也是 60。

**Contextual Retrieval 的数字**（[Anthropic，2024-09-19](https://www.anthropic.com/news/contextual-retrieval)，作者自测，指标是 top-20 检索失败率 = 1 − recall@20）：

| 做法 | 失败率 | 相对基线 |
|---|---|---|
| 基线（dense） | 5.7% | |
| 每块前面加 50–100 token 的上下文说明再 embed | 3.7% | −35% |
| 再加同样处理过的 BM25 | 2.9% | −49% |
| 再加 rerank（取 150 个、精排留 20 个） | 1.9% | −67% |

上下文说明是用 LLM 读整篇文档后为每块写的；借助 prompt caching（整篇文档只算一次），一次性成本约每百万文档 token \$1.02。

### 3. Rerank：准，但每个候选都要过一次模型

| | bi-encoder | cross-encoder | ColBERT（late interaction） |
|---|---|---|---|
| 怎么算 | 查询、文档各一个向量，内积 | 查询和文档拼起来过模型，输出一个分 | 查询、文档每个 token 一个向量，MaxSim 求和 |
| 文档侧能离线算吗 | 能 | 不能 | 能 |
| 在线计算 | 一次查询编码 + ANN | $K$ 次前向，每次 $q + c$ 个 token | 一次查询编码 + $K$ 次 MaxSim |
| 存储 | 每块一个向量 | 不需要索引 | 每个 token 一个向量 |

**cross-encoder 的在线开销**：$K = 150$、$q = 32$、$c = 300$：每个查询要过 $150 \times 332 \approx 5$ 万 token 的前向。ColBERT 论文表 1（MS MARCO，精排 1000 个候选，作者自测）：BERT-base 精排 10,700 ms、每个查询 97T FLOPs；ColBERT 61 ms、7B FLOPs。所以 cross-encoder 只能放在最后一步、只看几十到一两百个候选。

**ColBERT**（[arXiv 2004.12832](https://arxiv.org/abs/2004.12832)）：查询的每个 token 去文档里找最相似的 token，再求和：

$$
S(q, d) = \sum_{i \in q} \max_{j \in d} E_{q_i} \cdot E_{d_j}^{\top}
$$

文档的 token 向量离线算好，在线只编码查询（填充到 32 个 token、每个 128 维）。代价是存储：每个 token 一个向量。ColBERTv2（[arXiv 2112.01488](https://arxiv.org/abs/2112.01488) §3.3）把每个向量存成「质心 id + 1 或 2 bit / 维的残差」，每个向量 20 或 36 字节（原版 16 bit 是 256 字节）。按第 1 节的语料：$1.14\times10^9$ token × 36 字节 ≈ 41 GB，是单向量 fp32 方案（13.7 GB）的 3 倍。

### 4. 拼进 prompt 的 token 账：在 agent 里会被反复重发

一次检索拼进 $k$ 块，每块 $c'$ 个 token（含标题、来源等元数据）：$k = 20$、$c' = 300$ 是 6000 token。单轮问答里这只是一次输入：按 Sonnet 4.5 的输入价 \$3/MTok，\$0.018。

agent 里不一样：检索结果作为工具结果追加进上下文，之后每一轮都重发（见 [编排](/agent/orchestration) 第 1 节）。在 50 轮任务的第 10 轮检索，这 6000 token 还要被重发 40 次：

1. 不缓存：$6000 \times 41 \times 3 / 10^6 = \$0.74$。
2. 缓存：写一次 $6000 \times 3.75 / 10^6 = \$0.023$，之后 40 次命中 $6000 \times 40 \times 0.30 / 10^6 = \$0.072$，共约 \$0.095。

所以 agent 里的 RAG 要：$k$ 取小一点、rerank 后只留真正相关的；结果用过之后可以从上下文里删掉或换成摘要（但删中间的内容会破坏后面的缓存前缀，见 [上下文管理](/agent/context-management)）；或者交给 sub-agent 去检索、只回传结论。

### 5. 在线延迟账

一次检索的关键路径（量级是假设，按常见部署估）：

| 步骤 | 延迟量级 | 说明 |
|---|---|---|
| 查询改写（可选，LLM 调用） | 几百 ms | 多轮对话里把「它」换成实体 |
| 查询 embedding（API） | 几十 ms | 自建 GPU 服务可到几 ms |
| ANN 搜索 | 几 ms | HNSW，百万到千万级 |
| BM25 | 几 ms 到几十 ms | 和 ANN 并行 |
| RRF 融合 | 可忽略 | |
| cross-encoder rerank（$K = 50$–150） | 几十到几百 ms | 和候选数成正比 |
| 拼进 prompt 后的 prefill | 和 $k \cdot c'$ 成正比 | 命中缓存时很小 |

两个结论：dense 和 BM25 并行跑；延迟的大头通常是改写和 rerank 两次模型调用，要省就先砍改写（只在多轮、有指代时做），再按延迟预算定 $K$。

### 6. 长期记忆：写入、检索、遗忘

RAG 查的是外部文档；长期记忆存的是 agent 自己的经历：用户偏好、做过的决定、项目约定。

**分层（MemGPT，[arXiv 2310.08560](https://arxiv.org/abs/2310.08560)）**：把上下文窗口当内存、外部存储当磁盘。

1. 主上下文（窗口内）分三段：系统指令；工作上下文（固定大小、只能通过函数调用改写，放用户和任务的关键事实）；FIFO 消息队列（第一个位置放被挤出消息的递归摘要）。
2. 外部上下文：recall storage 存全部消息历史，archival storage 是可读写的任意文本库（论文用 PostgreSQL + pgvector 的 HNSW 索引）。
3. 内存压力：token 数到窗口的 70% 时插一条「内存压力」警告，让模型自己把要紧的东西存进档案；到 100% 时挤出约一半消息并重写递归摘要（§2.2）。
4. 模型通过函数调用搜索和写入外部存储，就像操作系统按需换页。Letta（MemGPT 的产品化）把工作上下文做成 memory blocks（带标签和字符上限），档案提供 `archival_memory_insert` / `archival_memory_search` 工具。

**检索打分（Generative Agents，[arXiv 2304.03442](https://arxiv.org/abs/2304.03442)）**：

$$
\text{score} = \alpha_{\text{rec}}\cdot\text{recency} + \alpha_{\text{imp}}\cdot\text{importance} + \alpha_{\text{rel}}\cdot\text{relevance}
$$

1. recency：距上次被取用的小时数 $h$ 的指数衰减 $0.995^{h}$。
2. importance：写入时让 LLM 打 1–10 分（刷牙是 1，分手是 10）。
3. relevance：查询和记忆 embedding 的余弦。
4. 三项各自 min-max 归一化到 [0, 1]，论文里权重都取 1。

衰减算一笔：$0.995^{24} = 0.887$（一天），$0.995^{168} = 0.431$（一周），$0.995^{720} = 0.027$（一个月）。一个月没被用到的记忆，新近度几乎为 0，只能靠重要度和相关度被取回；被取用一次，计时就重置。

**写入**：不是什么都存。每次会话结束（或每 $n$ 轮）让模型从对话里抽取值得记的事实，和已有记忆比对：新事实就加，和旧事实冲突就更新旧的（用户搬家了，旧地址作废），重复的不加。

**遗忘**：

1. 衰减：靠上面的 recency 让旧记忆自然排不上来。
2. 合并：多条相关记忆压成一条更高层的总结（Generative Agents 的 reflection：最近事件的重要度之和超过 150 时触发）。
3. 删除：冲突时以新的为准；用户要求删除时真删（合规要求）。

**用记忆代替整段历史的效果**（Mem0，[arXiv 2504.19413](https://arxiv.org/abs/2504.19413) 表 2，LOCOMO 长对话基准，作者自测）：

| | 每次带进上下文的 token | 总延迟 p95 | LLM 评审得分 |
|---|---|---|---|
| 全量上下文 | 26,031 | 17.12 s | 72.9% |
| Mem0（抽取后的记忆） | 1,764 | 1.44 s | 66.9% |

token 降到约 1/15、延迟降到约 1/12，代价是约 6 个点的准确率。上下文放得下又不在乎成本时，全量历史仍然最准；记忆系统换的是成本和延迟。

## 面试追问

::: details Q：既然有了 embedding 检索，为什么还要 BM25？
dense 检索对精确字符串不敏感：错误码、函数名、版本号在 embedding 空间里和相似的串挨得很近，排不到最前。BM25 正好擅长这些，但对同义改写无能为力。两路互补，用 RRF 按名次融合。Anthropic 的实验里在 contextual embedding 之上再加 BM25，top-20 失败率从 3.7% 降到 2.9%。
:::

::: details Q：RRF 为什么用名次不用分数？$k = 60$ 是怎么来的？
两路分数尺度不同：余弦在 0–1，BM25 没有上界、还随查询长度变，直接相加会被 BM25 主导。名次没有尺度问题。$k = 60$ 是 Cormack 等人预实验定的，作用是压平头部差距（不加时第 1 名是第 2 名的两倍），实验显示取 0 到 500 结果差别不大。
:::

::: details Q：rerank 用 cross-encoder 还是 ColBERT？
cross-encoder 最准（查询和文档在每一层都能交互），但文档侧不能预计算，每个候选都要一次完整前向，只能精排几十到一两百个。ColBERT 把文档每个 token 的向量离线存好，在线只算 MaxSim，ColBERT 论文里比 BERT 精排快约 175 倍（61 ms vs 10.7 s），代价是每个 token 一个向量的存储（v2 压到每个 20–36 字节）。常见组合：两路召回 → ColBERT 或轻量 cross-encoder 粗排 → 大 cross-encoder 精排 top-20。
:::

::: details Q：块切多大？
没有通用值。太大语义被平均、浪费 token；太小语义不完整。经验是几百 token，按文档结构切（代码按函数、文档按标题），重叠一小段防止关键句被切断。最终用自己数据上的 recall@k 选。框架默认：LlamaIndex 1024 token、重叠 20；LangChain 4000 字符、重叠 200。
:::

::: details Q：agent 里做 RAG 和单轮问答有什么不同？
检索结果会留在上下文里，被之后每一轮重发。50 轮任务第 10 轮拼进 6000 token，不缓存要再付 40 次，约 \$0.74；缓存后约 \$0.095。所以 $k$ 要小、rerank 要狠；或者让 sub-agent 去检索、只回传结论；检索也可以作为工具让 agent 自己决定何时调、查什么。
:::

::: details Q：长期记忆和 RAG 有什么区别？
RAG 查的是外部、基本只读的文档；记忆是 agent 自己写的、会变的。多出来三件事：写什么（从对话里抽事实，和旧记忆比对去重、处理冲突）、怎么取（相关度之外还要新近度和重要度）、怎么忘（衰减、合并、删除冲突的旧事实）。
:::

::: details Q：用记忆系统代替把全部历史塞进窗口，值不值？
看成本和延迟约束。Mem0 的实验里，每次只带约 1800 token 的抽取记忆，延迟 p95 从 17 s 降到 1.4 s，但准确率比全量历史低约 6 个点。窗口放得下、预算够时全量历史更准；对话极长、调用频繁时用记忆。两者也可以结合：近期原文 + 远期记忆。
:::

## 手撕

切块 → embed → 两路召回 → RRF → top-k。embedding 用词哈希代替真实模型（用 `zlib.crc32`，结果跨进程稳定），BM25 按上面的公式：

```python
import math, zlib
import numpy as np
from collections import Counter

def chunk(tokens, size=8, overlap=2):
    step = size - overlap                       # 每块往前走 size - overlap
    return [tokens[i:i + size] for i in range(0, max(len(tokens) - overlap, 1), step)]

def embed(tokens, d=256):
    v = np.zeros(d)                             # 玩具 embedding：词哈希到 d 维
    for t in tokens:
        v[zlib.crc32(t.encode()) % d] += 1.0
    return v / (np.linalg.norm(v) + 1e-9)       # 归一化后内积 = 余弦

def bm25_scores(query, docs, k1=1.2, b=0.75):
    N, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    out = []
    for d in docs:
        tf, s = Counter(d), 0.0
        for q in query:
            if q in tf:
                idf = math.log(1 + (N - df[q] + 0.5) / (df[q] + 0.5))
                s += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return np.array(out)

def rrf(rankings, k=60):
    score = Counter()
    for r in rankings:
        for rank, i in enumerate(r, start=1):
            score[i] += 1.0 / (k + rank)
    return [i for i, _ in score.most_common()]

def retrieve(query, chunks, E, top_k=3, n_cand=10):
    q = query.split()
    dense = np.argsort(-(E @ embed(q)))[:n_cand]
    sparse = np.argsort(-bm25_scores(q, chunks))[:n_cand]
    return rrf([dense.tolist(), sparse.tolist()])[:top_k]   # 真实系统这里再接 cross-encoder

text = ("the sandbox uses firecracker microvm with a guest kernel . "
        "gvisor intercepts syscalls in user space . containers share the host kernel . "
        "rate limits are enforced per tenant with a token bucket . "
        "pass at k is estimated without bias from n samples .")
chunks = chunk(text.split(), size=8, overlap=2)
E = np.stack([embed(c) for c in chunks])                 # 离线：每块一个向量
for i in retrieve("token bucket rate limits", chunks, E):
    print(i, " ".join(chunks[i]))
# 3 share the host kernel . rate limits are
# 4 limits are enforced per tenant with a token
# 5 a token bucket . pass at k is
```

常见变体：把 `embed` 换成真实模型、把暴力内积换成 HNSW 索引；加一步 cross-encoder 精排；给记忆检索加上 $0.995^h$ 的新近度项和重要度项。

## 参考

- [Robertson & Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond (2009)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)：BM25 公式与参数范围
- [Elasticsearch similarity 设置](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)：$k_1 = 1.2$、$b = 0.75$；[RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)：`rank_constant` 60
- [Cormack, Clarke, Büttcher, Reciprocal Rank Fusion (SIGIR'09)](http://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)：公式、$k = 60$、表 1
- [Anthropic：Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)（2024-09-19）：失败率、rerank 150 → 20、成本
- [ColBERT (arXiv 2004.12832)](https://arxiv.org/abs/2004.12832)：MaxSim、表 1 延迟和 FLOPs；[ColBERTv2 (arXiv 2112.01488)](https://arxiv.org/abs/2112.01488)：残差压缩
- [HNSW (arXiv 1603.09320)](https://arxiv.org/abs/1603.09320)：参数和每元素内存
- [OpenAI：New embedding models](https://openai.com/index/new-embedding-models-and-api-updates/)（2024-01-25）：维度、截短；[text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small) 价格（2026-10-13 查）
- [LlamaIndex：Basic strategies](https://developers.llamaindex.ai/python/framework/optimizing/basic_strategies/basic_strategies/)：默认块大小
- [MemGPT (arXiv 2310.08560)](https://arxiv.org/abs/2310.08560)：分层记忆、内存压力；[Letta memory blocks](https://docs.letta.com/v1-sdk/memory/memory-blocks/)、[archival memory](https://docs.letta.com/v1-sdk/memory/archival-memory/)
- [Generative Agents (arXiv 2304.03442)](https://arxiv.org/abs/2304.03442)：检索打分、reflection
- [Mem0 (arXiv 2504.19413)](https://arxiv.org/abs/2504.19413)：表 2 token、延迟、准确率
