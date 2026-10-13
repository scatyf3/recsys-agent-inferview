---
title: 生成式推荐：semantic ID、HSTU、OneRec 的训练与推理
status: draft
tags: [generative-rec, semantic-id, rq-vae, hstu, m-falcon, onerec, beam-search]
difficulty: 4
order: 12
related: [/recsys/long-sequence, /recsys/ann-retrieval, /recsys/ranking-serving, /recsys/llm-for-search, /recsys/funnel-overview]
recsys: [rs-g-sid, rs-g-hstu, rs-g-serve]
---

# 生成式推荐：semantic ID、HSTU、OneRec 的训练与推理

> 把物品变成几个 token、把用户行为变成一条序列，推荐就成了「下一个 token 预测」；难点从特征工程挪到了词表、训练组织和解码

## 一句话结论

生成式推荐有两条线。一条是 **semantic ID + 自回归生成**（TIGER、OneRec）：用 RQ-VAE / RQ-Kmeans 把物品的内容向量做 $L$ 层残差量化，TIGER 用 3 层 × 256 个码字再加 1 位去重位，词表只有 1024 个 token；OneRec 用 3 层 × 8192。模型读用户历史，在 SID 上做 $L$ 步 beam search 直接生成物品，OneRec 一次生成 512 个，替代了召回–粗排–精排的级联，运营成本是原系统的 10.6%，训练 / 推理 MFU 从 4.6% / 11.2% 提到 23.7% / 28.8%。另一条是 **HSTU**（Meta）：把物品和行为交错成一条序列，用 pointwise 注意力替代 DLRM 的特征交叉，生成式地组织训练（每个用户一条序列、所有位置都是目标，比逐条曝光省一个序列长度的因子），在 8192 长度上比 FlashAttention-2 的 Transformer 快 5.3–15.2 倍，模型到 1.5 万亿参数，线上指标 +12.4%。排序侧的推理用 M-FALCON：$m$ 个候选分 micro-batch 塞进同一次前向、共享用户历史的 KV，成本从 $O(m n^2 d)$ 降到约 $O(n^2 d + m n d)$，因此 285 倍 FLOPs 的模型在 1024 / 16384 个候选时 QPS 反而是原 DLRM 的 1.5 / 2.99 倍。卡点：训练侧是稀疏 embedding 的通信、码本一旦确定很难换、RL 要在线生成样本；推理侧是宽 beam、非法 SID 和每个请求都要编码的长历史。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $N$ | 物品数 | 十亿级 |
| $L$ | semantic ID 的层数（不含去重位） | 3 |
| $K$ | 每层码本的码字数 | 256（TIGER）、8192（OneRec） |
| $z,\ r_l$ | 物品的潜向量、第 $l$ 层的残差 | $r_0 = z$ |
| $e^{(l)}_k$ | 第 $l$ 层码本的第 $k$ 个码字 | |
| $n$ | 用户历史序列长度（token） | 2000（假设） |
| $m$ | 排序要打分的候选数 | 1024 |
| $b_m$ | M-FALCON 的 micro-batch 大小 | 128（假设） |
| $d$ | 模型维度 | 512 |
| $b$ | beam 宽 / 一次生成的物品数 | 512（OneRec） |

### 1. semantic ID：为什么、怎么做

**为什么不用原子 ID**：十亿级、每天新增几千万的物品 ID 做不了 softmax 的输出词表，embedding 表也随物品数线性增长，新物品没训练过就没有像样的 embedding。semantic ID 用 $L$ 个 token 表示一个物品，每个 token 来自大小为 $K$ 的码本：

$$
\text{可表示的物品数} = K^L,\qquad \text{词表大小} = L \cdot K
$$

1. TIGER：$K = 256$、$L = 3$，$256^3 = 1.68\times10^7$；再加 1 位去重位，词表 $4 \times 256 = 1024$ 个 token，而原子 ID 要 $N$ 行。
2. OneRec：3 层、$K = 8192$，$8192^3 = 5.5\times10^{11}$。报告里把 $K$ 从 8192 加到 32768，时长类指标明显提升。
3. 相似物品共享前缀（第一层是粗类目，后面逐层细化），知识能在相似物品间传递；新物品只要有内容向量就能编码出 SID，不需要先攒交互。

**残差量化（RQ）**：

$$
c_l = \arg\min_k \lVert r_l - e^{(l)}_k \rVert,\qquad r_{l+1} = r_l - e^{(l)}_{c_l},\qquad \text{SID} = (c_0, c_1, \dots, c_{L-1})
$$

算一个二维的例子：$z = (0.9, 0.3)$。

1. 第 0 层码本 $\{(1, 0),\ (0, 1)\}$：到 $(1,0)$ 的距离平方 $0.01 + 0.09 = 0.10$，到 $(0,1)$ 的是 $0.81 + 0.49 = 1.30$，取 $c_0 = 0$，残差 $r_1 = (-0.1, 0.3)$。
2. 第 1 层码本 $\{(0, 0.25),\ (-0.25, 0)\}$：到第一个 $0.01 + 0.0025 = 0.0125$，到第二个 $0.0225 + 0.09 = 0.1125$，取 $c_1 = 0$，残差 $r_2 = (-0.1, 0.05)$。
3. SID $= (0, 0)$，重建 $(1, 0.25)$，误差平方 0.0125。每多一层，误差就再被下一层的码本削一次。

**RQ-VAE**（TIGER 的做法，源自 [Lee et al. 2022](https://arxiv.org/abs/2203.01941)）：编码器把内容向量（TIGER 用 Sentence-T5 的 768 维）压到 32 维的 $z$，做上面的残差量化，解码器再重建，联合训练：

$$
\mathcal{L} = \lVert x - \hat x \rVert^2 + \sum_{l} \Big( \lVert \text{sg}[r_l] - e^{(l)}_{c_l} \rVert^2 + \beta \lVert r_l - \text{sg}[e^{(l)}_{c_l}] \rVert^2 \Big)
$$

sg 是 stop-gradient，$\beta = 0.25$。第一项管重建，第二项把码字拉向残差，第三项让编码器的输出靠近码字。容易出现**码本坍缩**（大部分输入落到少数几个码字上），TIGER 用第一个 batch 的 k-means 中心初始化码本，并训练到码本使用率至少 80%。

**RQ-Kmeans**（OneRec）：不训练编解码器，直接对每层的残差做 k-means 当码本。OneRec 报告在 3 层 × 8192 的设置下，比 RQ-VAE 重建误差低 25.18%，三层码本使用率都是 1.0，分布更均匀。

**碰撞和非法 ID**：

1. 碰撞：几个物品量化到同一个 SID，就在末尾加一位序号区分，$(12, 24, 52)$ 变成 $(12, 24, 52, 0)$ 和 $(12, 24, 52, 1)$。
2. 非法 ID：$L = 4$、$K = 256$ 时可能的 ID 有 $256^4 \approx 4.3\times10^{12}$ 个，真实物品只占极小一部分，自回归生成可能走到不对应任何物品的前缀。TIGER 统计 beam search 取 top-20 时有 0.3%–6% 是非法的。解法是解码时用**前缀树**约束，每一步只允许走到真实存在的前缀（见手撕）。

### 2. HSTU：把 DLRM 换成一个序列模型

HSTU（[arXiv 2402.17152](https://arxiv.org/abs/2402.17152)）把用户看过的物品和对它们的行为交错成一条序列 $\Phi_0, a_0, \Phi_1, a_1, \dots$，排序任务就是在 $\Phi_{i+1}$ 的位置预测 $a_{i+1}$，召回是预测下一个 $\Phi$。每层三步：

$$
U, V, Q, K = \text{Split}\big(\phi_1(f_1(X))\big)
$$

$$
A(X)V(X) = \phi_2\big(Q K^{\top} + \text{rab}^{p,t}\big)\, V
$$

$$
Y(X) = f_2\big(\text{Norm}(A(X)V(X)) \odot U\big)
$$

1. $f_1, f_2$ 是单个线性层，$\phi_1, \phi_2$ 都是 SiLU，rab 是带位置和时间的相对注意力偏置。
2. **不做 softmax**（pointwise 注意力）：推荐的词表是十亿级、不断变化的，softmax 归一化会抹掉「用户对这类内容兴趣有多强」的信息（和 DIN 不做 softmax 是同一个理由）。
3. $\odot U$ 的门控同时承担了 DLRM 里特征交叉和 MoE 的角色。
4. 激活更省：注意力之外的线性层从 Transformer 的 6 个减到 2 个，再激进地融合算子，bf16 下每层每 token 的激活从约 $33d$ 降到 $14d$，同样显存能堆 2 倍以上的层数。

**生成式训练省一个序列长度**。传统 DLRM 按曝光组织样本：用户有 $n$ 次曝光就有 $n$ 条样本，每条都要把用户历史编码一遍。HSTU 每个用户只出一条序列，所有位置一起算损失（按 $1/n_i$ 的概率采样用户，比如在一次请求或 session 结束时才产出样本）：

$$
\text{逐条曝光} \propto \sum_i n_i\,(n_i^2 d + n_i d^2),\qquad \text{生成式} \propto \sum_i (n_i^2 d + n_i d^2)
$$

$n = 1000$ 的用户，逐条要把同一段历史编码 1000 遍，生成式编码 1 遍，省 1000 倍。

**稀疏 embedding 的显存**：100 亿词表、512 维，fp32 参数加 Adam 的 $m, v$ 每个数 12 字节，$10^{10} \times 512 \times 12 = 61$ TB。HSTU 用按行的 AdamW，把优化器状态放到 CPU 内存，每个数在 HBM 上从 12 字节降到 2 字节。

**变长序列**：用 jagged 注意力 kernel 按真实长度算（2–5 倍），训练时 Stochastic Length 随机截短，见 [长序列](/recsys/long-sequence)。综合起来在 8192 长度上比 FlashAttention-2 的 Transformer 快 5.3–15.2 倍。

**规模**：指标（召回的 HR@100、排序的 NE）随训练算力呈幂律提升，跨了三个数量级，最大模型的总训练算力接近 GPT-3 / LLaMA-2；模型到 1.5 万亿参数；线上 A/B 主指标 +12.4%（作者自测）。

### 3. 排序推理：M-FALCON

排序要给 $m$ 个候选（上万个）打分，每个候选都要「看」同一段长度 $n$ 的用户历史。朴素做法是每个候选拼上历史各跑一次：

$$
\text{朴素} = O(m\, n^2 d)
$$

M-FALCON（Microbatched-Fast Attention Leveraging Cacheable OperatioNs）：

1. 把 $b_m$ 个候选**同时**接在历史后面，改注意力掩码，让每个候选只看历史和自己、不看别的候选。这样 $b_m$ 个候选的结果和分开算完全一样，成本是 $O((n + b_m)^2 d)$。
2. $m$ 个候选分成 $m / b_m$ 个 micro-batch，历史部分的 K / V **算一次缓存起来**，后面的 micro-batch 只算候选自己的 token。

算一笔（只数注意力的 $n^2$ 量级项，$d$ 相同略去）：$n = 2000$、$m = 1024$、$b_m = 128$。

1. 朴素：$1024 \times 2000^2 = 4.1\times10^9$。
2. 分 micro-batch、不缓存：$8 \times 2128^2 = 3.6\times10^7$。
3. 再缓存历史 KV：历史自身 $2000^2 = 4\times10^6$，加上每个候选看历史 $1024 \times 2128 = 2.2\times10^6$，共约 $6.2\times10^6$，是朴素的 1/660。

论文的结果：比生产 DLRM 复杂 285 倍的模型，在 1024 / 16384 个候选时 QPS 是 DLRM 的 1.50 / 2.99 倍。候选越多，历史的成本被摊得越薄。这和 [排序服务](/recsys/ranking-serving) 里「用户侧只算一次」是同一个思路，推到了注意力层；KV 缓存的概念同 LLM 推理里的 [KV cache](https://scatyf3.github.io/ai-infra-inferview/inference/kv-cache-paged-attention)。

### 4. 生成推理：在 SID 上 beam search

OneRec（[技术报告](https://arxiv.org/abs/2506.13695)）是 encoder-decoder：encoder 读用户的多尺度行为（最长 $10^5$ 条的终身序列先做层次 k-means 压缩），decoder 逐个生成目标物品的 3 个 SID token。一个请求：

1. encoder 编码用户历史一次。
2. decoder 做 $L = 3$ 步 beam search，beam 宽 $b = 512$（报告里 Pass@K 从 8 加到 512 指标一直涨，到 1024 收益很小，权衡成本选了 512）。
3. 每一步要对 $b \times K = 512 \times 8192 = 4.2\times10^6$ 个（前缀，下一个码字）打分取 top-512。
4. decoder 共处理 $b \cdot L = 1536$ 个 token；它们的 cross-attention 都读同一份 encoder 输出，encoder 的 K / V 算一次、所有 beam 共享。
5. 串行步数只有 3，远少于 LLM 生成几百个 token；每步的 batch 很宽，适合 GPU。beam search 的一般实现见 [beam search 手撕](https://scatyf3.github.io/ai-infra-inferview/handson/beam-search)。

**服务**：NVIDIA L20 推理（每台 4 卡、PCIe），TensorRT 编译计算图、cross-attention 和 MoE 写成自定义插件，加上 batching 和 MPS，吞吐提升 5 倍，推理 MFU 28.8%（报告结论部分写的是 28.6%，两处不一致）。部署在快手 / 快手极速版主场景，承接约 25% 的 QPS，App 停留时长 +0.54% / +1.24%（作者自测）。

**为什么成本反而降了**：级联系统里超过 50% 的服务资源花在通信和存储上（各级之间传候选、每级各取一遍特征），排序模型 MFU 只有个位数到十几。一个端到端模型把这些都省了，算力集中在少数大矩阵乘上，OPEX 是原来的 10.6%。

### 5. 训练和推理各卡在哪

| | 卡点 | 现在的做法 |
|---|---|---|
| 训练 | 用户侧仍有大量稀疏特征，embedding 查表和通信重 | OneRec 用快手的 GPU 参数服务器（统一 embedding 表、GPU 缓存、预取）；dense 部分单卡放得下，用 DP + ZeRO-1 + 梯度累积；也在试用 SID 代替视频 id 的稀疏 embedding |
| 训练 | 码本换了，所有物品的 SID 都变，模型要跟着重训 | 码本长期固定，新物品用现有码本编码 |
| 训练 | 数据量大 | OneRec 每天约 180 亿样本、540 亿 token；0.935B 模型约 1000 亿样本收敛 |
| 训练 | 只学曝光过的分布，突破不了旧系统的天花板 | 加 RL（ECPO）：1% 的用户请求走外部推理服务生成 512 个物品、奖励模型打分，训练每 1000 步把参数推给推理服务；RL 用 beam search 采样比 top-$k$ / top-$p$ 好 1.45% |
| 训练 | MoE 负载不均 | DeepSeek 的 loss-free 均衡 |
| 推理 | 非法 SID，RL 之后更多（负 advantage 把概率挤到了非法 token 上） | 前缀树约束解码，或生成后过滤 |
| 推理 | 每步 $b \times K$ 打分 + top-$k$ | 宽 batch 上的融合 kernel |
| 推理 | 每个请求都要编码长历史 | 离线压缩终身序列；同一用户相邻请求的历史大部分相同，可以像 LLM 的 prefix cache 一样复用（思路，报告未提） |
| 推理 | QPS 高、延迟紧 | OneRec 报告推荐系统要扛 40 万以上 QPS；TensorRT、batching、MPS |

## 面试追问

::: details Q：为什么用 semantic ID 而不是原子物品 ID？
原子 ID 的词表是十亿级、每天在变，输出 softmax 和 embedding 表都随物品数线性增长，新物品没有像样的 embedding。SID 用 $L$ 个、每个来自 $K$ 个码字的 token 表示物品，词表只有 $L \cdot K$（TIGER 是 1024），相似物品共享前缀，新物品只要有内容向量就能编码。
:::

::: details Q：RQ-VAE 和 RQ-Kmeans 有什么区别？码本坍缩怎么办？
RQ-VAE 训练编码器、解码器和码本，损失是重建 + 码本 / commitment 两项；RQ-Kmeans 直接对每层残差做 k-means，不训练网络。坍缩是大部分输入落到少数码字上，RQ-VAE 靠 k-means 初始化码本、长时间训练提高使用率；OneRec 报告 RQ-Kmeans 使用率 1.0、重建误差低 25%。
:::

::: details Q：生成出来的 SID 不对应任何物品怎么办？
解码时用前缀树约束：每一步只保留真实存在的前缀，非法分支直接不展开。不约束的话，TIGER 在 top-20 里有 0.3%–6% 是非法的；OneRec 发现加 RL 后非法输出明显增多。
:::

::: details Q：HSTU 的生成式训练为什么比按曝光训练便宜？
按曝光组织时，一个有 $n$ 次曝光的用户产出 $n$ 条样本，每条都把历史编码一遍；生成式每个用户一条序列，所有位置一起算损失，历史只编码一次，省掉一个 $n$ 的因子。实现上是在请求或 session 结束时才产出这个用户的样本。
:::

::: details Q：M-FALCON 为什么和逐个候选算的结果一样？
注意力掩码让每个候选只能看到历史和它自己，看不到同一个 micro-batch 里的其他候选，所以每个候选的输出和单独拼历史算完全相同。省下的是历史部分：它的 K / V 只算一次、被所有候选共享。
:::

::: details Q：OneRec 一个请求要算多少？
encoder 编码一次用户历史；decoder 做 3 步 beam search、beam 宽 512，每步对 $512 \times 8192 \approx 4.2$ M 个候选打分取 top-512，共处理 1536 个 decoder token，cross-attention 共享同一份 encoder 输出。串行步数只有 3，batch 很宽。
:::

::: details Q：生成式推荐为什么能比级联省钱？
级联系统一半以上的资源花在通信和存储上（各级之间传候选、各取一遍特征），排序模型 MFU 只有个位数到十几。端到端生成把这些省掉，算力集中在大矩阵乘上，OneRec 报告 OPEX 是原来的 10.6%，推理 MFU 28.8%。
:::

## 手撕

RQ-Kmeans 编码、碰撞去重位、前缀树约束的 beam search（用随机分布代替 decoder 的一步）：

```python
import numpy as np

rng = np.random.default_rng(0)
N, d, L, Kc = 5000, 16, 3, 32                   # 物品数、向量维度、SID 层数、每层码本大小

def kmeans(x, c, iters=10):
    cent = x[rng.choice(len(x), c, replace=False)].copy()
    for _ in range(iters):
        a = ((x[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for j in range(c):
            if (a == j).any():
                cent[j] = x[a == j].mean(0)
    return cent

# 1. RQ-Kmeans：每层对上一层的残差做 k-means，码字 = 最近中心的编号
x = rng.normal(size=(N, d))
r, books, codes = x.copy(), [], []
for l in range(L):
    C = kmeans(r, Kc)
    c = ((r[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    books.append(C); codes.append(c)
    r = r - C[c]                                  # 残差进入下一层
codes = np.stack(codes, 1)                        # [N, L]
print("relative recon error:", round(float((r ** 2).sum() / (x ** 2).sum()), 3))

# 2. 碰撞：相同 SID 的物品追加一位序号，让 ID 唯一
_, inv, cnt = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
dup = np.zeros(N, dtype=int)
seen = {}
for i, g in enumerate(inv):
    dup[i] = seen.get(g, 0); seen[g] = dup[i] + 1
sid = np.concatenate([codes, dup[:, None]], 1)   # [N, L + 1]，最后一位是去重位
print("colliding items:", int((cnt[inv] > 1).sum()), "max suffix:", dup.max(),
      "unique:", len(np.unique(sid, axis=0)) == N)

# 3. 前缀树约束的 beam search：每一步只允许走到真实存在的前缀
valid = {l: set(map(tuple, codes[:, :l + 1])) for l in range(L)}
def beam_search(logp_fn, beam=4):
    beams = [((), 0.0)]
    for l in range(L):
        cand = []
        for prefix, s in beams:
            lp = logp_fn(prefix)                  # [Kc] 下一个码字的 log 概率
            for t in range(Kc):
                if prefix + (t,) in valid[l]:     # 非法前缀直接跳过
                    cand.append((prefix + (t,), s + lp[t]))
        beams = sorted(cand, key=lambda z: -z[1])[:beam]
    return beams

fake_model = lambda prefix: np.log(rng.dirichlet(np.ones(Kc)))   # 代替 decoder 的一步
out = beam_search(fake_model)
print([b[0] for b in out], all(b[0] in valid[L - 1] for b in out))
# relative recon error: 0.355
# colliding items: 806 max suffix: 2 unique: True
# [(20, 31, 6), (20, 31, 14), (28, 28, 13), (20, 31, 9)] True
```

读输出：随机高斯数据没有聚类结构，3 层 × 32 的残差量化还剩 35.5% 的误差；5000 个物品里 806 个和别人撞了 SID，加一位去重位后全部唯一；约束解码返回的 4 个 SID 都是真实存在的。生产里前缀树存成每层「前缀 → 合法下一码字」的位图，在 GPU 上作为 logits 的掩码。

## 参考

- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)：RQ-VAE 结构与损失、3 × 256 + 去重位、1024 个 token、k-means 初始化、非法 ID 比例
- [Autoregressive Image Generation using Residual Quantization（RQ-VAE）](https://arxiv.org/abs/2203.01941)：残差量化的原始提出
- [HSTU: Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152)：式 (1)–(3)、生成式训练的复杂度、14d vs 33d 激活、60 TB embedding 与按行 AdamW、M-FALCON、5.3–15.2 倍、285 倍 FLOPs 与 1.5 / 2.99 倍 QPS、+12.4%、1.5 万亿参数
- [OneRec Technical Report](https://arxiv.org/abs/2506.13695)：RQ-Kmeans、码本 8192 → 32768、Pass@512、L20 + TensorRT、MFU 与 OPEX、25% QPS、训练数据量、ECPO 与在线生成、非法输出
- [OneRec: Unifying Retrieve and Rank with Generative Recommender](https://arxiv.org/abs/2502.18965)：session 级生成、级联的问题
