---
title: 用户长序列建模：DIN、SIM、TWIN 的计算与存储账
status: draft
tags: [din, target-attention, sim, twin, jagged-tensor, long-sequence]
difficulty: 4
order: 9
related: [/recsys/ranking-serving, /recsys/generative-rec, /recsys/ranking-model-compute, /recsys/embedding-lookup]
section: rs-serve
recsys: [rs-k-seq, rs-s-infer]
---

# 用户长序列建模：DIN、SIM、TWIN 的计算与存储账

> target attention 的成本是「候选数 × 序列长度」，序列从几百拉到几万，就得先检索再注意力、把物品侧的计算预先缓存、按真实长度而不是最长长度算

## 一句话结论

DIN 的 target attention 让每个候选对用户的每条历史行为算一个权重，成本是 $m \times L \times c_a$：本页假设 $m = 500$ 个候选、每对 $3.7\times10^4$ FLOPs，$L = 10^4$ 时每请求 $1.85\times10^{11}$ FLOPs，没法上线。SIM 把它拆成两级：GSU 从上万条行为里按类目（hard search）或内积（soft search）挑出约 100 条，ESU 只在这 100 条上做注意力，同样的例子降到约 $2.5\times10^9$，便宜 74 倍；SIM 线上序列最长 54000，类目索引（UBT）共 22 TB，比只能处理 1000 条的 MIMN 只多 5 ms。TWIN 指出两级用的相关性不一致（SIM hard 的 top-100 里只有 40 条是 ESU 眼中真正的 top-100），于是让 GSU 也用同一个 target attention，并把物品固有特征的投影每 15 分钟预计算一次、缓存 80 亿个视频，用户×物品交叉特征压成一维 bias，关键瓶颈省掉 99.3%，命中率提到 94/100。存储上，$10^4$ 条行为每条 16 字节，3.46 亿日活就是 55 TB（本页假设）；TWIN-V2 离线聚类压到 10%。批内序列长度差几个数量级，用 jagged tensor 按真实长度算，注意力成本从 $B L_{\max}^2$ 降到 $\sum_i L_i^2$。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $L$ | 一个用户的行为序列长度（条） | $10^3$～$10^5$ |
| $m$ | 一个请求里要打分的候选数 | 500（假设） |
| $d$ | 物品 embedding 维度 | 64（假设） |
| $c_a$ | DIN 注意力 MLP 给一对（行为，候选）打分的 FLOPs | $3.7\times10^4$（见第 1 节） |
| $K$ | GSU 选出的行为条数，也就是 ESU 的输入长度 | 100 |
| $h,\ d_k$ | 注意力头数、每头维度 | 4、32（假设） |
| $Q$ | 峰值 QPS | $10^5$（假设） |
| $U$ | 日活用户数 | $3.46\times10^8$（TWIN 报告的快手日活） |
| $b$ | 每条行为存储的字节数 | 16（假设） |
| $B$ | batch 里的样本数 | 4 |

1 GFLOP = $10^9$ FLOPs；1 TB = $10^{12}$ 字节。

### 1. DIN：target attention 的成本是 m × L

DIN 让用户兴趣随候选变化（[arXiv 1706.06978](https://arxiv.org/abs/1706.06978) 式 (3)）：

$$
v_U(A) = \sum_{j=1}^{L} a(e_j, v_A)\, e_j
$$

$e_j$ 是第 $j$ 条行为的 embedding，$v_A$ 是候选 $A$ 的 embedding，$a(\cdot)$ 是一个小 MLP，输入是 $e_j$、$v_A$ 和它们的交互项，输出一个权重。注意权重**不做 softmax**：DIN 想保留兴趣的强度，用户 90% 的历史是衣服时，T 恤候选拿到的权重和应该比手机候选大。

**成本**：每个候选都要和每条行为算一次 $a(\cdot)$：

$$
\text{FLOPs} = m \cdot L \cdot c_a
$$

假设 $a(\cdot)$ 的输入是 $e_j$、$v_A$、$e_j \odot v_A$ 拼成的 $3d = 192$ 维，过 $192 \to 80 \to 40 \to 1$：

1. $c_a = 2 \times (192 \times 80 + 80 \times 40 + 40) = 37{,}200 \approx 3.7\times10^4$ FLOPs。
2. $L = 1000$：$500 \times 1000 \times 3.7\times10^4 = 1.85\times10^{10}$ FLOPs / 请求。
3. $L = 10^4$：$1.85\times10^{11}$；乘 $Q = 10^5$ 是 $1.85\times10^{16}$ FLOP/s，按每卡有效 $1.1\times10^{13}$ FLOP/s（见 [搜推全景](/recsys/funnel-overview)）要约 1700 张卡，只算这一个模块。

所以 DIN 一类模型的序列长度停在几百到一千（HSTU 论文统计 DIN 用 1000，BST 用 20）。TWIN 也说 target attention 的成本把它的可用长度限制在最多几百。

### 2. SIM：先检索、再注意力

SIM（[arXiv 2006.05639](https://arxiv.org/abs/2006.05639)）把一步拆成两级：

1. **GSU**（General Search Unit）：从 $L$ 条行为里挑出和候选最相关的 $K$ 条。两种做法：
   - hard search：只取和候选**同类目**的行为，不用算，查索引；
   - soft search：用预训练 embedding 的内积 $r_j = (W_b e_j)\cdot(W_a v_A)$ 打分取 top-$K$。
2. **ESU**（Exact Search Unit）：在这 $K$ 条上做 DIN 式的 target attention。

**成本**（soft search，内积 $2d = 128$ FLOPs / 对）：

$$
\text{FLOPs} = \underbrace{m \cdot L \cdot 2d}_{\text{GSU}} + \underbrace{m \cdot K \cdot c_a}_{\text{ESU}}
$$

$L = 10^4$、$K = 100$：GSU $500 \times 10^4 \times 128 = 6.4\times10^8$，ESU $500 \times 100 \times 3.7\times10^4 = 1.85\times10^9$，合计 $2.5\times10^9$。对比第 1 节的 $1.85\times10^{11}$，便宜 74 倍。

**线上实际用的是 hard search**。离线实验里 soft search 的 top-$K$ 和 hard search 极为相似（电商里同类目的商品大多相似），soft 的效果只略好，权衡收益和资源后上了 hard search。配套的存储叫 UBT（User Behavior Tree），是 Key-Key-Value 结构：

$$
\text{用户 id} \to \text{类目 id} \to [\text{该类目下的行为}]
$$

1. 每天离线更新；分布式存储，总共 22 TB。
2. 一个请求的几百个候选通常只落在不到 20 个类目里，每个类目的子序列截断到 200 条（原始长度一般不到 150），所以一个请求最多取约 $20 \times 200 = 4000$ 条行为，网络流量有上界。
3. 结果：序列最长 54000，是 MIMN（最多 1000）的 54 倍；延迟只比 MIMN 多 5 ms。SIM 报告排序服务的延迟一般要小于 30 ms，峰值每秒超过 100 万用户；上线带来 CTR +7.1%、RPM +4.4%（作者自测）。

### 3. TWIN：让 GSU 和 ESU 用同一个相关性

SIM 的问题：GSU 用类目或预训练 embedding 打分，ESU 用学出来的注意力打分，两者不一致。TWIN（[arXiv 2302.02352](https://arxiv.org/abs/2302.02352)，快手）做了个实验：拿 ESU 当「神谕」在 $10^4$ 条行为上排出真正的 top-100，SIM hard 的 GSU 返回的 100 条里只命中 40 条。GSU 漏掉的，ESU 再怎么分配注意力也补不回来。

TWIN 的 CP-GSU（Consistency-Preserved GSU）直接用和 ESU **同结构、同参数**的多头 target attention 给 $10^4$ 条行为打分，难点是成本。它注意到瓶颈是**把所有行为做线性投影**，于是把每条行为的特征拆两半分别处理：

$$
\alpha_j = \frac{(x_t W_q)\,(x^h_j W_k)^{\top}}{\sqrt{d_k}} + (x^c_j W_c)\,\beta
$$

1. $x^h_j$：**物品固有特征**（video id、作者、时长、话题），和是谁看的无关。投影 $x^h_j W_k$ 对同一个视频在所有用户的序列里都一样，所以**预计算、缓存**：每 15 分钟用最新参数把候选池里所有视频的投影刷一遍。
2. $x^c_j$：**用户×物品交叉特征**（点击时间、播放时长、互动），每个用户每个视频只出现一次，没法缓存。TWIN 把每个交叉特征投影成**一维**，作为注意力分数里的 bias 项，$\beta$ 是可学的权重。
3. $x_t$ 是候选（target）的固有特征，$W_q, W_k, W_c$ 是投影矩阵。

**算一笔**（本页假设：固有特征 512 维、交叉特征 64 维、$h \cdot d_k = 4 \times 32 = 128$，$L = 10^4$）：

1. 不拆：每请求投影 $2 \times 10^4 \times (512 + 64) \times 128 = 1.47\times10^9$ FLOPs。
2. 拆后：固有部分变成按 video id 查表；交叉部分 $2 \times 10^4 \times 64 = 1.28\times10^6$ FLOPs。投影成本省掉 99.9%。TWIN 报告线上实际省 99.3%。
3. 打分本身还在：$m \cdot L \cdot h d_k \cdot 2 = 500 \times 10^4 \times 128 \times 2 = 1.28\times10^9$，和 SIM soft search 的 GSU 同量级。

**缓存多大**：TWIN 按频次截掉长尾，把候选池限制在 80 亿个视频，这样 15 分钟能刷完一轮，能覆盖 97% 的线上请求。按本页假设每个视频存 128 维 fp16 = 256 字节，80 亿个是 2 TB。

结果：CP-GSU 返回的 100 条命中真正 top-100 中的 94 条（没到 100 是因为 15 分钟的缓存延迟）；部署在快手排序系统，服务 3.46 亿日活，峰值每秒 3000 万个视频。ESU 只有 100 条，全部实时算、用最新同步的参数。

**TWIN-V2**（[arXiv 2407.16357](https://arxiv.org/abs/2407.16357)）：TWIN 用最近 $10^4$ 条，只覆盖最近 3–4 个月。V2 把用户全生命周期（最多 $10^5$ 条）离线做层次聚类，每个簇最多 20 条（平均约 10 条），一个簇聚合成一个虚拟物品，序列压到约 10%，GSU 的输入还是约 $10^4$。存储和计算都省 90%，聚类每 2 周全量重跑一次。

### 4. 存储与带宽：存 ID，不存向量

$$
\text{存储} = U \cdot L \cdot b
$$

1. 每条行为存 video id 8 B、时间戳 4 B、播放时长等 4 B，$b = 16$ 字节（假设）。$L = 10^4$：每个用户 160 KB。
2. $U = 3.46\times10^8$：$3.46\times10^8 \times 1.6\times10^5 = 5.5\times10^{13}$ B = 55 TB。SIM 报告的 UBT 是 22 TB，同一量级。
3. 存向量会怎样：每条存 64 维 fp16 = 128 字节，是存 ID 的 8 倍，440 TB。而且同一个视频在千万用户的序列里重复，向量应该按 id 去 embedding 表里查一次，而不是每个用户存一份。
4. 带宽：每个请求如果拉全量 $10^4$ 条，160 KB × $10^5$ QPS = 16 GB/s。SIM 按类目只拉需要的子序列，TWIN-V2 聚类压到 10%，都是在压这个数。

### 5. jagged tensor：按真实长度算

一个 batch 里用户的序列长度差几个数量级。补齐（padding）到最长的那条：

$$
\text{padding 后的注意力成本} \propto B \cdot L_{\max}^2,\qquad \text{按真实长度} \propto \sum_{i=1}^{B} L_i^2
$$

算一笔：$B = 4$，长度 $[100, 1000, 8000, 500]$。

1. token 数：补齐后 $4 \times 8000 = 32000$，真实 9600，70% 是空位。
2. 注意力：补齐 $4 \times 8000^2 = 2.56\times10^8$；真实 $100^2 + 1000^2 + 8000^2 + 500^2 = 6.53\times10^7$，省 3.9 倍。

**jagged tensor** 的存法：把所有用户的行为首尾相接放进一个一维的 `values`，再用 `offsets` 记每个用户从哪开始：上面的例子 `offsets = [0, 100, 1100, 9100, 9600]`，第 $i$ 个用户是 `values[offsets[i]:offsets[i+1]]`。TorchRec 的 `JaggedTensor` 和 FBGEMM 的 jagged 算子就是这个格式。

HSTU（[arXiv 2402.17152](https://arxiv.org/abs/2402.17152)）把注意力写成按真实长度分组的 GEMM（类似 FlashAttention 的融合 kernel），报告这一项就有 2–5 倍吞吐提升；训练时再用 Stochastic Length 随机截短：$\alpha = 1.6$ 时长度 4096 的序列多数时候被截到 776，去掉 80% 以上的 token，主任务 NE 变差不超过 0.002（0.2%，作者自测）。

### 6. 线上怎么省：清单

1. **用户侧只算一次**：一个请求里 $m$ 个候选共享同一条序列，序列的 key / value 投影算一次，候选只算 query。生成式里的 M-FALCON 是这个思路的极致，见 [生成式推荐](/recsys/generative-rec)。
2. **物品侧预计算**：只和物品有关的投影离线算好放缓存（TWIN），代价是几分钟到十几分钟的过期。
3. **两级检索**：GSU 便宜地从上万条里挑 100 条，ESU 精算。GSU 要和 ESU 尽量一致。
4. **存 ID 不存向量、按需拉取**：KKV 索引只拉候选类目的子序列；离线聚类压缩（TWIN-V2）。
5. **按真实长度算**：jagged tensor 和对应 kernel；训练时随机截短。

## 面试追问

::: details Q：DIN 的注意力权重为什么不做 softmax？
softmax 让权重和固定为 1，丢掉了兴趣的强度。DIN 的例子：用户历史 90% 是衣服、10% 是电子产品，T 恤候选激活了大部分历史，应该得到比手机候选更大的 $v_U$。不归一化，$\sum_j w_j$ 本身就近似反映了兴趣有多强。
:::

::: details Q：为什么 target attention 不能直接用在一万条行为上？
成本是 $m \cdot L \cdot c_a$，和序列长度、候选数都成正比。$m = 500$、$L = 10^4$、每对 $3.7\times10^4$ FLOPs，每请求 $1.85\times10^{11}$ FLOPs，$10^5$ QPS 下光这个模块就要上千张卡。所以先用 GSU 检索出 100 条。
:::

::: details Q：SIM 线上为什么用 hard search 而不是更好的 soft search？
离线发现两者选出的 top-$K$ 极为相似（电商同类目商品大多相似），soft 只略好。hard search 不用算分，只要一个「用户 → 类目 → 行为」的 KKV 索引，一个请求只拉不到 20 个类目、每个最多 200 条，存储和带宽都可控。
:::

::: details Q：TWIN 为什么可以缓存物品固有特征的投影，交叉特征却不行？
固有特征（id、作者、时长）和用户无关，同一个视频在所有用户的序列里投影结果一样，算一次能被成千上万个请求复用。交叉特征（这个用户什么时候看的、看了多久）每个用户每个视频只出现一次，没有复用，缓存没意义，所以把它压成一维 bias 来省算力。
:::

::: details Q：GSU 和 ESU 不一致到底有多大影响？
TWIN 用 ESU 在 $10^4$ 条上排出的 top-100 当真值：SIM hard 的 GSU 返回 100 条只命中 40 条，TWIN 命中 94 条。命中不了的那 60 条，ESU 根本看不到。
:::

::: details Q：一万条行为的序列怎么存？要存多大？
存 ID 和少量行为属性，不存向量，每条约 16 字节，一个用户 160 KB，3.46 亿用户 55 TB（本页假设，SIM 的 UBT 是 22 TB）。按 用户 → 类目 → 行为 组织，请求只拉需要的子序列；TWIN-V2 再离线聚类压到 10%。
:::

::: details Q：jagged tensor 解决什么问题？
批内序列长度差很大，补齐到最长会浪费大部分算力：长度 $[100, 1000, 8000, 500]$ 补齐后 70% 是空位，注意力成本多 3.9 倍。jagged 用 values + offsets 按真实长度存，配合按段计算的 kernel，计算量按 $\sum_i L_i^2$ 而不是 $B L_{\max}^2$。
:::

## 手撕

jagged 存法 + 两级（GSU 内积 top-$K$，ESU softmax 注意力）的 target attention：

```python
import torch

torch.manual_seed(0)
d, K = 16, 4                                   # embedding 维度、GSU 选出的行为数

# 一个 batch 两个用户，行为序列长度 6 和 3：jagged 存法 = values + offsets
lengths = torch.tensor([6, 3])
offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])  # [0, 6, 9]
values = torch.randn(int(lengths.sum()), d)    # [9, d]，没有 padding

cand = torch.randn(2, 5, d)                    # 每个用户 m = 5 个候选
Wq, Wk = torch.randn(d, d) / d**0.5, torch.randn(d, d) / d**0.5

def two_stage(u):
    E = values[offsets[u]:offsets[u + 1]]      # 这个用户的 L 条行为 [L, d]
    C = cand[u]                                # [m, d]
    # GSU：内积打分，每个候选取 top-K 条行为
    k = min(K, E.shape[0])
    idx = (C @ E.T).topk(k, dim=1).indices     # [m, k]
    sub = E[idx]                               # [m, k, d]
    # ESU：在 top-K 上做 target attention
    q = C @ Wq                                 # [m, d]
    key = sub @ Wk                             # [m, k, d]
    w = torch.softmax((key @ q.unsqueeze(-1)).squeeze(-1) / d**0.5, dim=1)  # [m, k]
    return (w.unsqueeze(-1) * sub).sum(1)      # 每个候选一个兴趣向量 [m, d]

out = torch.stack([two_stage(u) for u in range(2)])
print(offsets.tolist(), out.shape)
# [0, 6, 9] torch.Size([2, 5, 16])
```

要点：第二个用户只有 3 条行为，`k` 取 $\min(K, L)$，不补齐；GSU 的打分矩阵是 $[m, L]$，这就是 $m \cdot L$ 成本的来源。DIN 版本把 ESU 的 softmax 去掉、把内积换成 MLP。

## 参考

- [Deep Interest Network (DIN)](https://arxiv.org/abs/1706.06978)：式 (3) 的 target attention、不做 softmax 的理由
- [Search-based User Interest Modeling (SIM)](https://arxiv.org/abs/2006.05639)：GSU / ESU、hard / soft search、UBT 22 TB、54000 长度、+5 ms、30 ms 延迟要求
- [TWIN](https://arxiv.org/abs/2302.02352)：GSU / ESU 一致性（40 vs 94）、特征拆分、15 分钟缓存、80 亿视频、97% 覆盖、99.3%、3.46 亿日活与峰值 3000 万 / 秒
- [TWIN V2](https://arxiv.org/abs/2407.16357)：全生命周期层次聚类、簇大小 20、压到 10%、两周全量更新
- [HSTU: Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152)：jagged 注意力 kernel 2–5 倍、Stochastic Length、DLRM 常用序列长度
- [FBGEMM Jagged Tensor Operators](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/overview/jagged-tensor-ops/JaggedTensorOps.html)：values + offsets 的存法和算子
- [TorchRec `jagged_tensor.py`](https://github.com/pytorch/torchrec/blob/main/torchrec/sparse/jagged_tensor.py)：`JaggedTensor` / `KeyedJaggedTensor` 的定义
