---
title: 向量召回：双塔、logQ 校正、HNSW 与 IVF-PQ
status: draft
tags: [two-tower, logq, ann, hnsw, ivf-pq, faiss, gpu-search]
difficulty: 3
order: 10
related: [/recsys/funnel-overview, /recsys/ranking-serving, /recsys/generative-rec, /recsys/embedding-tables]
recsys: [rs-r-tower, rs-r-ann, rs-r-gpu]
---

# 向量召回：双塔、logQ 校正、HNSW 与 IVF-PQ

> 双塔把「用户 × 物品」拆成两个可以分开算的向量，ANN 索引再把「和一亿个向量算内积」变成只算几万次距离；每一步都在拿召回率换内存和延迟

## 一句话结论

双塔召回的打分是 $s(x, y) = \langle u(x), v(y)\rangle$，物品向量离线算好建索引，线上一个请求只跑一次用户塔再做 ANN 检索，代价是不能用用户×物品的交叉特征。训练用 in-batch softmax，热门物品作为负样本出现得太多，要把 logit 改成 $s - \log p_j$（Yi et al. 2019，YouTube 用流式哈希估计 $p_j$，$H = 5000$ 万个桶）。索引方面：暴力检索 $10^8$ 个 128 维 fp32 向量每个查询要读 51 GB；HNSW 每条向量约 $4d + 8M$ 字节（$d = 128$、$M = 32$ 时 768 B，一亿条 77 GB），召回高、延迟低但吃内存；IVF-PQ 把向量压成 $m$ 字节的 PQ 码（$m = 16$ 时压缩 32 倍），只扫 nprobe 个桶，桶数取 $\sqrt{P N}$ 时距离计算从 $10^8$ 次降到 $8\times10^4$ 次，精度损失用原始向量重排补回来。GPU 适合 IVF 扫描和批量暴力检索（批大时变成矩阵乘），不适合 HNSW 的串行图遍历；Faiss 的 GPU k-selection 比之前的 GPU 方案快 8.5 倍，CAGRA 这类 GPU 图索引在大 batch 下吞吐是 HNSW 的 33–77 倍（作者自测）。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $N$ | 索引里的物品数 | $10^8$（假设） |
| $d$ | 向量维度 | 128 |
| $u(x),\ v(y)$ | 用户塔、物品塔输出的向量 | $d$ 维 |
| $B$ | 训练 batch 里的（用户，正样本物品）对数 | 8192（YouTube） |
| $p_j$ | 物品 $j$ 出现在一个随机 batch 里的概率 | 热门 0.5、长尾 $10^{-4}$（假设） |
| $\tau$ | softmax 温度 | 0.05（YouTube） |
| $M$ | HNSW 每个点在上层的最大邻居数，第 0 层是 $2M$ | 32 |
| $K_{\text{IVF}}$ | IVF 的桶数（Faiss 里叫 nlist） | $4\times10^4$ |
| $P$ | 查询时扫的桶数（nprobe） | 16 |
| $m$ | PQ 把向量切成几段，每段用 1 字节编码 | 16 |
| $k$ | 要返回的近邻数 | 1000 |
| $n_q$ | 一批查询的条数 | 1 或 1024 |

1 GB = $10^9$ 字节。

### 1. 双塔：为什么能召回，代价是什么

用户塔 $u(\cdot)$ 只吃用户和上下文特征，物品塔 $v(\cdot)$ 只吃物品特征，打分是内积：

$$
s(x, y) = \langle u(x),\ v(y) \rangle
$$

1. **物品侧离线**：所有物品的 $v(y)$ 提前算好建 ANN 索引。YouTube 的做法是每隔几小时用最新模型对约 1000 万个视频重建一次索引；Instagram 是每天离线生成一次物品向量。
2. **用户侧在线**：请求来了用最新的用户特征跑一次用户塔，拿 $u(x)$ 去索引里找内积最大的 $k$ 个。一个请求只有一次塔的前向加一次检索。
3. **代价**：模型不能用「这个用户和这个物品」的交叉特征（比如用户看过这个作者几次），否则物品向量就和用户有关、没法预先算。交叉特征留给排序。

YouTube 的配置（[Yi et al. 2019](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)）：两个塔都是 $[1024, 512, 128]$ 的三层 DNN，输出 128 维，batch 8192。

### 2. in-batch 负样本与 logQ 校正

候选是千万级，全量 softmax 算不动。做法是一个 batch 里的其他正样本物品当负样本（in-batch softmax）：

$$
P_B(y_i \mid x_i) = \frac{e^{s(x_i, y_i)/\tau}}{\sum_{j \in B} e^{s(x_i, y_j)/\tau}}
$$

**偏差**：batch 里的物品是按热度出现的，热门物品出现在几乎每个 batch 里，被当成负样本的次数比长尾物品多几千倍，模型学到的是「压低热门」，而不是全量 softmax 想学的东西。

**校正**（Yi et al. 式 (3) 后，借自 sampled softmax 的 logQ）：每个 logit 减去该物品被采到的概率的对数：

$$
s^c(x_i, y_j) = s(x_i, y_j) - \log p_j
$$

为什么是减 $\log p_j$：物品 $j$ 作为负样本对分母的期望贡献是 $p_j \cdot e^{s}$，校正后变成 $p_j \cdot e^{s - \log p_j} = e^{s}$，和它出现得多频繁无关，相当于每个物品按 $1/p_j$ 加权，回到「每个物品各算一次」的全量 softmax。

**算一笔**：热门物品 $p = 0.5$，长尾物品 $p = 10^{-4}$，原始 logit 都是 2.0。

1. 热门：$2.0 - \ln 0.5 = 2.0 + 0.69 = 2.69$。
2. 长尾：$2.0 - \ln 10^{-4} = 2.0 + 9.21 = 11.21$。
3. 两者差 8.5，$e^{-8.5} \approx 2\times10^{-4} = 10^{-4} / 0.5$：热门物品每出现一次，在分母里的分量被压到长尾物品的 $1/5000$，正好抵消它多出现的 5000 倍。

**$p_j$ 怎么估**（Yi et al. 算法 2，流式频率估计）：物品在变、词表没法固定，所以用两个大小为 $H$ 的数组和一个哈希函数 $h$：

$$
B[h(y)] \leftarrow (1 - \alpha)\, B[h(y)] + \alpha\, \big(t - A[h(y)]\big),\qquad A[h(y)] \leftarrow t,\qquad \hat p = 1 / B[h(y)]
$$

$A$ 记物品上次出现的训练步，$B$ 是两次出现之间步数的滑动平均。论文的例子：每 50 步出现一次，$\hat p = 1/50 = 0.02$。YouTube 取 $H = 5000$ 万、$\alpha = 0.01$。哈希冲突时几个物品共用一个桶，间隔被低估、$\hat p$ 偏大；论文的算法 3 用多组数组和相互独立的哈希函数，取 $\hat p = 1/\max_i B_i$（思路同 count-min sketch）。

温度：Yi et al. 发现对向量做 L2 归一化再除以温度 $\tau$ 效果更好，线上实验 $\tau = 0.05$；校正后的 softmax 比不校正的在线上指标上多一截（表 3：+0.37% vs +0.20%，作者自测）。

### 3. 为什么要 ANN：暴力检索的账

一个查询和 $N$ 个向量算内积：

$$
\text{FLOPs} = 2 N d,\qquad \text{读的字节} = 4 N d \ \ (\text{fp32})
$$

$N = 10^8$、$d = 128$：$2.56\times10^{10}$ FLOPs，读 51.2 GB。CPU 上一个查询读 51 GB 就要以秒计，所以单个查询必须只看一小部分向量。下面两种索引是两条路：HNSW 靠图跳着走，IVF-PQ 靠聚类只扫几个桶、再把向量压小。

（批量查询在 GPU 上是另一回事，见第 6 节。）

### 4. HNSW：多层近邻图

结构（[Malkov & Yashunin](https://arxiv.org/abs/1603.09320)）：每个点随机分到一个最高层，层号按指数分布 $l = \lfloor -\ln(\text{unif}(0,1)) \cdot m_L \rfloor$，$m_L = 1/\ln M$，所以每往上一层点数约少 $M$ 倍，像跳表。每层是一张近邻图：上层的点少、边长，用来大步跳；第 0 层包含所有点，每个点最多 $M_{\max 0} = 2M$ 个邻居。

**查询**：从最高层的入口点开始，在每层贪心地走向离查询更近的邻居，走不动就下一层；到第 0 层时维护一个大小为 ef 的候选堆做 best-first 搜索，最后取前 $k$。跳数随 $N$ 对数增长。

**参数**：

1. $M$：论文建议 5–48；大 $M$ 适合高维、高召回，小 $M$ 适合低维、低召回。$M_{\max 0} = 2M$ 是实验选出来的，再大会变慢、费内存。
2. efConstruction：建图时的候选堆大小，要大到建图时的近邻召回接近 1（0.95 通常够）。
3. ef（efSearch）：查询时的候选堆大小，越大召回越高、越慢，是线上调召回 / 延迟的主旋钮。

**内存**（[Faiss 选型指南](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) 的估算）：每条向量要存原始向量和第 0 层的 $2M$ 条边（每条 4 字节的 int 编号），上层的边平均很少：

$$
\text{每条向量字节} \approx 4d + 2M \times 4
$$

$d = 128$、$M = 32$：$512 + 256 = 768$ 字节，$10^8$ 条 76.8 GB。边本身的开销论文估计是每个点 60–450 字节（$M$ 取 6–48，不含向量）。

**特点**：召回 / 延迟曲线是 CPU 上最好的一档；但向量存原始精度，内存大；插入可以增量做，删除只能打标记；图遍历每一步都依赖上一步的结果，是访存延迟受限的串行操作。

### 5. IVF-PQ：先分桶，再压缩

**5.1 IVF：只扫几个桶**。先用 k-means 把所有向量分到 $K_{\text{IVF}}$ 个桶（倒排表），查询时只扫离查询最近的 $P$ 个桶。距离计算次数（[Faiss 论文](https://arxiv.org/abs/2401.08281) 式 (18)，假设桶一样大）：

$$
N_{\text{dist}} = K_{\text{IVF}} + P \cdot \frac{N}{K_{\text{IVF}}}
$$

第一项是和所有中心比，第二项是扫 $P$ 个桶里的向量。对 $K_{\text{IVF}}$ 求导，最小值在 $K_{\text{IVF}} = \sqrt{P N}$。

算一笔：$N = 10^8$，$P = 16$：$K_{\text{IVF}} = \sqrt{1.6\times10^9} = 4\times10^4$，$N_{\text{dist}} = 4\times10^4 + 16 \times 2500 = 8\times10^4$，是暴力的 $1/1250$。Faiss 的经验值是 $K_{\text{IVF}}$ 取 $4\sqrt N$ 到 $16\sqrt N$（这里是 $4\times10^4$ 到 $1.6\times10^5$），训练 k-means 要 $30 K_{\text{IVF}}$ 到 $256 K_{\text{IVF}}$ 条样本。实际桶大小不均，热门桶会让扫描量比公式多。

**5.2 PQ：把向量压成 $m$ 个字节**。把 $d$ 维向量切成 $m$ 段，每段 $d/m$ 维，各自用一个 256 个中心的 k-means 量化，存中心的编号（1 字节）：

1. $d = 128$、$m = 16$：每条 16 字节，原始 fp32 是 512 字节，压 32 倍。
2. 加上 8 字节的 id，$10^8$ 条约 2.4 GB，原始向量 51.2 GB。
3. IVF-PQ 里 PQ 编码的是**残差** $x - c$（向量减去所在桶的中心），残差的范围小，同样的码本误差更小。

**5.3 ADC 查表算距离**。查询向量不压缩，对每一段预先算出它和该段 256 个中心的距离，得到一张 $m \times 256$ 的表；候选的距离就是按它的 $m$ 个编号查表再相加：

$$
\hat d(q, x)^2 = \sum_{i=1}^{m} T_i\big[c_i(x)\big],\qquad T_i[j] = \lVert q^{(i)} - C_i[j] \rVert^2
$$

1. 表的大小：$16 \times 256 \times 4$ B = 16 KB，放得进 L1 / GPU 的 shared memory。
2. 建表：$256 \times d \times 2 = 6.6\times10^4$ FLOPs（每个桶的残差不同，扫 $P$ 个桶就要建 $P$ 张，或用 Faiss 的分解技巧预计算一部分）。
3. 每个候选：16 次查表 + 16 次加法，不碰原始向量。

**5.4 重排**。PQ 距离有误差，召回会掉。做法是 PQ 先出 $R \gg k$ 个（比如 1000），再用原始向量（放内存或 SSD）精确重算距离取前 $k$。Faiss 指南里的 `IVF…,PQ…,RFlat` 就是这个结构。下面的手撕里，只用 PQ 的 10-recall@10 是 0.61，PQ 出 100 个再精排是 0.99。

**ScaNN**（[arXiv 1908.10396](https://arxiv.org/abs/1908.10396)）：召回要的是内积大的那些，量化误差在**平行于向量方向**上的分量直接改变内积，垂直分量影响小。ScaNN 用各向异性的量化损失，更重地惩罚平行分量，同样码长下内积排序更准。

### 6. 召回率、内存、延迟的取舍

召回率常用 $n$-recall@$k$：真正的前 $n$ 个近邻里有多少出现在返回的前 $k$ 个里。

| | HNSW | IVF-PQ（+ 重排） |
|---|---|---|
| 每条内存 | $4d + 8M$ ≈ 768 B | $m$ + id ≈ 24 B（+ 重排用的原始向量，可放 SSD） |
| 调召回的旋钮 | efSearch | nprobe、重排数 $R$ |
| 建索引 | 逐点插入，慢 | k-means 训练 + 编码，快 |
| 增量插入 | 支持 | 用已有中心编码后加进桶；分布漂移后要重训中心 |
| GPU 友好 | 差（串行遍历） | 好（顺序扫描 + 查表） |
| 适合 | 内存够、要低延迟高召回 | 十亿级、内存受限 |

工程上常见的组合：IVF 的粗量化器本身用 HNSW 找最近的 $P$ 个中心（$K_{\text{IVF}}$ 很大时），桶内用 PQ，最后重排。

### 7. GPU 检索与向量库

**GPU 擅长什么**（[Faiss 论文](https://arxiv.org/abs/2401.08281) 附录、[Johnson et al.](https://arxiv.org/abs/1702.08734)）：

1. 距离计算：一批查询和一批向量的内积就是矩阵乘，GPU 远强于 CPU。
2. 查表和扫描：PQ 的小表放 shared memory，所有 warp 并行查；IVF 的倒排表是顺序读，可以预取、流水。
3. **难点是 top-$k$**：CPU 上用堆，堆操作是串行的，放到 GPU 上会比所有算术都慢一个数量级。Faiss 专门写了 WarpSelect 一类的 k-selection，跑到理论峰值的 55%，整体比之前的 GPU 最近邻实现快 8.5 倍。
4. **不擅长图遍历**：HNSW 每一步依赖上一步，延迟受限，GPU 的并行度用不上。CAGRA（[arXiv 2308.15136](https://arxiv.org/abs/2308.15136)）重新设计了适合 GPU 的图：大 batch、90–95% 召回下吞吐是 HNSW 的 33–77 倍，单个查询在 95% 召回下快 3.4–53 倍（作者自测）。

**批量暴力检索的账**（本页推算）：$N = 10^7$、$d = 128$、fp16，库占 2.56 GB，一张卡放得下。

1. 每读一个元素（2 字节）要和 $n_q$ 个查询各做一次乘加，算术强度约 $n_q$ FLOPs / 字节。
2. $n_q = 1$：每个查询读 2.56 GB，按 2 TB/s 显存带宽约 1.3 ms，算力几乎全闲着。
3. $n_q = 1024$：$2 \times 1024 \times 10^7 \times 128 = 2.6\times10^{12}$ FLOPs，按有效 100 TFLOP/s（假设）约 26 ms 一批，摊到每个查询 25 µs，而且是精确结果。

所以离线的批量任务（i2i 相似、去重、给新物品找邻居）和能攒 batch 的在线召回适合 GPU 暴力或 GPU IVF；单条、低延迟、内存充裕的在线查询用 CPU HNSW 也很好。Faiss 的选型流程第一问就是「查询条数 $n_q$ 是否小于 $\sqrt N$」。

**向量库要额外解决的**：

1. **分片与合并**：索引按物品切成 $S$ 片，查询发到所有分片，各返回 top-$k$，在中心节点用大小为 $k$ 的堆合并 $S \cdot k$ 个结果，$O(S k \log k)$。各分片精确时合并结果也精确。扇出会放大尾延迟，见 [搜推全景](/recsys/funnel-overview)。
2. **副本**：按 QPS 加副本。
3. **新鲜度**：新物品要尽快可召回。HNSW 增量插入；IVF-PQ 用已有中心编码直接加进桶，定期全量重建（YouTube 是每隔几小时）。模型更新后所有物品向量都变了，要整体换索引，并保证用户塔和物品塔是同一版本。
4. **过滤**：按属性（地区、已下架）过滤。先检索再过滤可能剩不够 $k$ 个，要多取；先过滤再检索会破坏图 / 桶的结构，常见是在扫描时带上过滤条件。

## 面试追问

::: details Q：双塔为什么不能用交叉特征？那排序为什么可以？
双塔的价值在于物品向量能离线算好放进索引，要求物品塔的输入与用户无关。交叉特征（用户和这个作者的互动次数）同时依赖两边，一用就没法预先算，也就没法 ANN。排序只打几百个候选，可以每个候选现算，所以交叉特征都放在排序。
:::

::: details Q：in-batch 负样本为什么要 logQ 校正？校正的方向是什么？
batch 里的物品按热度出现，热门物品当负样本的次数远多于长尾，被过度打压。校正是 $s^c = s - \log p_j$。$p_j < 1$，所以 $-\log p_j$ 是正数，$p_j$ 越小加得越多：$p = 10^{-4}$ 加 9.21，$p = 0.5$ 只加 0.69。效果是热门物品每次出现时在分母里的分量相对缩小 $p_j$ 倍，正好抵消它多出现的次数。
:::

::: details Q：$p_j$ 在流式训练里怎么估？
维护两个哈希数组：$A$ 记上次出现的步数，$B$ 记两次出现间隔的滑动平均，$B \leftarrow (1-\alpha)B + \alpha(t - A)$，$\hat p = 1/B$。每 50 步出现一次就是 0.02。YouTube 用 $H = 5000$ 万个桶、$\alpha = 0.01$，多个哈希函数可以减少冲突带来的高估。
:::

::: details Q：一亿个 128 维向量，HNSW 和 IVF-PQ 各要多少内存？
HNSW 每条 $4d + 8M$，$M = 32$ 时 768 B，共约 77 GB。IVF-PQ 用 $m = 16$ 字节的码加 8 字节 id，共约 2.4 GB；要重排的话原始向量另存，可以放 SSD。
:::

::: details Q：IVF 的 nlist 怎么定？nprobe 呢？
距离计算次数 $K + P N / K$，在 $K = \sqrt{P N}$ 时最小；Faiss 经验取 $4\sqrt N$ 到 $16\sqrt N$。nprobe 是查询时的召回 / 延迟旋钮：翻倍大约让桶内扫描量翻倍，召回提升逐渐变缓，按目标召回率在离线曲线上选。
:::

::: details Q：为什么 HNSW 在 GPU 上不好做？
图遍历每一步要等上一步算出下一个点，是串行的、受访存延迟限制，GPU 的上万个线程用不上。GPU 擅长的是批量的距离计算（矩阵乘）和顺序扫描，所以 GPU 上用 IVF 或专门的 GPU 图（CAGRA）。
:::

::: details Q：召回模型更新了，索引怎么办？
物品塔一变，所有物品向量都要重算、索引整体重建，并且线上用户塔必须和索引用同一版模型，否则两个向量空间对不上。常见做法是新索引建好后和新用户塔一起原子切换；新物品在两次重建之间增量插入。
:::

## 手撕

IVF-PQ：粗聚类分桶、残差 PQ 编码、ADC 查表、可选的原始向量重排，和暴力检索比召回：

```python
import numpy as np

rng = np.random.default_rng(0)
N, d, nq, k = 20000, 32, 100, 10
nlist, nprobe, M, ksub = 64, 8, 8, 256        # IVF 桶数、查几个桶、PQ 子空间数、每个子空间码字数
dsub = d // M

def kmeans(x, c, iters=10):
    cent = x[rng.choice(len(x), c, replace=False)]
    for _ in range(iters):
        a = ((x[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for j in range(c):
            if (a == j).any():
                cent[j] = x[a == j].mean(0)
    return cent

xb = rng.normal(size=(N, d)).astype(np.float32) + rng.normal(size=(50, d))[rng.integers(0, 50, N)] * 3
xq = xb[rng.choice(N, nq)] + 0.1 * rng.normal(size=(nq, d))

# 1. 训练：粗聚类（IVF）+ 每个子空间一套 PQ 码本，PQ 编码的是残差 x - centroid
coarse = kmeans(xb, nlist)
assign = ((xb[:, None] - coarse[None]) ** 2).sum(-1).argmin(1)
resid = xb - coarse[assign]
books = np.stack([kmeans(resid[:, m*dsub:(m+1)*dsub], ksub, 5) for m in range(M)])  # [M, ksub, dsub]

# 2. 编码：每条向量 M 个字节
codes = np.stack([((resid[:, None, m*dsub:(m+1)*dsub] - books[m][None]) ** 2).sum(-1).argmin(1)
                  for m in range(M)], 1).astype(np.uint8)          # [N, M]
lists = [np.where(assign == c)[0] for c in range(nlist)]

# 3. 查询：选 nprobe 个最近的桶；每个桶建一张 M x ksub 的距离表，候选距离 = M 次查表求和
def search(q, rerank=0):
    probe = ((coarse - q) ** 2).sum(1).argsort()[:nprobe]
    ids, dist = [], []
    for c in probe:
        r = q - coarse[c]                                        # 查询相对这个桶中心的残差
        table = ((r.reshape(M, 1, dsub) - books) ** 2).sum(-1)  # [M, ksub]，ADC 查表
        ids.append(lists[c])
        dist.append(table[np.arange(M), codes[lists[c]]].sum(1))
    ids, dist = np.concatenate(ids), np.concatenate(dist)
    if not rerank:
        return ids[dist.argsort()[:k]]
    top = ids[dist.argsort()[:rerank]]                           # PQ 粗排出 R 个
    return top[((xb[top] - q) ** 2).sum(1).argsort()[:k]]        # 用原始向量精排

gt = ((xq[:, None] - xb[None]) ** 2).sum(-1).argsort(1)[:, :k]   # 暴力检索的真值
recall = lambda R: np.mean([len(set(search(q, R)) & set(g)) / k for q, g in zip(xq, gt)])
scanned = np.mean([sum(len(lists[c]) for c in ((coarse - q) ** 2).sum(1).argsort()[:nprobe]) for q in xq])
print(f"scanned {scanned:.0f} / {N} codes, {M} B/vector vs {4*d} B raw")
print(f"10-recall@10: PQ only {recall(0):.2f}, PQ top-100 + exact rerank {recall(100):.2f}")
# scanned 3794 / 20000 codes, 8 B/vector vs 128 B raw
# 10-recall@10: PQ only 0.61, PQ top-100 + exact rerank 0.99
```

读输出：扫了 19% 的码，每条只存 8 字节（原始 128 字节）；只用 PQ 距离召回只有 0.61，多取 100 个再用原始向量精排就到 0.99。重排的代价是要能随机访问原始向量。

in-batch softmax 加 logQ 校正，PyTorch 几行：

```python
import torch
import torch.nn.functional as F

def in_batch_loss(u, v, log_p, tau=0.05):
    # u, v: [B, d] 用户 / 正样本物品向量；log_p: [B] 每个物品被采到的概率的对数
    u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1)
    logits = u @ v.T / tau - log_p[None, :]      # 第 j 列是物品 j，整列减 log p_j
    return F.cross_entropy(logits, torch.arange(len(u)))
```

## 参考

- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)（Yi et al., RecSys 2019）：in-batch softmax、logQ 校正、算法 2 流式频率估计、YouTube 的塔结构 / batch / $H$ / $\alpha$ / 温度、索引每几小时重建
- [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)：召回用 ANN、负采样几千个
- [Scaling the Instagram Explore recommendations system](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)：双塔不能用交叉特征、物品向量每天离线生成
- [HNSW](https://arxiv.org/abs/1603.09320)：分层结构、$m_L = 1/\ln M$、$M_{\max 0} = 2M$、$M$ 取 5–48、每点 60–450 字节
- [Faiss: Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)：HNSW 内存公式、nlist 经验值、训练样本数、RFlat 重排
- [The Faiss library](https://arxiv.org/abs/2401.08281)：式 (18) 的 IVF 距离计算次数、ADC、GPU 实现的取舍
- [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)：GPU k-selection 55% 峰值、快 8.5 倍
- [ScaNN: Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396)：内积检索下各向异性的量化损失
- [CAGRA](https://arxiv.org/abs/2308.15136)：GPU 图索引相对 HNSW 的吞吐
