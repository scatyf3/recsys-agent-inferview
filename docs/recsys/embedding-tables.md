---
title: Embedding 表的账：参数量、哈希冲突、动态 embedding、准入与淘汰、压缩
status: draft
tags: [embedding, hashing, dynamic-embedding, admission, eviction, compression]
difficulty: 3
order: 2
related: []
recsys: [rs-e-table, rs-e-dynamic, rs-e-compress]
---

# Embedding 表的账：参数量、哈希冲突、动态 embedding、准入与淘汰、压缩

> 搜推模型的参数几乎全在 embedding 表里：先算清楚表有多大，再看 ID 怎么映射到行、哪些 ID 值得占一行、什么时候把它踢掉、每行能压到多小

## 一句话结论

参数量 = $\sum_t n_t d_t$，训练时每个参数还要带优化器状态。MLPerf DLRMv2 的 26 张表共 2.04 亿行、$d = 128$，就是 261 亿参数，fp32 104.5 GB；参考实现用逐元素 Adagrad，再加一份同样大的状态，共 209 GB。dense 部分只有约 1600 万参数，占 0.06%。Meta 的 12T 参数模型照这样算要 96 TB，换成 rowwise Adagrad 再把表存成 fp16，降到 24 TB。ID 空间远大于存得下的行数（Monolith 的一个线上模型原始 ID 空间是 $2^{48}$），所以要么哈希到固定行数，代价是冲突：$N$ 个 ID 放进 $m$ 行，冲突比例约 $N/(2m)$；要么用动态哈希表，一个 ID 一行，再配准入（出现够 $k$ 次才建行，或按概率准入）和淘汰（过期时间、LRU/LFU）控制总行数。压缩有四条路：低精度存储、rowwise 优化器、按频率给不同维度、商余（QR）分解。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $T$ | embedding 表（稀疏特征）的个数 | 26（MLPerf DLRMv2） |
| $n_t$ | 第 $t$ 张表的行数 | 最大的 4000 万 |
| $d$ | embedding 维度（每行几个数） | 128 |
| $s$ | 每个数的字节数 | fp32 4、fp16 2、int8 1 |
| $o$ | 优化器状态，每个参数折合几个字节 | Adagrad 4，Adam 8，rowwise Adagrad $4/d$ |
| $N$ | 实际出现的不同 ID 个数 | $10^8$ |
| $m$ | 哈希表的行数（桶数） | $10^8$ |
| $k$ | 准入阈值：出现几次才建行 | 2 |
| $p$ | 概率准入：每次见到一个新 ID 时建行的概率 | 0.1 |
| $\lambda$ | 每天新准入的 ID 数 | $5\times10^7$ |

单位：1 B（个数）= $10^9$，GB = $10^9$ 字节，TB = $10^{12}$ 字节。

### 1. 参数量与训练时的字节数

$$
P_{\text{emb}} = \sum_{t=1}^{T} n_t d_t, \qquad
\text{训练字节} = P_{\text{emb}} \cdot (s + o)
$$

**MLPerf DLRMv2**（参考实现 [torchrec_dlrm](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm) 的 `--num_embeddings_per_feature`、`--embedding_dim 128`）：

1. 26 张表的行数加起来是 204,184,588。其中 5 张各 4000 万行，1 张 307 万行，剩下 20 张合计约 112 万行。
2. 参数：$2.042\times10^8 \times 128 = 2.61\times10^{10}$，即 261 亿。
3. fp32 权重：$2.61\times10^{10} \times 4 = 104.5$ GB。
4. 参考实现开 `--adagrad` 时，embedding 走 FBGEMM 的 `EXACT_ADAGRAD`，每个元素一个 fp32 累加器，再加 104.5 GB，共 **209 GB**。
5. dense 部分（底层 MLP 13→512→256→128、3 层低秩 DCN、顶层 MLP 3456→1024→1024→512→256→1）按层算下来约 1600 万参数（本页按配置推算），占总参数的 0.06%。

**Meta model-F**（[Neo / ZionEX，arXiv 2104.05158](https://arxiv.org/abs/2104.05158) §8.5）：12T 参数，几张大表各约 100 亿行、256 列。朴素做法 fp32 权重加一份同样大的优化器状态，$12\times10^{12} \times 4 \times 2 = 96$ TB。先换 rowwise AdaGrad（状态从每元素一个变成每行一个），再把表存成 fp16，降到 24 TB，刚好放进原型集群的 4 TB HBM + 24 TB DRAM。

**为什么 embedding 常用 Adagrad，而且是 rowwise 的**：

1. 一行只有它的 ID 出现时才被更新。Adagrad 给每个坐标自己的学习率 $\eta / \sqrt{\sum g^2}$：出现少的行累加得少，步子大；热门行步子小。Google 广告的 FTRL 实践里，每坐标学习率比全局学习率的 AucLoss 低 11.2%（[McMahan et al. KDD 2013](https://research.google.com/pubs/archive/41159.pdf) §3.1，作者自测）。
2. 逐元素 Adagrad 的状态和权重一样大（上面的 209 GB）。rowwise 版每行只存一个标量（这一行梯度平方的均值的累加），状态从 $n \cdot d$ 个变成 $n$ 个：DLRMv2 只多 $2.04\times10^8 \times 4 = 0.82$ GB。
3. Adam 每参数要 8 字节状态，而且它的动量假设每步都有梯度，对隔很多步才出现一次的行不太合适；TBE 里 Adam 也有 rowwise 变体（`PARTIAL_ROWWISE_ADAM`）。

### 2. ID 怎么变成行号：哈希与冲突

**固定大小 + 哈希**：行号 $= h(\text{id}) \bmod m$。表大小固定，新 ID 来了不用扩容，代价是不同 ID 可能落在同一行、共用一个向量。

**冲突比例**。$N$ 个不同 ID 均匀落进 $m$ 个桶。某个桶一个都没落的概率是 $(1 - 1/m)^N \approx e^{-N/m}$，所以被占用的桶数约 $m(1 - e^{-N/m})$。有 $N$ 个 ID 却只占了这么多行，差出来的就是「被并到别人行里」的 ID：

$$
c = 1 - \frac{m}{N}\left(1 - e^{-N/m}\right) \approx \frac{N}{2m} \quad (N \ll m)
$$

| $N/m$ | $c$ |
|---|---|
| 0.1 | $1 - 10(1 - e^{-0.1}) = 4.8\%$ |
| 0.5 | $1 - 2(1 - e^{-0.5}) = 21.3\%$ |
| 1 | $e^{-1} = 36.8\%$ |
| 2 | $1 - 0.5(1 - e^{-2}) = 56.8\%$ |

想把冲突压到 5% 以下，行数要是 ID 数的 10 倍，大部分行空着。

**冲突为什么伤效果**。ID 的频率是长尾的：热门 ID 出现上百万次，冷门的不到十次。两个中高频 ID 共用一行，这一行的梯度是两者的混合，哪个都学不准；哈希函数不看频率，没法保证热门 ID 不撞。[Monolith](https://arxiv.org/abs/2209.07663) 在 MovieLens-25M 上把 ID 先 MD5 再映射到更小的空间：用户 ID 从 162,541 个变成 149,970 个（冲突 7.73%），电影 ID 冲突 2.86%；无冲突的版本从第一个 epoch 起 AUC 就更高，最后收敛得也更高（图 7，作者自测）。

**商余（QR）分解**：用两张小表给每个 ID 拼出一个唯一的组合（[Shi et al. arXiv 1909.02107](https://arxiv.org/abs/1909.02107)）：

$$
e(x) = E_q\left[\lfloor x / K \rfloor\right] \odot E_r\left[x \bmod K\right]
$$

$K$ 是余数表的行数。每个 $x$ 对应唯一的 (商, 余) 对，所以向量组合不重复；参数量 $(\lceil N/K \rceil + K) \cdot d$，取 $K = \sqrt{N}$ 时最小，为 $2\sqrt{N}\, d$。Monolith 线上模型的基线就是这么做的（组合用加法）：原始 ID 空间 $2^{48}$，$K = 2^{24}$，两张各 $2^{24}$ 行，合计 $2^{25}$ 行。代价是每一行被 $2^{24}$ 个 ID 共用，表达能力有限；Monolith 换成无冲突哈希表后线上 AUC 持续更高（图 8，作者自测）。

### 3. 动态 embedding：一个 ID 一行

不预先定行数，而是一张 key → value 的哈希表，key 是 64 位 ID，value 是这一行的向量、优化器状态和元数据（出现次数、最后出现时间）。新 ID 来了插入，过期的删除，表随数据增长。

1. **Monolith**：在 TensorFlow 里实现成 resource op 的 HashTable，底下是 cuckoo hashmap：两张表 $T_0, T_1$、两个哈希函数 $h_0, h_1$，一个 key 只可能在 $T_0[h_0(k)]$ 或 $T_1[h_1(k)]$，所以查找和删除最坏 $O(1)$；插入时位置被占就把原来的 key 踢到它的另一个位置，期望均摊 $O(1)$，成环就 rehash。
2. **GPU 上**：NVIDIA 的 [HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV) 把 key-value 放在 HBM 和主机内存里，桶满了才按分数淘汰，策略有 LRU、LFU、EpochLRU、EpochLFU 和自定义。TorchRec 也有 managed collision（ManagedCollisionCollection）模块，把原始 ID 重映射到不冲突的行号。

**每个 ID 占多少**（假设）：key 8 B，$d = 64$ 的 fp32 向量 256 B，rowwise Adagrad 4 B，次数和时间戳 8 B，共 276 B；哈希表装载因子 0.5（一半的槽空着，HierarchicalKV 的 benchmark 就按 0.5 测），折合每个 ID 552 B。$10^9$ 个 ID 就是 552 GB。ID 只进不出的话，表会一直涨，所以要有下面两道闸。

### 4. 准入：出现够多才给一行

长尾到什么程度：Google 的一些广告模型里，一半的不同特征在几十亿条样本里只出现过一次（McMahan et al. §4.1）。这些 ID 的向量几乎学不到东西，却和热门 ID 占同样的内存。准入就是「先别建行，看它还会不会再来」：

1. **计数阈值**：出现 $k$ 次才建行（Monolith 的阈值按模型调）。没准入的 ID 也要计数，用计数 Bloom filter 或 count-min sketch 存：内存固定，可能多数（误判成已到 $k$ 次），不会少数。
2. **概率准入**（Poisson inclusion）：每次见到一个还没有行的 ID，以概率 $p$ 建行。不用存任何状态；被准入前要见到的次数服从几何分布，期望 $1/p$。

一个出现 $c$ 次的 ID 在概率准入下被收进来的概率：

$$
P(\text{准入}) = 1 - (1 - p)^c
$$

$p = 0.1$：$c = 1$ 时 10%，$c = 10$ 时 $1 - 0.9^{10} = 65.1\%$，$c = 50$ 时 $1 - 0.9^{50} = 99.5\%$。只出现一两次的 ID 大多进不来，常见 ID 迟早进来。

McMahan et al. 表 2（作者自测）：

| 方法 | 省下的内存 | AucLoss 变差 |
|---|---|---|
| Bloom，$k = 2$ | 66% | 0.008% |
| Bloom，$k = 1$ | 55% | 0.003% |
| Poisson，$p = 0.03$ | 60% | 0.020% |
| Poisson，$p = 0.1$ | 40% | 0.006% |

Bloom 计数的取舍更好：同样省 60% 上下，效果损失更小。代价是一个热门 ID 刚出现的前几次没有自己的向量，只能用默认值（全 0 或一个共享的「未知」向量），新物料冷启动会慢一点。

### 5. 淘汰：长期不出现就删

已经准入的 ID 也会过时：用户不活跃了、视频下架了。Monolith 给每张表设过期时间，ID 超过这段时间没出现就删掉，时间按表调（对历史敏感的特征设长一些）。

**稳态有多少行**：用 Little 定律，系统里的平均数量 = 到达速率 × 平均停留时间：

$$
n_{\text{稳态}} \approx \lambda \cdot \text{TTL}
$$

假设每天新准入 $\lambda = 5\times10^7$ 个 ID，TTL 7 天：$3.5\times10^8$ 行，按上面每行 276 B（不算装载因子）约 97 GB。TTL 翻倍，表也翻倍。这里的停留时间是「最后一次出现之后再过 TTL」，热门 ID 一直在刷新时间戳，实际停留远长于 TTL，所以真实的表会比这个估计大，要按线上统计校准。

按容量淘汰时用 LRU 或 LFU（HierarchicalKV、FBGEMM 的 HBM cache 都提供这两种）。训练侧删了的 ID，同步到线上时也要删，否则服务侧的表只涨不缩，见 [在线学习与模型更新](/recsys/online-learning)。

### 6. 压缩：每行能多小

**6.1 低精度存储**。

1. 训练：fp32 → fp16 直接减半。难点是更新：一次更新量可能比 fp16 在这个数附近的间隔还小，直接舍入会丢掉。McMahan et al. 用随机舍入解决：存 16 位定点（q2.13：1 位符号、2 位整数、13 位小数），舍入时 $w_{\text{rounded}} = 2^{-13}\lfloor 2^{13} w + R \rfloor$，$R$ 在 $[0, 1)$ 上均匀，舍入误差均值为 0，小更新按概率累积下来。相比 64 位浮点省 75% 内存，没有可测的效果损失（作者自测）。
2. Meta model-A 把表换成 fp16 后吞吐本身没涨，但腾出的 HBM 让分片可以排得更均衡，最后吞吐涨了 20%（Neo §8.4，作者自测）。
3. 推理：按行量化成 int8 或 int4，每行另存一个 scale 和一个 bias 来反量化。$d = 128$：fp32 一行 512 B；int8 是 128 B 加 scale/bias（按两个 fp32 算 8 B）共 136 B，约 3.8 倍压缩。

**6.2 优化器状态**：rowwise Adagrad，见第 1 节。

**6.3 混合维度**（[Ginart et al. arXiv 1909.11810](https://arxiv.org/abs/1909.11810)）：按频率分组，热门 ID 用大维度，冷门 ID 用小维度 $d_b$，再乘一个 $d \times d_b$ 的投影矩阵变回统一的 $d$ 维。Criteo Kaggle 上参数减半时准确率高 0.1%，参数少 16 倍时准确率持平（作者自测）。

**6.4 QR / 哈希**：第 2 节，用冲突换内存，适合 ID 空间远大于预算、又不想维护动态表的场景。

## 面试追问

::: details Q：DLRMv2 的 embedding 训练时要多少显存？怎么算？
行数加起来 2.04 亿，乘 $d = 128$ 是 261 亿参数，fp32 104.5 GB。参考实现用逐元素 Adagrad，状态同样大，共 209 GB。换 rowwise Adagrad，状态只有 2.04 亿 × 4 B = 0.82 GB，合计约 105 GB。dense 部分约 1600 万参数，可以忽略。
:::

::: details Q：哈希冲突一定有害吗？什么时候可以用哈希？
冲突比例约 $N/(2m)$。冲突落在两个冷门 ID 上影响很小（它们本来就学不好）；落在两个中高频 ID 上就会互相干扰。ID 空间很大、预算紧、ID 分布稳定时，用哈希或 QR 很省事；新 ID 不断涌入、长尾又重要的场景（短视频、广告）更适合动态表。Monolith 在 MovieLens 和线上模型上都看到无冲突版本 AUC 更高（作者自测）。
:::

::: details Q：为什么 embedding 用 Adagrad，不用 Adam？
embedding 的一行只在 ID 出现时更新，Adagrad 的每坐标学习率天然让冷门行步子大、热门行步子小。rowwise 版每行只存一个累加器，状态几乎不占内存；Adam 每参数要 8 字节，而且动量假设每步都有梯度。dense 部分仍常用 Adam。
:::

::: details Q：准入阈值设高了有什么代价？
新 ID 在攒够次数之前没有自己的向量，只能用默认向量，新物料和新用户的冷启动变慢。常见的补救是让新 ID 先用它的属性特征（类目、作者等）的 embedding，或者给新物料单独设更低的阈值。
:::

::: details Q：动态表在多台机器上怎么放？
按 key 的哈希分到各个 shard（PS 节点或 GPU），每个 shard 内部是自己的哈希表。查表时先按 shard 把 ID 分桶、发给对应的机器，见 [Parameter Server](/recsys/parameter-server) 和 [Embedding 分片](/recsys/embedding-sharding)。
:::

::: details Q：被淘汰的 ID 又出现了怎么办？
当新 ID 处理：重新走准入，向量从初始值开始学。所以 TTL 不能短于这类 ID 的典型「回访间隔」，否则一直在丢掉学到的东西再重学。
:::

## 手撕

固定大小的哈希 embedding 和 QR embedding，对比参数量和冲突：

```python
import torch, torch.nn as nn

class HashEmb(nn.Module):
    def __init__(self, m, d):
        super().__init__()
        self.m, self.emb = m, nn.Embedding(m, d)
    def forward(self, ids):                      # ids: int64 原始 ID
        return self.emb(ids % self.m)            # 真实系统里先过一个哈希函数再取模

class QREmb(nn.Module):
    def __init__(self, n, K, d):
        super().__init__()
        self.K = K
        self.q = nn.Embedding((n + K - 1) // K, d)   # 商表
        self.r = nn.Embedding(K, d)                  # 余数表
    def forward(self, ids):
        return self.q(ids // self.K) * self.r(ids % self.K)   # 每个 ID 的 (商, 余) 唯一

torch.manual_seed(0)
n, d = 2**40, 16                                   # ID 空间 2^40
ids = torch.randint(0, n, (100_000,)).unique()     # N 个不同 ID（随机 ID 取模相当于均匀哈希）
N = len(ids)
for ratio in [10, 1]:
    used = (ids % (N * ratio)).unique().numel()    # 被占用的行数
    print(f"m = {ratio} x N: 冲突比例 {1 - used / N:.1%}")
K = 2**20                                          # K = sqrt(n)
qr = QREmb(n, K, d)
print("QR 参数:", sum(p.numel() for p in qr.parameters()), " 全表:", n * d)
# m = 10 x N: 冲突比例 4.8%
# m = 1 x N: 冲突比例 36.9%
# QR 参数: 33554432  全表: 17592186044416
```

冲突比例和第 2 节的公式（4.8%、36.8%）对得上。QR 用 $2\sqrt{n}\,d = 2^{21} \times 16$ 个参数，给 $2^{40}$ 个 ID 各拼出一个不重复的组合；全表要 17.6 万亿个参数。

## 参考

- [MLPerf DLRMv2 参考实现（torchrec_dlrm）](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm)：26 张表的行数、multi-hot 大小、$d = 128$、Adagrad 配置
- [Software-Hardware Co-design for Fast and Scalable Training of DLRMs（Neo / ZionEX，arXiv 2104.05158）](https://arxiv.org/abs/2104.05158)：model-F 的 96 TB → 24 TB、rowwise AdaGrad、fp16 表
- [Monolith（arXiv 2209.07663）](https://arxiv.org/abs/2209.07663)：cuckoo hashmap、准入与过期、MovieLens 冲突实验、$2^{48} \to 2^{25}$ 的 QR 基线
- [Ad Click Prediction: a View from the Trenches（McMahan et al. KDD 2013）](https://research.google.com/pubs/archive/41159.pdf)：每坐标学习率、Poisson / Bloom 准入、q2.13 随机舍入
- [Compositional Embeddings Using Complementary Partitions（arXiv 1909.02107）](https://arxiv.org/abs/1909.02107)：商余分解
- [Mixed Dimension Embeddings（arXiv 1909.11810）](https://arxiv.org/abs/1909.11810)：按频率分配维度
- [HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV)：GPU 上的 key-value embedding 存储与淘汰策略
- [FBGEMM TBE 训练算子文档](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/tbe_ops_training.html)：优化器类型（`EXACT_ROWWISE_ADAGRAD`、`PARTIAL_ROWWISE_ADAM` 等）
