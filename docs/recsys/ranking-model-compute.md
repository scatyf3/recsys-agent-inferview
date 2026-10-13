---
title: 排序模型的算力账：DLRM / DCN / 特征交叉，为什么 MFU 低，dense 部分怎么提速
status: draft
tags: [dlrm, dcn, feature-interaction, mfu, flops]
difficulty: 3
order: 6
related: []
recsys: [rs-k-interact]
---

# 排序模型的算力账：DLRM / DCN / 特征交叉，为什么 MFU 低，dense 部分怎么提速

> 排序模型参数几百亿，每条样本的计算却只有几十 MFLOPs；GPU 的时间大半花在搬 embedding 和等 all-to-all 上，dense 部分的提速靠大 batch、低精度、融合和重叠

## 一句话结论

DLRM 的 dense 部分是「底层 MLP 处理连续特征 → 和各个 embedding 做特征交叉 → 顶层 MLP 出分数」。MLPerf DLRMv2（DCN-v2 交叉）每条样本前向约 32 MFLOPs，训练约 96 MFLOPs，而参数有 261 亿：每个参数每条样本只摊到 0.004 FLOP，LLM 训练是每参数每 token 6 FLOP。交叉层里 DLRM 的两两点积几乎不花算力（0.09 MFLOPs），DCN-v2 的满秩交叉矩阵每层 $2D^2$（$D = 3456$ 时 24 MFLOPs），低秩到 $r = 512$ 降到 $4Dr = 7.1$ MFLOPs。Meta 的 model-A 在 128 张 A100 上跑 1.2 M 样本/秒，按每条样本 638 MFLOPs 前向、训练算 3 倍反推，每卡约 18 TFLOP/s，是 A100 TF32 峰值的 11%（本页推算），比 LLM 训练低几倍。原因是查表只搬字节、all-to-all 在关键路径上、上千张表带来的负载不均和小 kernel。dense 部分的提速手段：加大 batch、TF32/bf16 换 fp16 存表和通信、TBE 和交叉层融合、计算通信重叠、低秩交叉。

## 推导

### 0. 记号

| 符号 | 含义 | 例子（MLPerf DLRMv2） |
|---|---|---|
| $T$ | 稀疏特征（embedding 表）数 | 26 |
| $d$ | embedding 维度，也是底层 MLP 的输出维度 | 128 |
| $D$ | 交叉层的输入维度，$(T+1)d$ | 3456 |
| $r$ | DCN-v2 低秩分解的秩 | 512 |
| $F$ | 每条样本前向 FLOPs | 约 32 M |
| $B_l$ | 每卡 batch | 8192 |
| $\text{QPS}$ | 训练吞吐，样本/秒 | |
| $\Pi$ | 单卡峰值算力 | A100：TF32 156、bf16 312 TFLOP/s（稠密） |

1 M = $10^6$，1 T = $10^{12}$。一次乘加按 2 FLOPs 算；训练 ≈ 3 × 前向（反向对输入、对权重各一次同样大的矩阵乘，见 [训练算力账](https://scatyf3.github.io/ai-infra-inferview/train/compute-mfu)）。

### 1. DLRM 的结构

[DLRM（arXiv 1906.00091）](https://arxiv.org/abs/1906.00091)：

1. **底层 MLP**：连续特征（年龄、价格、历史 CTR 等，DLRMv2 是 13 个）过 MLP 变成一个 $d$ 维向量。
2. **embedding**：$T$ 个稀疏特征各查一张表、pooling，得到 $T$ 个 $d$ 维向量（[Embedding 查表](/recsys/embedding-lookup)）。
3. **特征交叉**：这 $T + 1$ 个向量两两做点积，得到 $(T+1)T/2$ 个标量，再和底层 MLP 的输出拼起来。
4. **顶层 MLP**：输出一个 logit，过 sigmoid 得点击率。

**为什么要显式交叉**：「这个用户 × 这个类目」这类二阶组合对点击率影响很大，MLP 理论上能学到乘积，但效率低；直接把向量两两相乘喂进去，等于把 FM（因子分解机）的二阶项做成了网络的一层。

**点积交叉的算力**：$T = 26$，$T + 1 = 27$ 个向量，$27 \times 26 / 2 = 351$ 对，每对 $d$ 次乘加：

$$
351 \times 2 \times 128 = 89{,}856 \approx 0.09\ \text{MFLOPs}
$$

顶层 MLP 的输入是 $128 + 351 = 479$ 维。交叉本身便宜，实现上是一个 batched matmul（每条样本 $27 \times 128$ 乘 $128 \times 27$），矩阵太小，GPU 利用率很低；再取上三角、拼接，都是搬数据的小操作。

### 2. DCN-v2：更强的交叉，更多的算力

[DCN-v2（arXiv 2008.13535）](https://arxiv.org/abs/2008.13535) 把所有向量拼成 $x_0 \in \mathbb{R}^D$，每层：

$$
x_{l+1} = x_0 \odot \left(W_l x_l + b_l\right) + x_l
$$

$\odot$ 是逐元素乘。每过一层交叉阶数加一：第一层得到 $x_0$ 各分量的两两乘积，第二层三阶，以此类推；$+ x_l$ 是残差。

| | 参数 / 层 | FLOPs / 层 / 样本 | $D = 3456$ |
|---|---|---|---|
| 满秩 $W_l \in \mathbb{R}^{D \times D}$ | $D^2$ | $2D^2$ | 1194 万参数，23.9 MFLOPs |
| 低秩 $W_l = U_l V_l^\top$，$U_l, V_l \in \mathbb{R}^{D \times r}$ | $2Dr$ | $4Dr$ | $r = 512$：354 万参数，7.1 MFLOPs |

低秩省 3.4 倍。DCN-v2 论文观察到学出来的满秩矩阵本身就是低秩的，所以拆开几乎不损失效果（作者自测）。

### 3. 一条样本多少 FLOPs

DLRMv2 的配置（[MLPerf 参考实现](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm)：底层 MLP 13→512→256→128，3 层 $r = 512$ 的低秩 DCN，顶层 MLP 3456→1024→1024→512→256→1）：

| 部分 | 前向 FLOPs / 样本 |
|---|---|
| 底层 MLP | $2(13 \cdot 512 + 512 \cdot 256 + 256 \cdot 128) = 0.34$ M |
| DCN 3 层 | $3 \times 4 \times 3456 \times 512 = 21.2$ M |
| 顶层 MLP | $2(3456 \cdot 1024 + 1024^2 + 1024 \cdot 512 + 512 \cdot 256 + 256) = 10.5$ M |
| **合计** | **32.1 M**，训练 ×3 ≈ 96 M |

（embedding pooling 的加法和逐元素操作不计。）

**和 LLM 比**：

| | 参数 | 训练 FLOPs / 样本（token） | FLOPs / 参数 |
|---|---|---|---|
| DLRMv2 | 261 亿（99.94% 在 embedding） | 96 M | 0.004 |
| 7B dense LLM | 70 亿 | $6 \times 7\times10^9 = 42$ G | 6 |

LLM 每个参数在每个 token 上都参与矩阵乘；DLRM 每条样本只碰到 214 行 embedding 和 1600 万 dense 参数。所以搜推的瓶颈是存储和通信，不是算力。

**一步要多久**（理想情况）：每卡 $B_l = 8192$，$96\text{ M} \times 8192 = 786$ GFLOPs，A100 bf16 312 TFLOP/s 上 2.5 ms。同一步每卡还要读约 0.9 GB 的 embedding（前向，反向更多）、做两次各约 95 MB 的 all-to-all（[Embedding 分片](/recsys/embedding-sharding)）。单节点时这几样在同一个量级；多节点时 all-to-all 走网卡，很快就压过计算。

### 4. MFU：用 Meta 的数字反推

MFU = 实际做的模型 FLOPs / 峰值，分子只算模型本身的计算（查表的搬运不算）。[Neo（arXiv 2104.05158）](https://arxiv.org/abs/2104.05158) 表 3 的 model-A：7930 亿参数、约 1000 张表、每条样本 638 MFLOPs、每卡 batch 512，在 16 个节点 128 张 A100 上 1.2 M 样本/秒（同一篇论文 §8.4 说各项优化叠加后最高到 1.5 M，这里按表 3 的 1.2 M 算）。

论文没写 638 MFLOPs 是前向还是整个训练步，两种都算（本页推算）：

1. 按前向、训练 ×3：$1.2\times10^6 \times 638\times10^6 \times 3 = 2.30\times10^{15}$ FLOP/s，每卡 $2.30\times10^{15} / 128 = 18$ TFLOP/s。A100 TF32 稠密峰值 156：**11.5%**；bf16 312：5.7%。
2. 按整个训练步：每卡 6.0 TFLOP/s，TF32 峰值的 3.8%。

对照：Llama 3 405B 在 H100 上 bf16 MFU 38–43%（作者自测，见 [训练算力账](https://scatyf3.github.io/ai-infra-inferview/train/compute-mfu)）。DLRM 低一个量级上下。

### 5. 为什么 MFU 低

1. **查表只搬字节**：fp32 算术强度 0.25 FLOP/B，时间算在步长里，FLOPs 不进分子（[Embedding 查表](/recsys/embedding-lookup)）。
2. **all-to-all 在关键路径上**：dense 前向的输入要等 output all-to-all 回来；跨节点后通信时间不随卡数减少（[Embedding 分片](/recsys/embedding-sharding)）。
3. **负载不均**：上千张表的大小、pooling 长度、维度（model-A 是 4 到 384）差很多，最慢的卡决定步长。model-A 的基线不到 700 K 样本/秒，就是各卡查表时间差太大（Neo §8.4）。
4. **小 kernel 多**：每张表、每个特征的预处理、交叉层的拼接和切片，都是很短的 kernel，launch 开销和尾部效应占比高。
5. **矩阵窄**：每卡 batch 512 时，MLP 的矩阵乘是 $512 \times$ 几千，交叉层的 bmm 每个只有 $27 \times 27$，填不满 tensor core。
6. **数据和 CPU**：样本读取、解析、哈希、拷贝到 GPU，喂不饱就空转。

### 6. dense 部分怎么提速

| 手段 | 做法 | 效果（Neo model-A，作者自测） |
|---|---|---|
| 均衡切表 | table / column / data 并行混合 | +20% |
| fp16 存表 | 表大小减半，腾出显存再均衡 | 再 +20% |
| 量化通信 | all-to-all 前向 fp16、反向 bf16 | 近 30% 加速，质量无损 |
| 加大 batch | 全局 64K → 256K，配合调好的超参 | 每卡矩阵变宽、通信更饱和 |
| 合计 | | 比 TF32、64K batch 的基线 +87% |

另外几条通用的：

1. **算子融合**：TBE 把所有表合成一个 kernel；交叉层的 bmm、取上三角、拼接可以写成一个 kernel；整步用 CUDA Graph 录下来，消掉 launch 开销。
2. **低精度算 dense**：TF32 / bf16，MLP 的矩阵乘本身能吃满 tensor core。
3. **流水**：数据拷贝、input all-to-all 和上一个 batch 的计算重叠（TorchRec `TrainPipelineSparseDist`）。
4. **结构上省**：DCN 用低秩，交叉只在必要的特征对上做。
5. **把算力花在值得的地方**：既然 GPU 闲着，可以把模型的 dense 部分做大（更深的交叉、序列建模），每条样本多花算力换效果，MFU 反而上去。这是 [生成式推荐](/recsys/generative-rec) 和长序列模型的出发点。

## 面试追问

::: details Q：DLRM 的特征交叉为什么用点积？算力多少？
点积就是两个特征向量的二阶交互，相当于把 FM 的二阶项做成网络的一层；MLP 学乘积效率低。$T = 26$ 时 27 个向量 351 对，每对 128 次乘加，约 0.09 MFLOPs，算力可以忽略，问题是矩阵太小、利用率低。
:::

::: details Q：DCN-v2 的交叉层每层多少 FLOPs？为什么做低秩？
满秩 $W \in \mathbb{R}^{D \times D}$ 每条样本 $2D^2$，$D = 3456$ 时 23.9 MFLOPs、1194 万参数；低秩 $UV^\top$、$r = 512$ 时 $4Dr = 7.1$ MFLOPs、354 万参数。论文观察到学出来的矩阵本来就近似低秩，拆开几乎不损失效果。
:::

::: details Q：同样几百亿参数，为什么排序模型比 LLM 便宜这么多？
参数几乎全在 embedding 表里，每条样本只查几百行；dense 部分只有千万级参数。DLRMv2 训练每条样本约 96 MFLOPs，7B LLM 每 token 42 GFLOPs。每参数每样本 0.004 FLOP 对 6 FLOP。
:::

::: details Q：排序模型训练的 MFU 一般多少？低在哪？
按 Meta model-A 的数字反推约 TF32 峰值的 11%（假设 638 MFLOPs 是前向、训练按 3 倍算）。低在：查表只搬字节、all-to-all 在关键路径上、表多导致负载不均和小 kernel、每卡 batch 小矩阵窄、数据喂不饱。
:::

::: details Q：GPU 算力用不满，怎么办？
两条路。一是让现有计算更快：均衡切表、fp16 表、量化 all-to-all、加大 batch、融合 kernel、流水重叠，Neo 叠起来 +87%。二是反过来利用空闲算力：把 dense 部分做大，比如更深的交叉、用户行为序列的 attention，用算力换效果。
:::

## 手撕

DLRM 的点积交叉和 DCN-v2 的低秩交叉层，再按配置数一遍每条样本的 FLOPs：

```python
import torch, torch.nn as nn

def dot_interaction(dense, sparse):
    """dense: [B, d]（底层 MLP 输出）；sparse: [B, T, d]（T 个 pooled embedding）"""
    x = torch.cat([dense.unsqueeze(1), sparse], dim=1)        # [B, T+1, d]
    z = torch.bmm(x, x.transpose(1, 2))                       # [B, T+1, T+1] 两两点积
    i, j = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
    return torch.cat([dense, z[:, i, j]], dim=1)              # [B, d + (T+1)T/2]

class LowRankCross(nn.Module):
    """DCN-v2：x_{l+1} = x0 * (U (V x_l) + b) + x_l"""
    def __init__(self, D, r, layers):
        super().__init__()
        self.V = nn.ModuleList(nn.Linear(D, r, bias=False) for _ in range(layers))
        self.U = nn.ModuleList(nn.Linear(r, D) for _ in range(layers))
    def forward(self, x0):
        x = x0
        for V, U in zip(self.V, self.U):
            x = x0 * U(V(x)) + x
        return x

B, T, d, r = 4, 26, 128, 512
dense, sparse = torch.randn(B, d), torch.randn(B, T, d)
print("dot 交互输出维度:", dot_interaction(dense, sparse).shape[1])     # 128 + 351
D = (T + 1) * d
x0 = torch.cat([dense.unsqueeze(1), sparse], 1).flatten(1)            # [B, 3456]
cross = LowRankCross(D, r, layers=3)
print("DCN 输出:", tuple(cross(x0).shape), " 参数:", sum(p.numel() for p in cross.parameters()))

def mlp_flops(sizes):                                                 # 每条样本前向：2 * in * out
    return sum(2 * a * b for a, b in zip(sizes[:-1], sizes[1:]))
f = mlp_flops([13, 512, 256, 128]) + 3 * 4 * D * r + mlp_flops([D, 1024, 1024, 512, 256, 1])
print(f"DLRMv2 前向 ≈ {f / 1e6:.1f} MFLOPs / 样本，训练 ≈ {3 * f / 1e6:.0f} MFLOPs / 样本")
# dot 交互输出维度: 479
# DCN 输出: (4, 3456)  参数: 10627200
# DLRMv2 前向 ≈ 32.1 MFLOPs / 样本，训练 ≈ 96 MFLOPs / 样本
```

常见变体：把 `LowRankCross` 换成满秩（`nn.Linear(D, D)`）对比参数和 FLOPs；点积交叉只保留指定的特征对；用 `torch.profiler` 看 bmm 和 MLP 各占多少时间。

## 参考

- [Deep Learning Recommendation Model for Personalization and Recommendation Systems（DLRM，arXiv 1906.00091）](https://arxiv.org/abs/1906.00091)：底层 / 顶层 MLP、点积交叉
- [DCN V2（arXiv 2008.13535）](https://arxiv.org/abs/2008.13535)：交叉层公式、低秩分解
- [MLPerf DLRMv2 参考实现（torchrec_dlrm）](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm)：层宽、DCN 层数与秩、batch
- [TorchRec `DLRM_DCN`](https://github.com/meta-pytorch/torchrec/blob/main/torchrec/models/dlrm.py)：交叉层输入维度 $(T+1)d$
- [Software-Hardware Co-design for Fast and Scalable Training of DLRMs（Neo / ZionEX，arXiv 2104.05158）](https://arxiv.org/abs/2104.05158)：表 3 model-A 规模与吞吐、§8.4 各项优化的收益
