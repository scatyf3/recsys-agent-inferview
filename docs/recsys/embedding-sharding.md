---
title: Embedding 分片与 all-to-all：table/row/column-wise、TorchRec planner、训练流水
status: draft
tags: [sharding, all-to-all, torchrec, planner, pipeline, model-parallel]
difficulty: 4
order: 5
related: []
recsys: [rs-p-shard, rs-p-a2a, rs-hw-net]
---

# Embedding 分片与 all-to-all：table/row/column-wise、TorchRec planner、训练流水

> embedding 按表、按行、按列切到各张卡上，dense 部分每张卡一份；两边之间靠两次 all-to-all 换数据，这两次通信卡在关键路径上，所以要切得均衡、传得少、尽量和计算重叠

## 一句话结论

GPU 同步训练的标准形态是「embedding 模型并行 + dense 数据并行」：每张卡拿自己那份样本的 ID，经 input all-to-all 发给持有对应表的卡，查完的 pooled 向量再经 output all-to-all 发回来。TorchRec 支持 7 种切法：数据并行（小表复制）、table-wise（整张表放一张卡）、row-wise、column-wise，以及节点内按行或按列、节点间按表的两种层次组合，再加 grid。output all-to-all 每卡收 $B_l \sum_t d_t s$ 字节：DLRMv2 在 8 卡上每卡 95 MB（fp32）。表要切得均衡：DLRMv2 里一张 4000 万行、每条样本查 100 次的表占全部查表字节的 47%，整张放一张卡，8 卡的查表效率只有 27%。Meta 的 Neo 用 table/column/data 混合切法、fp16 表、fp16/bf16 的 all-to-all 和更大的 batch，在 128 张 A100 上把 model-A 的吞吐比 TF32 基线提高 87%。TorchRec 的 `TrainPipelineSparseDist` 同时有 3 个 batch 在飞：$i+2$ 在拷到卡上、$i+1$ 在做 input all-to-all、$i$ 在查表和算 dense。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $R$ | 卡数（rank 数） | 8 或 64 |
| $g$ | 每个节点的卡数 | 8 |
| $B_g$ | 全局 batch | 65536 |
| $B_l$ | 每卡 batch，$B_g / R$ | 8192（$R = 8$） |
| $T$ | 表数 | 26 |
| $d_t$ | 第 $t$ 张表的维度 | 128 |
| $L_t$ | 第 $t$ 张表每条样本几个 ID | DLRMv2 里 1 到 100 |
| $s$ | 每个数的字节数 | fp32 4、fp16 2 |
| $\text{BW}_{\text{nv}}$ | 节点内 NVLink 单向带宽，每卡 | 300 GB/s |
| $\text{BW}_{\text{net}}$ | 跨节点网卡带宽，每卡 | 25 GB/s（200 Gbps） |

GB = $10^9$、MB = $10^6$ 字节。带宽取 Meta ZionEX 节点（[arXiv 2104.05158](https://arxiv.org/abs/2104.05158) 表 2）：8 张 A100、节点内 scale-up 2.4 TB/s 单向（每卡 300 GB/s），每卡一块 200 Gbps 的 RoCE 网卡。模型取 MLPerf DLRMv2：26 张表，$d = 128$，每条样本合计 214 个 ID。

### 1. 一步训练里数据怎么流

每张卡读自己的 $B_l$ 条样本，dense 部分（MLP、交叉层）每卡一份完整副本，embedding 表分散在各卡上：

1. **input dist**：每张卡把自己样本里的 ID 按「这一行在哪张卡上」分好，all-to-all 发出去（TorchRec 里 ID 用 KeyedJaggedTensor 表示：按特征名分组的 values 加 lengths）。
2. **lookup**：每张卡在自己持有的表（分片）上做 TBE 查表、pooling，结果是「所有发来的样本」在「本卡的表」上的 pooled 向量。
3. **output dist**：all-to-all 把 pooled 向量发回样本所在的卡。之后每张卡有自己 $B_l$ 条样本在全部 $T$ 张表上的向量。
4. **dense 前向、反向**：本卡算交叉层和 MLP。
5. **反向的 output dist**：pooled 向量的梯度按原路 all-to-all 发回持有表的卡，TBE 融合反向加优化器原地更新（[Embedding 查表](/recsys/embedding-lookup)）。
6. **dense 梯度 AllReduce**：和普通数据并行一样，可以和反向重叠。

两次 all-to-all 和 AllReduce 的区别：AllReduce 的梯度在反向过程中逐层产生，可以边算边发；output all-to-all 的结果是 dense 前向的输入，不到就没法往下算，在关键路径上（Neo §2 原话：数据依赖导致这些 all-to-all 在关键路径上）。

### 2. 切法

TorchRec 的 `ShardingType`（[types.py](https://github.com/meta-pytorch/torchrec/blob/main/torchrec/distributed/types.py) 的注释）：

| 切法 | 怎么切 | 前向通信 |
|---|---|---|
| `DATA_PARALLEL` | 每张卡一份完整副本 | 无；反向对整张表的梯度做 AllReduce |
| `TABLE_WISE` | 整张表放一张卡 | input / output 各一次 all-to-all |
| `ROW_WISE` | 按行号分段，切到所有卡 | ID 按行段分桶发出；各卡的部分和要 reduce-scatter |
| `COLUMN_WISE` | 按维度切成几块，每块当一张小表放一张卡 | ID 要复制给每个列分片；输出拼接 |
| `TABLE_ROW_WISE` | 节点内按行切，节点间按表放 | row-wise 的 reduce-scatter 只走 NVLink |
| `TABLE_COLUMN_WISE` | 节点内按列切，节点间按表放 | 同上，列分片 |
| `GRID_SHARD` | 每张卡拿一部分行和一部分列 | 两者都有 |

**2.1 table-wise**。最简单，ID 和输出都不用额外处理，通信最省（Neo §4）。持有表 $t$ 的卡要为全部 $B_g$ 条样本算这张表，然后把每张卡的 $B_l$ 行发回去。每张卡在 output all-to-all 里收到的字节：

$$
V_{\text{out}} = B_l \cdot \sum_{t} d_t \cdot s \cdot \frac{R - 1}{R}
$$

（$\frac{R-1}{R}$：平均有 $1/R$ 的表就在本卡上，不用传。）

DLRMv2，$R = 8$，$B_l = 8192$：

1. 每条样本的 pooled 向量 $26 \times 128 \times 4 = 13{,}312$ B。
2. $V_{\text{out}} = 8192 \times 13{,}312 \times 7/8 = 95.4$ MB，反向梯度再传一次同样大。
3. 单节点走 NVLink：$95.4\ \text{MB} / 300\ \text{GB/s} = 0.32$ ms。
4. input dist：$8192 \times 214$ 个 ID，int64 是 14.0 MB，发出去 7/8；ID 用 int32 能减半。

缺点：一张表大于一张卡的显存就放不下；表的大小和访问量差别很大，很难分均匀。

**2.2 row-wise**。大表按行号分成 $R$ 段，第 $r$ 段在第 $r$ 张卡上。ID 按所属行段分桶发出；每张卡对**全部** $B_g$ 条样本算出「只用本卡这段行」的部分和，再 reduce-scatter，每张卡拿到自己 $B_l$ 条样本的完整和。一张表的通信：每张卡发 $\frac{R-1}{R} B_g d s$，table-wise 只有持有表的那一张卡发这么多。所有表都按行切，每卡的通信量是 table-wise 的 $R$ 倍（Neo：通信量随 trainer 数线性增长）。好处是能放下任意大的表、负载天然均匀。

**2.3 column-wise**。把 $d = 128$ 切成 4 块各 32 维，每块是一张独立的小表，按 table-wise 放。通信模式和 table-wise 一样，代价是 ID 要发给每个列分片（input 量乘以分片数），rowwise 优化器的状态也变成每个分片一份。适合维度大的表，用来把一张热表拆开分担（Neo §4）。

**2.4 data parallel**。小表直接复制：前向不通信，反向对整张表 AllReduce，每卡收发约 $2 n d s$。和 table-wise 比：

1. 1000 行、$d = 128$ 的表：AllReduce 约 $2 \times 1000 \times 512 = 1.0$ MB。
2. 同一张表走 table-wise，每卡 output 要收 $8192 \times 512 \times 7/8 = 3.7$ MB，反向再传一次。

粗略地说行数 $n$ 小于 $B_l/2$ 量级的表复制更划算（两边都乘了 $d s$，比的是 $2n$ 和 $B_l$）。

**2.5 层次组合**。节点内 NVLink 比跨节点网卡快一个数量级（300 对 25 GB/s），所以 `TABLE_ROW_WISE` 把一张表分给一个节点，在节点内按行切：通信量大的 reduce-scatter 只在节点内走，跨节点仍是 table-wise 的 all-to-all（Neo 的「先按表、再按行」的层次切法）。

### 3. 切得均衡：规划器

**不均衡的代价**。查表和 dense 计算要等所有卡做完下一次 all-to-all，所以耗时由最重的卡决定，效率 = 平均负载 / 最大负载。

**DLRMv2 的例子**。26 张表里 5 张各 4000 万行。其中第 22 个特征的表每条样本查 100 个 ID，而 214 个 ID 里剩下的 114 个分散在其他 25 张表上：

1. 这一张表占全部查表字节的 $100 / 214 = 46.7\%$。
2. table-wise 放在 8 卡之一，理想每卡 12.5%，这张卡至少 46.7%：查表效率 $12.5 / 46.7 = 27\%$。
3. 显存上也放不进 40 GB 的卡：$4\times10^7 \times 128 \times 4 = 20.5$ GB，加逐元素 Adagrad 状态共 41 GB。
4. 按列切成 4 份（各 32 维）放 4 张卡：每份占查表字节的 $100 \times 32 \times 4 / 109{,}568 = 11.7\%$，代价是这张表的 100 个 ID 要发给 4 张卡。按行切成 8 份也行，代价是 reduce-scatter。

**TorchRec 的 `EmbeddingShardingPlanner`**（[planners.py](https://github.com/meta-pytorch/torchrec/blob/main/torchrec/distributed/planner/planners.py)）：

1. **storage reservation**：从每张卡的 HBM 里扣掉框架、dense 部分和激活要用的，剩下的给 embedding。
2. **enumerator**：对每张表枚举候选方案：切法 × 计算 kernel（放 HBM、UVM 缓存等），并给每个方案估显存和耗时（perf / storage estimator）。
3. **proposer**：从候选里组合出整体方案，默认几种一起上：网格搜索、两种贪心、统一切法。
4. **partitioner**：把每个方案的分片分到具体的卡上，满足显存约束，默认 `GreedyPerfPartitioner`；放不下的方案反馈给 proposer。
5. 在可行方案里选估计耗时最好的。

**Neo 的代价模型**（§4.1）：代价 = 通信量（按消息大小估） + 负载不均。每卡的查表负载估为 表数 × 全局 batch × 平均每条样本 ID 数 × 维度；分配用贪心（按代价从大到小，每次给当前总代价最小的卡）或 Karmarkar–Karp 差分法。model-A 约 1000 张表，基线不到 700 K 样本/秒，剖析发现各卡查表时间差很大；用 table-wise + column-wise + data parallel 的混合切法后吞吐涨 20%（作者自测）。

### 4. 训练流水：sparse dist 和 dense 计算重叠

一步里能并行的东西：下一个 batch 的数据拷贝和 ID 分发，不依赖这一步的计算结果。

**TorchRec `TrainPipelineSparseDist`**（[train_pipelines.py](https://github.com/meta-pytorch/torchrec/blob/main/torchrec/distributed/train_pipeline/train_pipelines.py) 的 docstring）：同时最多 3 个 batch，分在三个 CUDA stream 上：

| batch | 在做什么 | stream |
|---|---|---|
| $i+2$ | 从主机内存拷到 GPU | memcpy stream |
| $i+1$ | input dist（ID 的 all-to-all） | data dist stream |
| $i$ | 查表、output dist、前向、反向、优化器 | 默认 stream |

MLPerf DLRMv2 的参考实现就用这个 pipeline（`TRAIN_PIPELINE_STAGES = 3`）。

为什么不影响结果：batch $i+1$ 提前做的只是发 ID，不读表；它真正查表要等 batch $i$ 的反向把表更新完，所以和不加流水的训练数学上一样。更激进的做法是把 batch $i+1$ 的查表也提前，和 batch $i$ 的 dense 计算重叠，代价是查到的是还没被 batch $i$ 更新的旧值（TorchRec 的 `TrainPipelineSemiSync` 属于这一类）。

**Neo 的重叠方式**（§4.2）：batch $i+1$ 的 input all-to-all 和 batch $i$ 顶层 MLP 的前向重叠；pooled embedding 的 output all-to-all 和底层 MLP（处理连续特征的那部分）的前向重叠，因为底层 MLP 不依赖 embedding。剖析图里 CPU 到 GPU 的拷贝完全被藏住，暴露的通信时间远小于 all-to-all 加 AllReduce 的串行时间（图 10，作者自测）。

### 5. 网络：从一个节点到多个节点

全局 batch 不变，从 8 卡（1 个节点）扩到 64 卡（8 个节点）：

| | $R = 8$ | $R = 64$ |
|---|---|---|
| $B_l$ | 8192 | 1024 |
| 每卡 output all-to-all 收 | 95.4 MB | $1024 \times 13{,}312 \times 63/64 = 13.4$ MB |
| 其中跨节点 | 0 | $\times 56/63$ = 11.9 MB |
| 通信时间 | 95.4 MB / 300 GB/s = 0.32 ms | 11.9 MB / 25 GB/s = 0.48 ms |
| 每卡 dense 计算（按 0.84 ms / 8192 条） | 0.84 ms | 0.11 ms |

每卡的 dense 计算缩小 8 倍，通信时间反而变长：跨节点带宽只有 NVLink 的 1/12。这就是 DLRM 弱扩展效率低的原因，Neo 在 16 个节点上 model-A 的扩展效率约 50%、model-I 约 75%（§8.3）。对策：

1. **压缩 all-to-all**：前向用 fp16、反向用 bf16，Neo 实测近 30% 加速、训练质量无损（作者自测）。
2. **更大的 batch**：全局 batch 从 64K 加到 256K，每卡的计算变多、通信更饱和（同样要调好超参）。
3. **层次切法**：通信量大的 row-wise 部分留在节点内。

Neo 在 model-A 上把这些叠起来（优化切法、fp16 表、量化通信、256K batch），比 TF32、64K batch 的基线吞吐提高 87%（图 11，作者自测）。all-to-all 本身的算法见 [集合通信](https://scatyf3.github.io/ai-infra-inferview/parallel/collective-comm)，计算通信重叠的通用手法见 [计算通信 Overlap](https://scatyf3.github.io/ai-infra-inferview/parallel/comm-overlap)；LLM 里同样形态的 all-to-all 是 MoE 的 dispatch / combine（[MoE 与 EP](https://scatyf3.github.io/ai-infra-inferview/parallel/moe-ep)）。

## 面试追问

::: details Q：为什么 embedding 用模型并行、dense 用数据并行？
embedding 表大（几十 GB 到几十 TB）但每步只碰很少的行，复制到每张卡放不下，也没必要；dense 部分小（DLRMv2 约 1600 万参数）但计算密集，复制一份、各算各的样本、梯度 AllReduce 最简单。两边之间用 all-to-all 换数据。
:::

::: details Q：table-wise、row-wise、column-wise 各自的代价？
table-wise 通信最省，但大表放不下、负载难均匀。row-wise 能放任意大的表、负载均匀，但每张卡要为全局 batch 算部分和再 reduce-scatter，通信随卡数线性增长。column-wise 通信模式和 table-wise 一样，能把一张热表拆给多张卡，但 ID 要复制给每个列分片。小表直接数据并行。
:::

::: details Q：output all-to-all 为什么不能像梯度 AllReduce 那样藏在计算后面？
AllReduce 的梯度在反向里逐层产生，后面的层还在算时前面层的梯度已经可以发。output all-to-all 的结果是交叉层和顶层 MLP 的输入，不到就没法往下算，所以在关键路径上。能藏的只有不依赖它的部分：底层 MLP 的前向、下一个 batch 的 input all-to-all 和数据拷贝。
:::

::: details Q：TrainPipelineSparseDist 会不会让模型用到旧参数？
不会。它提前做的是下一个 batch 的数据拷贝和 ID 分发，不读表；下一个 batch 的查表仍在这一步的反向更新之后。把查表也提前的变体（semi-sync）才会用到旧一步的 embedding。
:::

::: details Q：规划器在优化什么？
在每张卡的显存约束下，让最慢的那张卡尽量快：代价 = 查表负载（表数 × batch × 每条样本 ID 数 × 维度）的不均衡 + 通信量。TorchRec 的做法是对每张表枚举切法和放置位置、估显存和耗时，再用网格搜索和贪心组合，按耗时估计选最好的。
:::

::: details Q：从 1 个节点扩到 8 个节点，为什么吞吐涨不到 8 倍？
全局 batch 不变时每卡计算缩小 8 倍，但 all-to-all 的大部分流量从 NVLink（300 GB/s）换到网卡（25 GB/s），每卡通信时间反而变长；再加上负载不均。Neo 在 16 个节点上 model-A 的扩展效率约 50%。对策是压缩通信、加大 batch、把行切留在节点内。
:::

## 手撕

row-wise 分片的 ID 分桶：给定每张卡负责的行段，把一个 batch 的 ID 分到各卡，并换成卡内的局部行号（这就是 input dist 前在本卡上做的事）：

```python
import torch

def bucketize_row_wise(ids, num_rows, R):
    """ids: [nnz] 全局行号；返回每张卡收到的局部行号，以及每个 ID 去了哪张卡"""
    block = (num_rows + R - 1) // R                  # 每张卡负责 block 行
    owner = ids // block                             # 这一行在哪张卡
    order = torch.argsort(owner, stable=True)        # 按目标卡排好，all-to-all 要求连续
    counts = torch.bincount(owner, minlength=R)      # 发给每张卡几个（all-to-all 的 split sizes）
    local = (ids - owner * block)[order]             # 换成卡内局部行号
    return list(torch.split(local, counts.tolist())), owner

ids = torch.tensor([5, 98, 33, 61, 7, 50, 99])
per_rank, owner = bucketize_row_wise(ids, num_rows=100, R=4)
print(owner.tolist())
print([t.tolist() for t in per_rank])
# [0, 3, 1, 2, 0, 2, 3]
# [[5, 7], [8], [11, 0], [23, 24]]
```

`counts` 就是 all-to-all 的 split sizes，发完以后每张卡在自己的局部表上查表。反向时按 `order` 的逆排列把结果放回原位。常见变体：table-wise 的分桶（owner 由表号决定）、column-wise 的复制（同一组 ID 发给多张卡）、用 `torch.distributed.all_to_all_single` 真正发出去。

## 参考

- [Software-Hardware Co-design for Fast and Scalable Training of DLRMs（Neo / ZionEX，arXiv 2104.05158）](https://arxiv.org/abs/2104.05158)：§4 四种切法与代价模型、§4.2 流水、§8 扩展效率与吞吐优化（混合切法 +20%、量化通信约 30%、合计 +87%）、表 2 节点配置
- [TorchRec `ShardingType`](https://github.com/meta-pytorch/torchrec/blob/main/torchrec/distributed/types.py)：7 种切法的定义
- [TorchRec `EmbeddingShardingPlanner`](https://github.com/meta-pytorch/torchrec/blob/main/torchrec/distributed/planner/planners.py)：storage reservation、enumerator、proposer、partitioner
- [TorchRec train pipelines](https://github.com/meta-pytorch/torchrec/blob/main/torchrec/distributed/train_pipeline/train_pipelines.py)：`TrainPipelineSparseDist` 的三阶段流水
- [MLPerf DLRMv2 参考实现（torchrec_dlrm）](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm)：表规模、multi-hot 大小、3 阶段 pipeline
