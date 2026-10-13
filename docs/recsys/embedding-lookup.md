---
title: Embedding 查表：EmbeddingBag / TBE kernel、访存账、HBM/DRAM/SSD 多级缓存
status: draft
tags: [embeddingbag, tbe, fbgemm, memory-bandwidth, cache, uvm]
difficulty: 3
order: 3
related: []
recsys: [rs-k-tbe, rs-e-cache, rs-hw-mem]
---

# Embedding 查表：EmbeddingBag / TBE kernel、访存账、HBM/DRAM/SSD 多级缓存

> 查表几乎不做计算，只搬字节：所以 kernel 要把上千张表塞进一次 launch、把反向和优化器融在一起，放不进 HBM 的部分靠缓存命中率撑住带宽

## 一句话结论

EmbeddingBag 是「按 ID 取行、按 bag 求和」：每取一行 $d \cdot s$ 字节只做 $d$ 次加法，fp32 的算术强度 0.25 FLOP/B，比 A100 的拐点（约 200 FLOP/B）低三个数量级，完全被带宽卡住。MLPerf DLRMv2 每条样本查 214 行、每行 512 B，共 110 KB；全局 batch 65536 时前向一步就要读 7.2 GB。FBGEMM 的 TBE（Table Batched Embedding）把所有表拼成一次 kernel（Meta 的模型有上千张表，不合并就是每步上千次 launch），反向不产生梯度张量，而是排序、合并重复行后直接在原地做优化器更新。表放不进 HBM 时，用 HBM 当 DRAM 的缓存：有效带宽 $1 / (h/B_{\text{HBM}} + (1-h)/B_{\text{PCIe}})$，命中率 90% 时只剩约 220 GB/s，99% 时约 960 GB/s，所以命中率的每一个 9 都很值钱。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $B$ | 一步的样本数（batch） | 全局 65536，每卡 8192 |
| $T$ | 表的个数 | 26 |
| $L_t$ | 第 $t$ 张表每条样本平均几个 ID（pooling factor） | DLRMv2 合计 $\sum_t L_t = 214$ |
| $d$ | embedding 维度 | 128 |
| $s$ | 每个数的字节数 | fp32 4 |
| $B_{\text{HBM}}$ | 单卡 HBM 带宽 | 1555 GB/s（A100 40 GB） |
| $B_{\text{PCIe}}$ | GPU 经 PCIe 读主机内存的有效带宽 | 25 GB/s（假设） |
| $h$ | HBM 缓存命中率 | 0.9 / 0.99 |

GB = $10^9$ 字节。硬件数字取自 Meta ZionEX 节点（[arXiv 2104.05158](https://arxiv.org/abs/2104.05158) 表 2、§8.1）：8 张 A100，合计 320 GB HBM、12.4 TB/s（每卡 1.55 TB/s）；4 路 CPU 共 1.5 TB DDR、320 GB/s；每张卡一块 200 Gbps 的 RoCE 网卡。

### 1. EmbeddingBag 做什么

一条样本的一个稀疏特征是一组 ID（一个 bag），比如「最近点过的 3 个商品」。输出是这组 ID 的向量之和（或均值、加权和）：

$$
\text{out}[b, t] = \sum_{i \in \text{bag}(b, t)} w_i \cdot E_t[\text{idx}_i]
$$

$w_i$ 是可选的逐 ID 权重（`per_sample_weights`），不给就是 1。输入用 CSR 格式：所有 ID 拼成一个一维的 `indices`，再给一个 `offsets` 标出每个 bag 的起点。FBGEMM 的 TBE 里 `offsets` 长 $B \cdot T + 1$（[TBE 文档](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/tbe_ops_training.html)）。

例：一张表、3 条样本，bag 分别是 $[3, 7]$、$[1]$、$[\,]$：`indices = [3, 7, 1]`，`offsets = [0, 2, 3, 3]`。第 3 个 bag 是空的，输出全 0。

为什么不用 `nn.Embedding` 先取出 $[\text{nnz}, d]$ 再求和：中间结果要写回显存再读一遍。EmbeddingBag 在一个 kernel 里边取边加，每个 bag 只写一次 $d$ 维结果。

### 2. 访存账：为什么一定是带宽瓶颈

每取一行：读 $d \cdot s$ 字节，做 $d$ 次加法。

$$
\text{算术强度} = \frac{d}{d \cdot s} = \frac{1}{s} \ \text{FLOP/B} \quad (\text{fp32 为 } 0.25)
$$

A100 bf16 稠密峰值 312 TFLOP/s、HBM 1.555 TB/s，拐点约 $312 / 1.555 \approx 200$ FLOP/B。查表比拐点低 800 倍，只要算带宽。而且访问是随机的：每行 512 B 连续，但下一行在哪完全由 ID 决定，几乎没有跨行的局部性，只能靠热门行反复出现。

**DLRMv2 一步要读多少**（MLPerf 参考实现的 multi-hot 大小加起来 214）：

1. 每条样本：$214 \times 128 \times 4 = 109{,}568$ B，约 110 KB。
2. 全局 batch 65536：$110\ \text{KB} \times 65536 = 7.18$ GB，这只是前向读权重。
3. 8 卡平分（理想情况，分片见 [Embedding 分片](/recsys/embedding-sharding)）：每卡 0.90 GB，按 1.555 TB/s 读完要 0.58 ms。
4. 反向加优化器：每个被查到的行（去重后）要读梯度、读权重、读状态、写权重、写状态，字节数是前向的几倍。

对比 dense 部分：DLRMv2 每条样本前向约 32 MFLOPs（[排序模型的算力账](/recsys/ranking-model-compute)），8192 条在 312 TFLOP/s 上是 0.84 ms。查表和 dense 计算在同一个量级，而查表一个 FLOP 都不进 MFU 的分子。

### 3. TBE：把上千张表合成一次 kernel

**问题一：kernel 太多**。Meta 的 model-A 有约 1000 张表（Neo 表 3）。每张表单独一个 EmbeddingBag，前向、反向、优化器各一次，每步几千次 launch。按每次几微秒的 launch 开销算（假设 5 µs × 3000 = 15 ms），比查表本身还慢。

**TBE 的做法**（FBGEMM `SplitTableBatchedEmbeddingBagsCodegen`）：

1. 所有表的权重拼进一块连续显存，记下每张表的起始偏移。`embedding_specs` 给出每张表的（行数，维度，放在哪，在哪算）。
2. 一个 kernel 处理全部 $B \times T$ 个 bag：每个 warp 负责若干个 bag，按表偏移加上 ID 找到行。
3. 不同表可以有不同维度（维度要是 4 的倍数）。

**问题二：梯度太大**。如果反向像普通参数一样产生一个和表一样大的梯度张量，DLRMv2 就是又一个 104.5 GB。实际上一步只碰到被查到的那些行。

**TBE 把反向和优化器融在一起**（文档原话：backward 和 optimizer 融合，反向时直接更新表）：

1. 对本步所有（行号，来自哪个 bag）按行号排序。
2. 同一行出现多次的，把来自各个 bag 的输出梯度加起来（segment reduce）。排序后求和的顺序固定，结果是确定性的。
3. 对每个不同的行，直接在原地做优化器更新（读状态、算、写回）。

梯度张量从来没有完整出现过。优化器可选 `EXACT_SGD`（默认）、`EXACT_ADAGRAD`、`EXACT_ROWWISE_ADAGRAD`、`ADAM`、`PARTIAL_ROWWISE_ADAM`、`LAMB` 等；`NONE` 表示不更新、输出稀疏梯度。代价是优化器必须写进 kernel，换一个新优化器要改 FBGEMM。

### 4. 多级存储：HBM 放不下怎么办

| 层 | 容量（ZionEX 每节点） | 带宽 | 访问方式 |
|---|---|---|---|
| HBM | 320 GB | 12.4 TB/s（每卡 1.55） | GPU 直接读 |
| DRAM | 1.5 TB | 320 GB/s | GPU 经 PCIe，或 CPU 查完再拷 |
| SSD | 数 TB 到数十 TB | 单盘几 GB/s（典型 NVMe，假设） | 按块读，延迟几十微秒 |

FBGEMM 的 `EmbeddingLocation` 给出四种放法：

1. `DEVICE`：整张表在 HBM。
2. `MANAGED`：表在 UVM（统一虚拟内存），GPU 访问时由驱动按页迁移。受 PCIe 带宽限制。
3. `MANAGED_CACHING`：表在 UVM，HBM 里放一个缓存。缓存行数 = `cache_load_factor`（默认 0.2）× 所有表的总行数，替换策略 LRU（默认）或 LFU。
4. `HOST`：表在 CPU 内存，在 CPU 上算。

Neo 的软件缓存：32 路组相联，组相联度等于 warp 大小，一个 warp 一次比对一组；LRU 或 LFU。PCIe 和 HBM 之间约 50 倍的带宽差，软件缓存比直接用 UVM 端到端快约 15%（§5.2，作者自测）。

**有效带宽**。命中的部分走 HBM，没命中的走 PCIe，读同样字节数的时间按比例相加：

$$
B_{\text{eff}} = \frac{1}{\dfrac{h}{B_{\text{HBM}}} + \dfrac{1-h}{B_{\text{PCIe}}}}
$$

$B_{\text{HBM}} = 1555$、$B_{\text{PCIe}} = 25$ GB/s：

| $h$ | $B_{\text{eff}}$ |
|---|---|
| 0.9 | $1/(0.9/1555 + 0.1/25) = 218$ GB/s |
| 0.99 | 965 GB/s |
| 0.999 | 1465 GB/s |

命中率 90% 时，10% 的未命中占了 87% 的时间。

**命中率能有多高**。ID 访问通常接近幂律。假设一张 4000 万行的表按 Zipf（指数 1）被访问，缓存放最热的 20% 行。第 $i$ 热的行被访问的概率正比于 $1/i$，前 $K$ 行的总概率是调和数之比：

$$
h = \frac{H_K}{H_N} \approx \frac{\ln K + 0.577}{\ln N + 0.577} = \frac{\ln(8\times10^6) + 0.577}{\ln(4\times10^7) + 0.577} = \frac{16.47}{18.08} = 91\%
$$

只缓存热门行就能有九成命中，但离 99% 还远；而且这是静态的最优情况，LRU 在流量变化时会更低。补救：

1. **预取**：下一个 batch 的 ID 在这一步就知道了（[训练流水](/recsys/embedding-sharding)里 batch $i+1$ 的 ID 已经到卡上），提前把没命中的行拷进缓存，PCIe 传输和本步计算重叠。
2. **batch 内去重**：一个 batch 里同一行只取一次。
3. **按表决定放哪**：小而热的表整张放 HBM，大而冷的表用缓存，这是分片规划器要做的选择。

**SSD 这一层**。DRAM 也放不下时（几十 TB 的表），把冷行放 SSD。读 SSD 是按块（几 KB）的随机读，延迟在几十微秒，只能按 batch 提前批量取：先统计下个 batch 用到的行，从 SSD 读进 DRAM 或 HBM 缓存，训练时再查。Neo 的论文把 HBM + DDR + SSD 列为它的多级存储；百度 AIBox 把模型放在单机 SSD 上训练（Neo 的相关工作一节）。

## 面试追问

::: details Q：为什么说 embedding 查表是 memory bound？给个数。
每读一行 $d \cdot s$ 字节只做 $d$ 次加法，fp32 算术强度 0.25 FLOP/B；A100 的拐点约 200 FLOP/B，差 800 倍。而且是随机访问，几乎没有复用。所以优化方向全是少搬字节：低精度表、去重、缓存、把多次 launch 合成一次。
:::

::: details Q：TBE 相比逐表的 EmbeddingBag 快在哪？
三件事。一，所有表一次 launch，上千张表时省下几千次 kernel 启动；二，反向不生成完整梯度，按行排序、合并重复行后原地更新，省掉和表一样大的梯度张量；三，优化器融进 kernel，权重和状态只读写一次。代价是优化器种类固定在 FBGEMM 里。
:::

::: details Q：一个 batch 里同一个 ID 出现很多次，反向怎么处理？
TBE 先按行号排序，把同一行来自各个 bag 的梯度加起来，再对这一行做一次优化器更新。不先合并的话，多个线程同时写同一行会有竞争，用原子加又失去确定性，而且 Adagrad 这种非线性更新拆成多次做结果也不一样。
:::

::: details Q：UVM 和软件缓存有什么区别？
UVM 由驱动按页（KB 级）迁移，粒度粗、受 PCIe 限制，GPU 缺页时要等。软件缓存按行（几百字节）管理，组相联、可选 LRU/LFU，还能配合预取：Neo 报告比 UVM 端到端快约 15%（作者自测）。FBGEMM 的 `MANAGED_CACHING` 就是 UVM 打底、HBM 做缓存。
:::

::: details Q：缓存命中率从 90% 提到 99%，有效带宽变多少？
按 $1/(h/B_{\text{HBM}} + (1-h)/B_{\text{PCIe}})$，HBM 1555、PCIe 25 GB/s：90% 是 218 GB/s，99% 是 965 GB/s，4.4 倍。未命中的那部分走慢路径，决定了总时间。
:::

## 手撕

用 gather + `index_add_` 实现 sum pooling 的 EmbeddingBag，和 `F.embedding_bag` 对拍，再看反向的梯度落在哪些行：

```python
import torch, torch.nn.functional as F

def embedding_bag_sum(weight, indices, offsets):
    """weight: [n, d]；indices: [nnz]；offsets: [B+1]，第 b 个 bag 是 indices[offsets[b]:offsets[b+1]]"""
    B = offsets.numel() - 1
    lengths = offsets[1:] - offsets[:-1]                          # 每个 bag 几个 ID
    bag_of = torch.repeat_interleave(torch.arange(B), lengths)    # 每个 ID 属于哪个 bag
    rows = weight[indices]                                        # gather：[nnz, d]
    out = torch.zeros(B, weight.shape[1], dtype=weight.dtype)
    return out.index_add_(0, bag_of, rows)                        # 按 bag 累加

torch.manual_seed(0)
W = torch.randn(10, 4)
indices = torch.tensor([3, 7, 1, 2, 2, 9])
offsets = torch.tensor([0, 2, 3, 3, 6])     # 4 个 bag：[3,7] [1] [] [2,2,9]
ours = embedding_bag_sum(W, indices, offsets)
ref = F.embedding_bag(indices, W, offsets[:-1], mode="sum")
print(torch.allclose(ours, ref), ours[2])

W.requires_grad_(True)
embedding_bag_sum(W, indices, offsets).sum().backward()
print(W.grad[:, 0])                          # 只有出现过的行非零，ID 2 出现两次梯度为 2
# True tensor([0., 0., 0., 0.])
# tensor([0., 1., 2., 1., 0., 0., 0., 1., 0., 1.])
```

这里的 `W.grad` 是和表一样大的稠密张量，真实的 TBE 不会生成它：反向直接按 (行号, 梯度) 排序合并后更新那 5 行。常见变体：加 `per_sample_weights`（`rows` 先乘权重）、mean pooling（除以 `lengths`，注意空 bag）、多张表拼成一块权重加偏移。

## 参考

- [FBGEMM TBE 训练算子文档](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/tbe_ops_training.html)：`embedding_specs`、`EmbeddingLocation`、`cache_load_factor`、`cache_algorithm`、优化器类型、`indices` / `offsets` 格式
- [Software-Hardware Co-design for Fast and Scalable Training of DLRMs（Neo / ZionEX，arXiv 2104.05158）](https://arxiv.org/abs/2104.05158)：§5 kernel 融合与软件缓存（32 路组相联、比 UVM 快约 15%）、表 2 ZionEX 节点配置、表 3 模型规模
- [MLPerf DLRMv2 参考实现（torchrec_dlrm）](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm)：multi-hot 大小、batch 配置
- [DLRM（arXiv 1906.00091）](https://arxiv.org/abs/1906.00091)：EmbeddingBag 在 DLRM 里的用法
