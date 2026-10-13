---
title: Parameter Server：pull/push、同步与异步、staleness、PS 与 AllReduce 混合架构
status: draft
tags: [parameter-server, async-sgd, staleness, hybrid-training, allreduce]
difficulty: 3
order: 4
related: []
recsys: [rs-p-ps, rs-p-dense, rs-hw-cpu]
---

# Parameter Server：pull/push、同步与异步、staleness、PS 与 AllReduce 混合架构

> 参数按 key 切到一组 server 上，worker 每步只拉自己用到的行、推回梯度；稀疏的行很少被两个 worker 同时改，所以可以异步，稠密部分每步全员都改，必须同步

## 一句话结论

PS 把参数按 key 哈希到多台 server，worker 每步 pull 这个 batch 用到的行、push 梯度，优化器在 server 上跑。每步流量 = 去重后的行数 × $d \cdot s$，一个 4096 条样本、每条 200 个 ID、30% 不重复、$d = 64$ 的 batch 是 63 MB 拉加 63 MB 推。异步训练里一次更新用的参数平均落后 $W-1$ 步；但对一个每步出现概率为 $q$ 的稀疏行，期间被别的 worker 改过的次数只有约 $(W-1)q$，冷门行几乎不受影响。所以主流做法是混合：embedding 异步（或放在 PS 上），dense 部分用 AllReduce 同步。快手的 PERSIA 这样做到 100T 参数：embedding 异步、dense 同步，开源数据集上 AUC 和全同步差不到 0.1%，全异步差 0.5%–1.0%；Taobao-Ad 上比阿里 XDL 的同步模式快 7.12 倍（作者自测）。Meta 从 CPU 异步 PS（45 个 PS + 15 个 trainer）换到 GPU 全同步（128 张 A100），同一个模型快了 40 倍。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $W$ | worker 数 | 100 |
| $S$ | server 数 | 50 |
| $B$ | 每个 worker 每步的样本数 | 4096 |
| $L$ | 每条样本的 ID 总数（所有稀疏特征加起来） | 200 |
| $u$ | batch 内去重后剩下的比例 | 0.3（假设） |
| $d$ | embedding 维度 | 64 |
| $s$ | 每个数的字节数 | 4 |
| $P_{\text{dense}}$ | dense 参数量 | $1.6\times10^7$ |
| $\tau$ | staleness：算梯度时用的参数比当前落后几次更新 | |
| $q$ | 某一行在一个 worker 的一个 batch 里出现的概率 | $10^{-4}$ 到 0.5 |
| $\text{BW}$ | 网卡带宽 | 25 GB/s（200 Gbps）、1.25 GB/s（10 Gbps） |

GB = $10^9$ 字节，MB = $10^6$ 字节；1 Gbps = 0.125 GB/s。

### 1. 架构：key 分片、pull、push

[Li et al. OSDI 2014](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu) 的 PS 架构：

1. **server**：参数按 key 切成若干段，每台 server 管一段（一致性哈希或取模）。server 上存参数和它的优化器状态。
2. **worker**：读样本，从样本里收集这一步用到的 key，向对应的 server **pull** 这些行；前向、反向；把这些行的梯度 **push** 回去。
3. **server 收到梯度就更新**：优化器状态从不离开 server，worker 只传参数和梯度。

**每步流量**：

$$
U = u \cdot B \cdot L, \qquad
\text{pull} = \text{push} = U \cdot d \cdot s
$$

算一笔（$u$ 为假设）：

1. 查表次数 $4096 \times 200 = 819{,}200$。
2. 去重后 $U = 0.3 \times 819{,}200 = 245{,}760$ 行。
3. pull：$245{,}760 \times 64 \times 4 = 62.9$ MB，push 一样大。
4. 200 Gbps 网卡：62.9 MB / 25 GB/s = 2.5 ms；早期 10 Gbps 的 CPU 集群：50 ms。
5. 不去重就是 210 MB，所以 worker 拉之前一定先去重（手撕）。

**dense 参数放 PS 上为什么不行**。dense 参数每步每个 worker 都全部用到：$1.6\times10^7 \times 4 = 64$ MB，100 个 worker 每步共往 server 推 6.4 GB、再拉 6.4 GB。放在少数几台 server 上，那几台的网卡就是瓶颈（单台 25 GB/s 要 256 ms）；摊到所有 server 上，总流量和 ring AllReduce 相当（每个 worker 收发约 $2 \times 64 = 128$ MB，见 [集合通信](https://scatyf3.github.io/ai-infra-inferview/parallel/collective-comm)），但多了一跳，还要专门的机器。所以 dense 部分后来都改成 worker 之间直接 AllReduce。

### 2. 同步、异步、有界延迟

| 模式 | 规则 | 代价 |
|---|---|---|
| BSP（同步） | 每步等所有 worker 的梯度到齐再更新，所有人用同一版参数 | 最慢的 worker 决定速度 |
| ASP（异步） | 谁算完谁 push，server 收到就更新，worker 拉到什么版本就用什么 | 梯度是在旧参数上算的 |
| SSP / 有界延迟 | 允许异步，但最快的 worker 不能领先最慢的超过 $\tau_{\max}$ 步 | 折中，$\tau_{\max}$ 要调 |

有界延迟是 Li et al. 2014 提出的一致性模型之一，$\tau_{\max} = 0$ 就是 BSP，$\tau_{\max} = \infty$ 就是 ASP。

**staleness 有多大**。异步时，一个 worker 拉参数、算完、推回去的这段时间里，其他 $W - 1$ 个 worker 大约各推了一次。

1. **dense 参数**：每个 worker 每步都改全部 dense 参数，所以 $\tau \approx W - 1$。$W = 100$ 时梯度是在落后 99 次更新的参数上算的。
2. **稀疏的一行**：只有包含它的 batch 才会改它。期间被别的 worker 改过的次数约

$$
\mathbb{E}[\tau_{\text{row}}] \approx (W - 1)\, q
$$

| 行的热度 $q$ | $W = 100$ 时的 $\mathbb{E}[\tau_{\text{row}}]$ |
|---|---|
| $10^{-4}$（长尾 ID） | 0.0099 |
| $10^{-2}$ | 0.99 |
| 0.5（热门 ID） | 49.5 |

长尾的行几乎从不和别人冲突，异步对它们几乎无损；热门行和 dense 参数则每步都在被别人改。这就是 PERSIA 论文的两条观察：更新只碰模型的一小部分时，异步几乎不引入偏差；更新高度重叠时（如 dense 网络），staleness 限制收敛。

### 3. 混合：embedding 异步，dense 同步

[PERSIA](https://arxiv.org/abs/2111.05897)（快手 + ETH，2021）：

1. embedding 层占模型 99.99% 以上的参数，但计算很少；dense 网络参数少、计算密集。
2. **embedding 异步更新**：worker 把 embedding 梯度发回 PS 就继续，不等更新完成；下一步取 embedding 可以提前做。准备 embedding、更新 embedding 的时间都藏进 dense 的前向反向里。
3. **dense 同步更新**：NN worker 之间 AllReduce，梯度同步和反向重叠。
4. 三种角色：embedding PS（存表，CPU 大内存）、embedding worker（CPU，负责查表、pooling、转发）、NN worker（GPU，跑 dense）。最大规模的实验在 Google Cloud 上：8 台 8×A100 做 NN worker，100 台 30 vCPU 机器做 embedding worker，30 台 12 TB 内存的机器做 embedding PS，跑到 100T 参数。

效果（表 2 与 §6.2，作者自测）：

| | Taobao-Ad | Criteo-Ad | Kwai-Video |
|---|---|---|---|
| PERSIA 混合 | 65.86% | 77.93% | 80.47% |
| XDL 同步 | 65.93% | 78.02% | 80.51% |
| XDL 异步 | 65.66% | 77.10% | 80.01% |

1. 混合和全同步的 AUC 差在开源数据集上小于 0.1%，在 Kwai-Video 上小于 0.001%；全异步差 0.5%–1.0%。
2. 论文原话：生产推荐模型上 0.1% 的准确率下降可能对应百万美元级的收入损失。
3. 速度：Taobao-Ad 上比 XDL 同步快 7.12 倍、比 XDL 异步快 8.4 倍、比 PaddlePaddle 快 1.71 倍。

### 4. 从 CPU PS 到 GPU 同步

| 架构 | 稀疏部分 | dense 部分 | 代表 |
|---|---|---|---|
| CPU PS | PS 上 | PS 上（Meta 早期平台的 dense 是 Hogwild 式异步） | Meta 早期 DLRM 训练、阿里 XDL（同步、异步两种模式都有） |
| 混合 | PS（CPU 内存），异步 | GPU，AllReduce 同步 | PERSIA |
| GPU 同步 | GPU 上模型并行，all-to-all | GPU，AllReduce 同步 | Meta Neo / ZionEX、TorchRec、HugeCTR |

Meta 的对比（[Neo，arXiv 2104.05158](https://arxiv.org/abs/2104.05158) §8.2，作者自测）：同一个 model-A，之前的分布式 CPU 异步平台用 45 个 PS 和 15 个 trainer，每个 trainer 的 batch 约 150，再加机器就开始掉效果；ZionEX 全同步训练用 64K 的 batch，质量持平或更好，128 张 A100 上 1.2 M 样本/秒，快 40 倍。

GPU 同步的前提是 embedding 能放进（或缓存进）HBM，并且 all-to-all 够快，见 [Embedding 分片与 all-to-all](/recsys/embedding-sharding)。表大到几十 TB、或者 GPU 不够多时，PS 仍然是更便宜的选择：DRAM 每 GB 比 HBM 便宜得多。

### 5. CPU 侧：server 和特征处理

PS 方案里 CPU 干两份活：

1. **server**：本质是一个大内存的分布式 key-value 存储，瓶颈是内存容量和网卡，不是算力。热门 key 集中的那台 server 会成为热点，对策是按哈希打散、热门 key 在 worker 上留一份本地缓存（容忍一点 staleness），以及 batch 内去重。
2. **样本和特征处理**：读样本、解析、对 ID 做哈希、去重、按 server 分桶。这部分在 GPU 训练里同样存在。Meta 的训练数据是 PB 级、存在分布式文件系统上，Neo 专门用一层 reader 节点（数据接入服务）读数据、做预处理再喂给 GPU 节点，否则 GPU 节点的主机网络和 CPU 先被打满（§2、图 7）。

### 6. 容错

server 挂了，它那一段参数就没了。做法是每台 server 定期把自己的分片快照到持久存储，挂了就换机器、从最近的快照恢复那一段。Monolith 每天快照一次，算过一笔账：1000 台 PS、每台每天 0.01% 的故障率，平均 10 天坏一台，丢掉 1/1000 的用户一天的更新，可以接受（见 [在线学习与模型更新](/recsys/online-learning)）。同步训练的整体容错思路见 [大规模训练的容错](https://scatyf3.github.io/ai-infra-inferview/train/fault-tolerance)。

## 面试追问

::: details Q：为什么 embedding 可以异步，dense 不行？
一行 embedding 只在它的 ID 出现时被改，每步出现概率 $q$ 很小时，异步期间被别的 worker 改过的次数约 $(W-1)q$，几乎为 0。dense 参数每个 worker 每步都改，staleness 约 $W - 1$。PERSIA 实测混合方案的 AUC 和全同步差不到 0.1%，全异步差 0.5%–1.0%（作者自测）。
:::

::: details Q：PS 每步的通信量怎么估？
去重后的行数 × $d \cdot s$，pull 和 push 各一份。4096 条样本 × 200 个 ID、30% 不重复、$d = 64$ fp32：245,760 行，62.9 MB，200 Gbps 网卡 2.5 ms。不去重要 210 MB。
:::

::: details Q：优化器状态为什么放在 server 上？
状态和参数一样按 key 分片，放在 server 上就不用来回传：worker 只推梯度，server 本地读状态、更新、写回。放在 worker 上的话，每次 pull 都要连状态一起拉，Adagrad 多一份，Adam 多两份。
:::

::: details Q：既然 GPU 同步更快，PS 还有什么用？
容量和成本。表到几十 TB 时，DRAM 比 HBM 便宜得多；GPU 数量有限、或者模型 dense 部分很小时，CPU PS 加少量 GPU 更划算。在线学习的线上 serving 也常是 PS 形态（serving PS）。
:::

::: details Q：PS 上的热点怎么处理？
一是 key 按哈希打散，不按 ID 段分；二是 batch 内去重，同一个热门 ID 一步只拉一次；三是热门 key 在 worker 上缓存，隔几步再刷新，容忍一点 staleness；四是把特别热的 key 复制到多台 server。
:::

## 手撕

一个进程内的玩具 PS：key 按取模分到 shard，server 端跑 rowwise Adagrad；worker 先去重再 pull，反向后把合并好的梯度 push 回去：

```python
import torch

class Shard:                                   # 一台 server：自己那部分行 + rowwise Adagrad 状态
    def __init__(self, d, lr=0.1):
        self.w, self.acc, self.d, self.lr = {}, {}, d, lr
    def pull(self, keys):
        for k in keys:                         # 第一次见到的 key 现建（真实系统这里有准入）
            if k not in self.w:
                self.w[k], self.acc[k] = torch.randn(self.d) * 0.01, 0.0
        return torch.stack([self.w[k] for k in keys])
    def push(self, keys, grads):               # 优化器在 server 上跑，worker 只发梯度
        for k, g in zip(keys, grads):
            self.acc[k] += g.pow(2).mean().item()
            self.w[k] -= self.lr * g / (self.acc[k] ** 0.5 + 1e-8)

class PS:
    def __init__(self, n_shards, d):
        self.shards = [Shard(d) for _ in range(n_shards)]
    def _route(self, keys):                    # key -> shard：按取模
        return [[k for k in keys if k % len(self.shards) == s] for s in range(len(self.shards))]
    def pull(self, keys):
        parts = self._route(keys)
        rows = {k: r for p, sh in zip(parts, self.shards) if p for k, r in zip(p, sh.pull(p))}
        return torch.stack([rows[k] for k in keys])
    def push(self, keys, grads):
        g = dict(zip(keys, grads))
        for p, sh in zip(self._route(keys), self.shards):
            if p:
                sh.push(p, torch.stack([g[k] for k in p]))

def worker_step(ps, ids, offsets):
    uniq, inv = torch.unique(ids, return_inverse=True)        # batch 内去重，只拉不同的行
    rows = ps.pull(uniq.tolist()).requires_grad_(True)        # [U, d]
    B = offsets.numel() - 1
    bag_of = torch.repeat_interleave(torch.arange(B), offsets[1:] - offsets[:-1])
    pooled = torch.zeros(B, rows.shape[1]).index_add(0, bag_of, rows[inv])
    loss = pooled.pow(2).sum()                                 # 代替真实模型
    loss.backward()                                            # rows.grad 已按行合并好重复 ID
    ps.push(uniq.tolist(), rows.grad)
    return len(ids), len(uniq)

torch.manual_seed(0)
ps = PS(n_shards=4, d=8)
ids = torch.randint(0, 50, (200,))                             # 200 次查表，ID 空间小，重复多
offsets = torch.arange(0, 201, 10)                             # 20 条样本，每条 10 个 ID
print("查表次数 / 去重后拉取行数:", worker_step(ps, ids, offsets))
print("各 shard 行数:", [len(s.w) for s in ps.shards])
# 查表次数 / 去重后拉取行数: (200, 47)
# 各 shard 行数: [12, 12, 11, 12]
```

关键两行：`torch.unique(..., return_inverse=True)` 去重并记住每次查表对应哪一行；`rows[inv]` 的反向会自动把重复 ID 的梯度加到同一行上，push 的就是合并后的梯度。常见变体：多个 worker 线程共享一个 PS 模拟异步、给 shard 加版本号统计 staleness、push 前把梯度转 fp16 省一半流量。

## 参考

- [Scaling Distributed Machine Learning with the Parameter Server（Li et al. OSDI 2014）](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu)：server / worker、key 分片、有界延迟一致性
- [PERSIA（arXiv 2111.05897）](https://arxiv.org/abs/2111.05897)：embedding 异步 + dense 同步、99.99% 参数在 embedding、AUC 对比表 2、相对 XDL 的加速、Google Cloud 集群配置
- [Software-Hardware Co-design for Fast and Scalable Training of DLRMs（Neo / ZionEX，arXiv 2104.05158）](https://arxiv.org/abs/2104.05158)：CPU 异步 PS（45 PS + 15 trainer）到 GPU 同步的 40 倍、reader 层
- [Monolith（arXiv 2209.07663）](https://arxiv.org/abs/2209.07663)：worker-PS 架构、PS 每日快照与故障率的账
- [XDL（alibaba/x-deeplearning）](https://github.com/alibaba/x-deeplearning)：阿里的稀疏大模型训练框架（DLP-KDD 2019），PERSIA 的对比基线
