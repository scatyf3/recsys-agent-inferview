---
title: 排序服务：取特征、batching、CPU vs GPU、embedding 服务、缓存与降级
status: draft
tags: [ranking-serving, feature-store, batching, embedding-server, cache, degradation]
difficulty: 3
order: 11
related: [/recsys/funnel-overview, /recsys/long-sequence, /recsys/embedding-lookup, /recsys/embedding-sharding, /recsys/sample-pipeline, /recsys/online-learning]
recsys: [rs-s-feature, rs-s-infer, rs-s-cache]
---

# 排序服务：取特征、batching、CPU vs GPU、embedding 服务、缓存与降级

> 精排一个请求的 dense 计算在 GPU 上不到 1 ms，时间都花在取特征、查 embedding 和网络往返上；服务设计就是把这些搬运压下去

## 一句话结论

精排一个请求要给 $m$ 个候选打分（本页假设 $m = 500$）：用户侧特征取一次，物品侧和交叉特征各取 $m$ 次，每请求约 280 KB，$10^5$ QPS 就是 28 GB/s，所以物品特征要放本地缓存、按分片批量读，并把线上用到的特征值落日志防止训练 / 服务不一致。计算分两类：embedding 查表的算术强度只有 0.25 FLOPs/字节（Facebook 生产数据），是访存操作；dense MLP 和 attention 才是 GPU 的活。GPU 端到端推理里数据搬运占 60–80% 时间，batch 小时赢不了 CPU（DeepRecSys），所以要让请求内的候选共享用户侧计算，再跨请求攒 batch，用几毫秒排队换吞吐（DIN 的 request batching 等优化让单机 QPS 翻倍）。TB 级的 embedding 拆成独立的分片服务，dense 部分在 GPU 上多副本；热门 embedding 缓存在 GPU / CPU 里，HugeCTR HPS 的三级缓存在命中率够高时对未命中的 key 先返回默认值、后台异步补上。过载时按事先排好的顺序降级：特征超时用默认值 → 减候选数 → 关掉长序列等重模块 → 退到缓存 / 热门结果 → 限流。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $m$ | 精排一个请求的候选数 | 500（假设） |
| $F_i,\ F_c$ | 每个候选的物品特征、交叉特征字段数 | 50、20（假设） |
| $s$ | 每个特征值的字节数 | 8 |
| $Q$ | 峰值 QPS | $10^5$（假设） |
| $c_u$ | 用户侧计算（用户塔、序列编码）的 FLOPs | $10^9$（假设） |
| $c_i$ | 每个候选的 dense 计算 FLOPs | $2\times10^7$（假设） |
| $F_{\text{eff}}$ | 一张卡的有效算力 | $1.1\times10^{13}$ FLOP/s（见 [搜推全景](/recsys/funnel-overview)） |
| $n_s$ | 每个候选的稀疏 id 数 | 20（假设） |
| $d$ | embedding 维度 | 32 |
| $h$ | 缓存命中率 | 0.95（假设） |
| $\lambda$ | 一个实例收到的请求速率，次 / 秒 | 2000（假设） |
| $b$ | 跨请求攒的 batch（按请求数） | 8 |

KB = $10^3$ 字节，GB = $10^9$ 字节。

### 1. 一个精排请求的时间线

```
粗排结果（user_id, 500 个 item_id）
  │
  ├─ 取特征 ── 用户特征 + 行为序列（1 次）
  │          ├ 物品特征（500 次，本地缓存优先）
  │          └ 交叉特征（500 次，用户×物品 / 用户×作者的统计）
  ├─ 组装张量（稀疏 id 列表 + dense 值）
  ├─ embedding 查表（本地缓存 → embedding 服务）
  ├─ dense 前向（GPU）：用户侧一次，候选侧 500 个一个 batch
  ├─ 多目标分数（点击、时长、点赞……）按公式融合
  └─ 返回 top-50 给重排
```

按 [搜推全景](/recsys/funnel-overview) 的预算，精排有 40 ms：取特征约 10 ms、推理约 30 ms（假设）。先看 dense 计算本身要多久：

$$
t_{\text{dense}} = \frac{c_u + m\, c_i}{F_{\text{eff}}} = \frac{10^9 + 500 \times 2\times10^7}{1.1\times10^{13}} \approx 1\ \text{ms}
$$

只有 1 ms。剩下的时间去了哪：取特征的网络往返、查 embedding、CPU 上的特征处理和张量组装、CPU 到 GPU 的拷贝、kernel 启动。OneRec 报告里，快手原来的级联系统超过 50% 的服务资源花在通信和存储上，排序模型推理 MFU 只有 11.2%。下面几节就是在压这些。

### 2. 在线特征获取

**量**：

$$
\text{每请求字节} = (F_u + m\,(F_i + F_c)) \cdot s
$$

1. 物品侧：$500 \times 50 \times 8 = 200$ KB；交叉：$500 \times 20 \times 8 = 80$ KB；用户侧几 KB 忽略。
2. 合计约 280 KB / 请求，$Q = 10^5$ 时 28 GB/s。

**怎么压**：

1. **物品特征放本地缓存**。物品在请求之间共享，曝光服从幂律，少量热门物品占了大部分请求。命中率 90% 时物品侧远程只剩 20 KB。代价是过期：物品的统计特征（近一小时 CTR）要设 TTL。
2. **交叉特征没法跨用户共享**，按用户 id 路由，让一个用户的交叉特征落在同一个分片，一次批量读拿完。
3. **批量读，少扇出**：一个请求如果要去 20 个分片取特征，就要等最慢的那个分片，尾延迟按 $1 - (1-p)^k$ 放大（见 [搜推全景](/recsys/funnel-overview)）。
4. **超时给默认值**：特征服务超时不等，填默认值继续算。模型训练时要见过缺失值（训练样本里也随机置空一部分），否则默认值会让分数乱跳。

**训练 / 服务一致性**：线上取到的特征和离线拼样本时算的特征只要有一点不同（时间窗口、过期、默认值），模型就学到了线上看不到的东西。最稳的做法是**把线上实际用到的特征值落日志**（feature snapshot），训练直接用这份日志和后来的标签拼样本，见 [样本管线](/recsys/sample-pipeline)。

### 3. CPU 还是 GPU

两类计算的性质完全不同（[Gupta et al., HPCA 2020](https://arxiv.org/abs/1906.03109)，Facebook 生产数据）：

| 算子 | 算术强度（FLOPs / 字节） | LLC 未命中（MPKI） |
|---|---|---|
| embedding 查表（SparseLengthsSum） | 0.25 | 8 |
| RNN | 5.5 | 0.5 |
| FC | 18 | 0.2 |
| CNN | 141 | 0.06 |

1. embedding 查表是随机访存加一点加法，瓶颈是内存带宽和延迟，CPU 的大内存更合适，或者交给专门的 embedding 服务。
2. dense MLP、attention 是矩阵乘，适合 GPU，但要 batch 够大。
3. 推荐模型占 Facebook 数据中心 AI 推理周期的 79%（同一篇论文），所以这件事值得专门优化。

**GPU 什么时候赢**（[DeepRecSys](https://arxiv.org/abs/2001.02772)）：GPU 上推理时，把输入从 CPU 搬到 GPU 平均占端到端时间的 60–80%；batch 小时 GPU 比 CPU 还慢，batch 大到一定程度（因模型而异）才反超。DeepRecSys 的调度器按请求大小切分：小请求留在 CPU，大请求送 GPU，在严格尾延迟下吞吐翻倍，在生产环境把延迟降了 30% 以上。

**GPU 上的小 kernel 问题**（[COLD](https://arxiv.org/abs/2007.16122)）：粗排模型换成 fp16 后，kernel 本身很快，**kernel 启动**变成瓶颈；用 MPS 让多个进程共享 GPU、减少启动开销，fp16 + MPS 吞吐翻倍。同一系统里三种粗排模型的对比：

| 模型 | QPS | 延迟 |
|---|---|---|
| 向量内积（双塔） | 60000+ | 2 ms |
| COLD（轻量 DNN） | 6700 | 9.3 ms |
| DIEN（精排级） | 629 | 16.9 ms |

### 4. batching：请求内共享，请求间攒批

**4.1 请求内：用户侧只算一次**。一个请求的 $m$ 个候选天然是一个 batch，而且共享同一个用户：

$$
\text{朴素} = m\,(c_u + c_i),\qquad \text{共享} = c_u + m\, c_i
$$

$c_u = 10^9$（用户塔加行为序列编码）、$c_i = 2\times10^7$、$m = 500$：朴素 $5.1\times10^{11}$，共享 $1.1\times10^{10}$，省 46 倍。做法是用户侧子图跑一次、输出在 batch 维上广播给所有候选。生成式排序里 M-FALCON 把这件事推到注意力层，见 [生成式推荐](/recsys/generative-rec)。

**4.2 请求间：攒批换吞吐**。单个请求的 batch（500 个候选）对 GPU 往往还不够大，就把相邻的几个请求拼成一次前向。代价是排队：请求以速率 $\lambda$ 到达，凑满 $b$ 个请求，第一个请求要等

$$
t_{\text{wait,max}} = \frac{b - 1}{\lambda},\qquad t_{\text{wait,avg}} \approx \frac{b - 1}{2\lambda}
$$

$\lambda = 2000$、$b = 8$：第一个请求等 3.5 ms，平均 1.75 ms。所以攒批总是带一个**最长等待**（比如 2 ms），到点就发，不管凑没凑满。DIN 的线上服务用了 request batching、显存访问优化和多 kernel 并发，单机 QPS 翻倍。

### 5. embedding 服务拆分

**为什么要拆**：稀疏参数是 TB 级的。HSTU 论文的例子：词表 100 亿、512 维，光 fp32 权重就是 $10^{10} \times 512 \times 4 = 20.5$ TB（训练时再加优化器状态是 60 TB），fp16 也有 10 TB，单机放不下。而 dense 部分通常只有几 GB。

所以拆成两种服务：

1. **embedding 服务**：按 id 哈希分片，放 CPU 内存（冷的放 SSD），只做查表，多副本扛 QPS。
2. **dense 推理服务**：GPU 上跑 MLP / attention，每个实例一份完整的 dense 参数，按 QPS 水平扩。

**查表的量**：

1. 物品侧：$m \cdot n_s = 500 \times 20 = 10^4$ 次，每行 $d = 32$ 维 fp16 = 64 字节，640 KB。
2. 用户侧：行为序列 1000 个 id，64 KB。
3. 合计约 0.7 MB / 请求，$10^5$ QPS 就是 70 GB/s，不能全走网络。

**多级缓存**：物品 id 的访问服从幂律，在 dense 服务本地（GPU 显存或 CPU 内存）缓存热门行。命中率 $h = 0.95$ 时物品侧远程只剩 $640 \times 0.05 = 32$ KB，加上用户侧 64 KB，约 96 KB / 请求，集群合计 9.6 GB/s。再加两条：

1. 一个 batch 内先对 id 去重再查。
2. 行存量化（fp16 → int8）再减半，见 [embedding 表](/recsys/embedding-tables)。

**HugeCTR HPS 的做法**（[arXiv 2210.08804](https://arxiv.org/abs/2210.08804)）：三级存储，第 1 级 GPU embedding 缓存放热 key，第 2 级 CPU 内存（可以是 Redis 一类的分布式内存库）放部分参数，第 3 级每个节点本地 SSD 上的 RocksDB 存一份完整副本做容错。缓存未命中时有两种模式：命中率高于阈值时，未命中的 key **先返回默认 embedding**、后台异步把真值拉进缓存，不让一次未命中拖慢请求；命中率低（比如刚启动的预热期）时同步拉取。报告相对 CPU 基线端到端快 5–62 倍（随 batch 增大），吞吐 240 万–640 万样本 / 秒（batch 1024–131072，作者自测）。

**参数更新**：模型每几分钟更新一次时（快手 TWIN 是每 5 分钟同步一次参数），embedding 服务要能增量地接收新行、更新旧行，缓存要失效或刷新，见 [在线学习](/recsys/online-learning)。

### 6. 缓存什么

| 缓存对象 | 为什么能缓存 | 代价 | 典型做法 |
|---|---|---|---|
| 物品特征 | 跨请求共享、幂律 | 统计特征过期 | 本地 LRU + TTL |
| 热门 embedding 行 | 同上 | 参数更新后要刷新 | GPU / CPU 多级缓存 |
| 物品侧预计算结果 | 只依赖物品 | 跟不上最新参数 | TWIN 每 15 分钟刷一遍投影 |
| 用户特征 / 序列 | 一个 session 内连续请求 | 新行为进不来 | 短 TTL，新行为追加 |
| 下一屏的候选 | 用户会继续下滑 | 推荐变得不新鲜 | 一次多排一些，后续请求先用 |
| 搜索结果 | 热门 query 高度重复 | 新内容进不来 | 按 query 缓存，短 TTL |

推荐几乎不能缓存最终结果（每个用户不一样），能缓存的是**被很多请求共享的中间量**。

### 7. 降级

降级是事先排好、可以一键打开的开关，按「对效果的伤害从小到大」排：

| 级别 | 动作 | 省什么 | 伤害 |
|---|---|---|---|
| 0 | 特征 / embedding 超时用默认值 | 尾延迟 | 个别请求分数变差 |
| 1 | 召回某一路超时就丢 | 尾延迟 | 候选少一路 |
| 2 | 减候选数（精排 500 → 200） | 算力线性减少 | 好候选可能被截掉 |
| 3 | 关掉重模块（长序列分支） | 大头算力 | 个性化变弱 |
| 4 | 换轻模型或直接用粗排分 | 几乎全部精排算力 | 明显 |
| 5 | 返回缓存结果 / 热门列表 | 全部 | 不个性化 |
| 6 | 限流，按优先级拒绝 | 保护系统 | 部分用户失败 |

要点：

1. 每一级都要有指标触发（排队长度、P99、错误率）和恢复条件，不能只靠人工。
2. 模型要在「缺特征、缺一路召回」的输入上训练过，降级时分数才不会乱。
3. 准入控制比排队重要：过载时让请求排队只会让所有人超时。

## 面试追问

::: details Q：精排的 dense 计算才 1 ms，为什么延迟预算要几十毫秒？
时间主要在搬运：取用户 / 物品 / 交叉特征的网络往返（扇出到多个分片要等最慢的）、查 embedding、CPU 上的特征处理和张量组装、CPU 到 GPU 的拷贝。DeepRecSys 测得 GPU 推理里数据搬运占端到端 60–80%；OneRec 报告级联系统一半以上的资源花在通信和存储上。
:::

::: details Q：排序推理放 CPU 还是 GPU？
看算子和 batch。embedding 查表算术强度 0.25 FLOPs/字节，是访存操作，放 CPU 或独立的 embedding 服务；dense MLP / attention 放 GPU，但 batch 小时搬运开销盖过计算，GPU 反而慢。所以请求内共享用户侧计算、请求间攒批，必要时按请求大小分流（DeepRecSys）。
:::

::: details Q：跨请求攒 batch 会增加多少延迟？怎么控制？
到达速率 $\lambda$、凑 $b$ 个请求，第一个请求最多等 $(b-1)/\lambda$，平均约一半。$\lambda = 2000$、$b = 8$ 时最多 3.5 ms。用「凑满 $b$ 或等满 max_wait 就发」控制上界，max_wait 从延迟预算里切出来。
:::

::: details Q：为什么要把 embedding 拆成独立服务？
稀疏参数 TB 级（100 亿 × 512 维 fp32 是 20 TB），单机放不下，而 dense 部分只有几 GB。拆开后 embedding 服务按 id 分片、放大内存机器，dense 服务放 GPU、按 QPS 水平扩，两边独立扩缩容。代价是每请求多一次网络查表，用本地多级缓存（命中率 95% 时远程流量降到 1/20）和 batch 内去重来压。
:::

::: details Q：embedding 缓存没命中怎么办？
同步拉会把这个请求拖慢。HPS 的做法是命中率高于阈值时先返回默认 embedding、后台异步拉真值进缓存，下次就命中了；命中率低（预热期）时才同步拉。前提是模型对少量 id 用默认值不敏感。
:::

::: details Q：训练 / 服务不一致（skew）怎么防？
最稳的是把线上实际用到的特征值落日志，训练用这份日志加后到的标签拼样本，而不是离线重新算一遍特征。另外监控线上特征分布和训练集分布的差异、默认值的占比。
:::

::: details Q：过载时你会怎么降级？
按伤害从小到大：超时用默认值 → 丢超时的召回路 → 精排候选数减半 → 关长序列这类重模块 → 换轻模型 → 返回缓存 / 热门 → 按优先级限流。每级有自动触发和恢复的指标，模型要在缺失输入上训练过。
:::

## 手撕

跨请求攒批（dynamic batching）：凑够 `max_batch` 个候选或等满 `max_wait` 就发一次推理，再把分数按请求切回去：

```python
import asyncio, time

class Batcher:
    """攒批：候选数凑到 max_batch（可能略超），或最早的请求等了 max_wait 秒，就发一次推理。"""
    def __init__(self, infer, max_batch=2048, max_wait=0.002):
        self.infer, self.max_batch, self.max_wait = infer, max_batch, max_wait
        self.q = asyncio.Queue()

    async def score(self, cands):                  # 一个请求：一组候选
        fut = asyncio.get_running_loop().create_future()
        await self.q.put((cands, fut))
        return await fut

    async def run(self):
        while True:
            batch = [await self.q.get()]           # 阻塞等第一个请求
            n, deadline = len(batch[0][0]), time.monotonic() + self.max_wait
            while n < self.max_batch:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.q.get(), left)
                except asyncio.TimeoutError:
                    break
                batch.append(item)
                n += len(item[0])
            flat = [c for cands, _ in batch for c in cands]
            scores = self.infer(flat)              # 一次前向打完所有请求的候选
            i = 0
            for cands, fut in batch:               # 按请求切回去
                fut.set_result(scores[i:i + len(cands)])
                i += len(cands)

async def main():
    calls = []
    def infer(xs):
        calls.append(len(xs))
        return [x * 0.1 for x in xs]
    b = Batcher(infer, max_batch=1000, max_wait=0.005)
    task = asyncio.create_task(b.run())
    res = await asyncio.gather(*(b.score(list(range(300))) for _ in range(8)))
    print(calls, len(res), res[0][:3])
    task.cancel()

asyncio.run(main())
# [1200, 1200] 8 [0.0, 0.1, 0.2]
```

8 个请求、每个 300 个候选，拼成两次前向（每次 4 个请求、1200 个候选）。常见变体：按请求的 deadline 而不是固定 max_wait 决定何时发；大请求单独走 GPU、小请求留在 CPU（DeepRecSys 的思路）。

## 参考

- [The Architectural Implications of Facebook's DNN-based Personalized Recommendation](https://arxiv.org/abs/1906.03109)（Gupta et al., HPCA 2020）：推荐占 79% AI 推理周期、SLS 算术强度 0.25 FLOPs/B、LLC 未命中
- [DeepRecSys](https://arxiv.org/abs/2001.02772)：GPU 推理里数据搬运占 60–80%、按请求大小分流 CPU / GPU、吞吐翻倍、各模型 SLA
- [COLD](https://arxiv.org/abs/2007.16122)：fp16 后 kernel 启动成瓶颈、MPS、三种粗排模型的 QPS / 延迟
- [Deep Interest Network (DIN)](https://arxiv.org/abs/1706.06978)：request batching 等优化让单机 QPS 翻倍
- [HugeCTR Hierarchical Parameter Server](https://arxiv.org/abs/2210.08804)：GPU / CPU / SSD 三级、默认值 + 异步插入、5–62 倍
- [HSTU](https://arxiv.org/abs/2402.17152)：100 亿 × 512 维词表的内存账
- [TWIN](https://arxiv.org/abs/2302.02352)：参数每 5 分钟同步、物品侧投影 15 分钟刷新
- [OneRec Technical Report](https://arxiv.org/abs/2506.13695)：级联系统的通信 / 存储占比与 MFU
- [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)：扇出与尾延迟
