---
title: 样本与特征管线：样本拼接、延迟标签、负采样、训练-服务一致性、流式管线
status: draft
tags: [sample-join, delayed-feedback, negative-sampling, calibration, feature-store, streaming]
difficulty: 3
order: 7
related: []
recsys: [rs-d-join, rs-d-neg, rs-d-store, rs-d-stream]
---

# 样本与特征管线：样本拼接、延迟标签、负采样、训练-服务一致性、流式管线

> 一条训练样本 = 曝光那一刻的特征 + 之后才到的标签；拼接窗口、负样本怎么采、采完怎么把概率改回来、特征是不是线上那一份，每一步都会悄悄改变模型学到的分布

## 一句话结论

样本由两条流拼成：曝光时记下的特征（按 request ID 索引）和之后陆续到达的用户行为。拼接要等一个窗口 $W$，缓存量是 $\lambda f W$：每秒 10 万次曝光、每条 5 KB、等 30 分钟，就是 900 GB，所以近的放内存、远的落盘（Monolith 的 joiner 就这么做）。点击几分钟内就到，窗口能等齐；转化可能晚几天，按指数延迟、均值 1 天算，等 1 小时只能收到 4% 的正样本，于是先当负样本训、正样本到了再补一条（Twitter 的 fake negative，校正 $p = b/(1-b)$）。负样本太多时按比例 $w$ 下采样，预测值要校正回 $q = p / (p + (1-p)/w)$。召回的 in-batch 负样本按流行度被采到，要减去 $\log p_j$（logQ 修正）。特征必须用线上打分时的那一份（在服务时把特征记进曝光日志），否则训练和服务看到的不是同一个分布。整条链路用 Kafka + Flink 拼成流，在线训练直接消费，批训练从落盘的数据读。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $\lambda$ | 曝光速率 | $10^5$ 次/秒（假设） |
| $f$ | 一条曝光的特征字节数 | 5 KB（假设） |
| $W$ | 拼接窗口：曝光后最多等多久的标签 | 30 分钟 |
| $\mu$ | 标签延迟的均值 | 点击约分钟级，转化约 1 天（假设） |
| $p$ | 真实正样本率，或模型在采样后分布上的预测 | 1% |
| $w$ | 负样本保留比例（下采样率） | 0.1 |
| $q$ | 校正回原分布的预测 | |
| $b$ | fake negative 分布下的正样本概率 | |
| $p_j$ | 召回训练里 item $j$ 被采进一个 batch 的概率 | |

KB = $10^3$、GB = $10^9$、TB = $10^{12}$ 字节。

### 1. 样本拼接

**为什么要拼**。打分时只有特征，没有标签；标签（点击、点赞、转化）要等用户行动。两件事发生在不同时间、不同服务，各写各的日志：

1. **曝光流**：一次请求的每个候选、打分时用到的特征，带上 request ID。Facebook 广告的做法是在排序服务里把「这个广告 + 排序它时用的特征」写进曝光流（[He et al. ADKDD 2014](https://quinonero.net/Publications/predicting-clicks-facebook.pdf) §4）。
2. **行为流**：点击、点赞、转化，也带 request ID（或曝光 ID）。
3. **joiner**：按 request ID 做流对流的 join。窗口内等到了行为就是正样本，窗口到期还没等到就记为负样本。

**缓存要多大**。窗口里所有还没结案的曝光都要留着：

$$
M_{\text{join}} = \lambda \cdot f \cdot W
$$

$10^5 \times 5\ \text{KB} \times 1800\ \text{s} = 9\times10^{11}$ B = 900 GB（假设的数字）。[Monolith](https://arxiv.org/abs/2209.07663) 的 joiner：最近的特征在内存缓存里，等得久的落到磁盘 KV；行为来了先查内存、没命中再查磁盘（§2.2.2）。行为流和特征流到达的顺序没有保证，所以每个请求要有唯一 key。

**窗口取多长**。窗口越长，标签越全，但缓存越大、样本越晚进训练。Facebook 的经验是用内存可接受的窗口就能把「本该是正样本却被记成负样本」的偏差压到零点几个百分点，而且这个偏差可以测出来再修正。

### 2. 延迟标签

**能等到多少**。设标签延迟服从均值 $\mu$ 的指数分布（[Chapelle KDD 2014](https://doi.org/10.1145/2623330.2623634) 对转化延迟的建模假设），窗口 $W$ 内收到的正样本比例：

$$
F(W) = 1 - e^{-W/\mu}
$$

| 标签 | $\mu$（假设） | $W$ | $F(W)$ |
|---|---|---|---|
| 点击 | 2 分钟 | 5 分钟 | $1 - e^{-2.5} = 91.8\%$ |
| 点击 | 2 分钟 | 30 分钟 | $\approx 100\%$ |
| 转化 | 1 天 | 1 小时 | $1 - e^{-1/24} = 4.1\%$ |
| 转化 | 1 天 | 1 天 | 63.2% |
| 转化 | 1 天 | 7 天 | 99.9% |

点击可以靠窗口等齐；转化要么等好几天（模型就不新鲜了），要么先训再改。

**Fake negative**（[Ktena et al. arXiv 1907.06558](https://arxiv.org/abs/1907.06558)，Twitter 广告）：

1. 每次曝光立刻作为负样本进训练。
2. 之后用户真的互动了，再复制一条同样的特征、标为正样本进训练。
3. 于是每个正样本在数据里出现两次（一次负、一次正），负样本一次。模型学到的是偏了的分布：

$$
b = \frac{p}{1 + p}, \qquad p = \frac{b}{1 - b}
$$

推导：真实正样本率 $p$ 的一批曝光，进训练的样本数是 $1 + p$（全部曝光各一条负样本，加上 $p$ 条补的正样本），其中正样本 $p$ 条。

例：$p = 2\%$，模型学到 $b = 0.02/1.02 = 1.96\%$，线上用 $b/(1-b) = 2\%$ 改回来。论文还给了直接在 loss 里做重要性加权的版本（FN weighted）。Twitter 的线上实验里模型每 10 分钟产出一版（§4.2.3）。

### 3. 负采样与校正

**排序：负样本下采样**。CTR 1% 时 99% 的样本是负的，训练时间大头花在信息量少的负样本上。保留全部正样本、负样本只留比例 $w$：

1. 训练集的正样本率变成 $\dfrac{p}{p + (1-p)w}$。CTR 1%、$w = 0.1$：$0.01 / (0.01 + 0.099) = 9.2\%$。
2. 模型学到的是这个分布，线上要校正回去（He et al. §6.3）：

$$
q = \frac{p}{p + (1 - p)/w}
$$

$p$ 是模型的输出。代入 $p = 0.0917$、$w = 0.1$：$q = 0.0917 / (0.0917 + 9.083) = 1.0\%$。

等价的写法是 log-odds 平移：$\text{logit}(q) = \text{logit}(p) + \ln w$（负样本少了 $w$ 倍，赔率就高估了 $1/w$ 倍）。Monolith 在服务时做的就是这个 log odds 校正。AUC 只看排序，不受影响；NE 和校准度（预测均值 / 实际点击率）受影响，所以出价、多目标融合这类要用绝对概率的地方必须校正。

**召回：in-batch 负样本与 logQ 修正**（[Yi et al. RecSys 2019](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)，YouTube）。双塔召回常用同一个 batch 里别人的正样本当自己的负样本（召回的结构见 [ANN 与召回](/recsys/ann-retrieval)）。问题是热门 item 在 batch 里出现得多，被当成负样本的次数也多，被压得过狠。修正是在 softmax 前减去它被采到的概率的对数：

$$
s^c(x_i, y_j) = s(x_i, y_j) - \log p_j
$$

$p_j$ 不知道，从流里估：对每个 item 记下上次出现的步数，用滑动平均估两次出现之间的间隔 $\delta_j$，$p_j \approx 1/\delta_j$（论文的流式频率估计，数组按 item ID 哈希索引，不需要固定词表）。

例：热门 item 每 10 步出现一次，$p = 0.1$，$-\log p = 2.30$；冷门 item 每 $10^4$ 步一次，$-\log p = 9.21$。修正后冷门 item 的 logit 相对热门的多加 6.9。

### 4. 训练-服务一致性

**不一致从哪来**：

1. **两套代码**：线上用 C++ 实时算特征，离线用 SQL / Spark 重算，逻辑、默认值、截断不一样。
2. **时间穿越**：离线回填时用了样本时间之后的数据（比如「这个商品过去 7 天的点击率」用的是今天的值），训练时偷看了未来，线上拿不到，离线指标虚高。
3. **新鲜度不同**：线上特征有分钟级延迟，离线算的是准点值。

**做法**：

1. **服务时记特征**：打分时把实际用到的特征值写进曝光日志，训练直接用这份。训练看到的就是线上看到的，一致性最好；代价是日志量大（下一节的账），而且新加的特征没有历史日志。
2. **point-in-time join**：新特征要回填历史时，每条样本只取「时间戳 ≤ 请求时间」的最新特征值。
3. **feature store**：同一个特征定义同时产出离线表（带时间戳的历史，供训练回填）和在线 KV（最新值，供服务查），共用一份转换代码。
4. **监控**：对比线上记录的特征分布和离线重算的分布（缺失率、均值、分位数），差异超过阈值报警。

### 5. 流式管线

Monolith 的流式引擎（§2.2.1，图 4）：

1. 两个 Kafka 队列：一个记用户行为，一个记特征（服务时写入）。
2. 一个 Flink 作业做 online joiner，拼出训练样本，并在这里做负采样，写进第三个 Kafka 队列。
3. **在线训练**直接从这个队列读；**批训练**由一个转储作业先写到 HDFS，攒够了再读。同一份样本两条路共用。

**数据量的账**（假设）：每秒 $10^5$ 次曝光，每天 $8.64\times10^9$ 条，每条 5 KB，每天 43 TB 原始样本。负样本保留 10%、CTR 1% 时，进训练的是 $10^5 \times (0.01 + 0.99 \times 0.1) = 1.09\times10^4$ 条/秒，只有曝光量的 11%；在线训练的吞吐要长期高于这个速率，否则积压越来越多，模型越来越旧。

**工程问题**：

1. **去重与恰好一次**：joiner 重启后可能重放，同一个 request ID 的样本要去重，否则正样本被算两次。
2. **迟到数据**：超过窗口才到的行为，要么丢掉，要么按 fake negative 的方式补一条正样本。
3. **回放**：Kafka 留几天的数据，模型出问题时可以回到某个 offset 重新训（见 [在线学习与模型更新](/recsys/online-learning)）。

## 面试追问

::: details Q：样本拼接的窗口怎么定？
看标签延迟的分布和内存预算。缓存量是曝光速率 × 每条特征大小 × 窗口；收到的正样本比例按指数延迟是 $1 - e^{-W/\mu}$。点击几分钟就到，窗口 30 分钟基本收齐；转化延迟以天计，靠窗口等就太晚，改用 fake negative 或专门的延迟反馈模型。
:::

::: details Q：负样本下采样以后，预测值为什么要校正？怎么校正？
模型学的是采样后的分布，正样本率被抬高了：CTR 1%、保留 10% 负样本，训练集正样本率 9.2%。用 $q = p / (p + (1-p)/w)$ 改回来，等价于 logit 加 $\ln w$。只看排序（AUC）不用改，出价和多目标融合要用绝对概率，必须改。
:::

::: details Q：fake negative 为什么是 $b = p/(1+p)$？
每次曝光都先进一条负样本，真正互动的再补一条正样本。真实正样本率为 $p$ 时，每 1 次曝光对应 $1 + p$ 条训练样本，其中 $p$ 条正样本，所以 $b = p/(1+p)$，反过来 $p = b/(1-b)$。
:::

::: details Q：in-batch 负样本有什么偏差？logQ 修正做了什么？
batch 里的 item 是按流行度出现的，热门 item 被当成负样本的次数多，被压得过狠，召回结果偏向冷门。修正是 softmax 前给每个 item 的分数减去 $\log p_j$，$p_j$ 是它被采到的概率，用流式频率估计（记录两次出现的间隔）得到。
:::

::: details Q：怎么保证训练和服务的特征一致？
最可靠的是服务时把实际用到的特征写进曝光日志，训练直接用。要回填新特征时用 point-in-time join，只取请求时刻之前的值。用 feature store 让离线、在线共用一份特征定义，再监控两边的分布差异。
:::

::: details Q：在线训练的吞吐要多少才够？
至少要长期大于样本产生的速率。按每秒 10 万曝光、CTR 1%、负样本保留 10% 算，约 1.1 万条/秒；跟不上就会积压，模型越来越旧，在线训练就失去意义。
:::

## 手撕

流式频率估计 + 带 logQ 修正的 in-batch softmax：

```python
import torch, torch.nn.functional as F

class FreqEstimator:
    """流式估计 item 被采到的概率：记下上次出现的步数，估计两次出现的平均间隔 delta，p = 1 / delta"""
    def __init__(self, H=2**20, alpha=0.01):
        self.A = torch.zeros(H)             # A[h]：上次出现在哪一步
        self.B = torch.ones(H)              # B[h]：两次出现之间的平均间隔
        self.H, self.alpha = H, alpha
    def update(self, item_ids, step):
        h = torch.unique(item_ids % self.H)  # 按哈希索引，不需要固定词表
        self.B[h] = (1 - self.alpha) * self.B[h] + self.alpha * (step - self.A[h])
        self.A[h] = step
    def prob(self, item_ids):
        return 1.0 / self.B[item_ids % self.H]

def in_batch_softmax_logq(u, v, item_ids, est):
    """u, v: [B, d] 用户 / item 向量；第 i 行的正样本是第 i 个 item，其余 item 当负样本"""
    logits = u @ v.T                                          # [B, B]
    logits = logits - torch.log(est.prob(item_ids))[None, :]  # logQ 修正：每列减去该 item 的 log p
    return F.cross_entropy(logits, torch.arange(len(u)))

torch.manual_seed(0)
est = FreqEstimator()
n_items, Bsz = 1000, 256
pop = 1.0 / torch.arange(1, n_items + 1)                       # Zipf 流行度
for step in range(1, 3001):
    batch = torch.multinomial(pop, Bsz, replacement=True)      # 每步按流行度采一个 batch
    est.update(batch, step)
true_p = 1 - (1 - pop / pop.sum()) ** Bsz                      # 一个 batch 里至少出现一次的概率
for i in [0, 9, 99]:
    print(f"item {i:3d}: 估计 {est.prob(torch.tensor([i])).item():.4f}  真实 {true_p[i]:.4f}")
u, v = torch.randn(Bsz, 16), torch.randn(Bsz, 16)
loss = in_batch_softmax_logq(u, v, batch, est)                 # 随机向量，只演示调用
# item   0: 估计 1.0000  真实 1.0000
# item   9: 估计 0.9485  真实 0.9680
# item  99: 估计 0.3040  真实 0.2898
```

估计值和真实的「每个 batch 至少出现一次」的概率接近；$\alpha$ 越大跟得越快、噪声越大。常见变体：多个哈希函数取最大间隔以减少冲突的影响；把负样本下采样和 $q = p/(p + (1-p)/w)$ 校正写成函数，用模拟数据验证校准度回到 1。

## 参考

- [Practical Lessons from Predicting Clicks on Ads at Facebook（He et al. ADKDD 2014）](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)：online joiner、窗口与偏差、负样本下采样与校正公式、NE
- [Monolith（arXiv 2209.07663）](https://arxiv.org/abs/2209.07663)：Kafka + Flink 的流式引擎、内存 + 磁盘 KV 的 joiner、负采样与 log odds 校正
- [Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR Prediction（Ktena et al. arXiv 1907.06558）](https://arxiv.org/abs/1907.06558)：fake negative weighted / calibration
- [Modeling Delayed Feedback in Display Advertising（Chapelle KDD 2014）](https://doi.org/10.1145/2623330.2623634)：延迟的指数分布建模
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations（Yi et al. RecSys 2019）](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)：in-batch softmax 的 logQ 修正、流式频率估计
