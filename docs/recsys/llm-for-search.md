---
title: LLM 用于搜推：生成特征、相关性精排、离线蒸馏与成本账
status: draft
tags: [llm, relevance, distillation, cross-encoder, embedding, cost]
difficulty: 3
order: 13
related: [/recsys/funnel-overview, /recsys/ranking-serving, /recsys/generative-rec, /recsys/ann-retrieval]
recsys: [rs-g-llm]
---

# LLM 用于搜推：生成特征、相关性精排、离线蒸馏与成本账

> LLM 的成本跟着「调用次数」走：放在线上是 QPS × 候选数次，放在离线是物品数或标签数次，差好几个数量级；所以 LLM 的能力通常是离线进来、蒸馏成小模型再上线

## 一句话结论

一次 cross-encoder 前向约 $2 N_{\text{LLM}} T$ FLOPs：8B 模型、256 个 token 是 $4.1\times10^{12}$。线上给每个搜索请求的 100 个候选打分，$10^4$ QPS 就要约 $4\times10^{18}$ FLOP/s，按 H100 有效 400 TFLOP/s 算要 1 万张卡，单个请求在一张卡上也要约 1 秒（本页推算）。同一个模型离线打 3000 万条标签只要约 85 GPU 小时，给每天新增的 4500 万个视频生成文本特征约 128 GPU 小时。所以主流做法是：LLM 离线生成 embedding、文本特征（caption、标签）和相关性标签，蒸馏到能在毫秒级服务的小模型。Pinterest 用微调的 Llama-3-8B 当相关性教师（5 档准确率 0.602，比 mBERT 高 12.5%），给 3000 万条日志打软标签（人工标注的 100 倍、覆盖 45 种以上语言），线上学生模型是吃 embedding 和 BM25 等特征的前馈网络，准确率从只用 30 万人工标签的 0.484 提到 0.548，上线后 nDCG@20 +2.18%，搜索满足率美国 +0.7%、非美国 +2.0%（作者自测）。RankGPT 的 listwise 重排要对 100 个候选串行调 9 次 LLM（窗口 20、步长 10），延迟以秒计，同样更适合当教师：蒸馏出的 435M 学生在 BEIR 上平均 nDCG 比 3B 的 monoT5 高 1.67。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $N_{\text{LLM}}$ | LLM 参数量 | $8\times10^9$（Llama-3-8B） |
| $T$ | 一次调用的输入 token 数（query + 文档文本） | 256（假设） |
| $C$ | 一次 cross-encoder 前向的 FLOPs，$\approx 2 N_{\text{LLM}} T$ | $4.1\times10^{12}$ |
| $Q$ | 搜索峰值 QPS | $10^4$（假设） |
| $m$ | 每个请求要 LLM 打分的候选数 | 100 |
| $F_{\text{eff}}$ | 一张 H100 的有效算力 | $4\times10^{14}$ FLOP/s（bf16 dense 峰值 989 TFLOP/s × 40%，假设） |
| $D$ | 离线要处理的条数（标签数、物品数） | $3\times10^7$ |
| $w,\ s$ | listwise 重排的窗口大小、步长 | 20、10 |
| $h$ | 缓存命中率 | 0.7（假设） |

### 1. 四种用法，成本跟着什么走

| 用法 | 在哪跑 | 成本正比于 | 例子 |
|---|---|---|---|
| 生成物品 embedding / 文本特征 | 离线 | 物品数（新增 + 更新） | 图片 caption、内容向量、semantic ID 的输入 |
| 打相关性 / 偏好标签，蒸馏 | 离线 | 标签条数 | Pinterest 相关性教师 |
| query 理解、改写 | 离线预算 + 在线缓存 | 去重后的 query 数 | 头部 query 预先算好 |
| 在线精排 / 重排 | 在线 | QPS × 候选数 | RankGPT、LLM cross-encoder |

前三种的成本和流量无关或被缓存挡住，第四种和流量成正比。下面把账算清楚。

### 2. 在线用 LLM 打分：算不起

一次 cross-encoder 前向（prefill，没有生成），按每 token $2N$ FLOPs（见 [prefill 的 roofline](https://scatyf3.github.io/ai-infra-inferview/inference/prefill-decode-roofline)）：

$$
C = 2 N_{\text{LLM}} T,\qquad \text{FLOP/s} = Q \cdot m \cdot C,\qquad \text{卡数} = \frac{Q m C}{F_{\text{eff}}}
$$

1. $C = 2 \times 8\times10^9 \times 256 = 4.1\times10^{12}$ FLOPs / 对。
2. $Q m = 10^4 \times 100 = 10^6$ 对 / 秒，$4.1\times10^{18}$ FLOP/s。
3. 卡数：$4.1\times10^{18} / 4\times10^{14} \approx 1.0\times10^4$ 张 H100。
4. 延迟：一个请求 $100 \times 4.1\times10^{12} = 4.1\times10^{14}$ FLOPs，一张卡约 1 s；要压到 30 ms，得把一个请求摊到 30 多张卡上。

换成 1 亿参数的小 cross-encoder（BERT-base 量级）：$C = 5.1\times10^{10}$，每秒 $10^6$ 对是 $5.1\times10^{16}$ FLOP/s，约 130 张卡，可以上线但仍不便宜。Pinterest 线上的学生模型更轻：前馈网络，输入是预先算好的 query / Pin embedding（SearchSAGE、PinSAGE、视觉向量）加上 BM25、历史互动率、词重叠这类匹配特征，算力可以忽略。

### 3. 离线用 LLM：便宜得多

**打标签**：

$$
\text{GPU 秒} = \frac{D \cdot C}{F_{\text{eff}}}
$$

Pinterest 的学生用了 3000 万条教师标签：$3\times10^7 \times 4.1\times10^{12} = 1.23\times10^{20}$ FLOPs，$/4\times10^{14} = 3.1\times10^5$ GPU 秒，约 85 GPU 小时（本页推算，按 8B、256 token）。一次性的，和线上流量无关。

**生成物品特征**：快手每天新发布约 4500 万个视频（TWIN 论文的统计）。每个视频用 8B 模型处理 256 个文本 token：$4.5\times10^7 \times 4.1\times10^{12} = 1.8\times10^{20}$ FLOPs / 天，约 128 GPU 小时 / 天，相当于 5–6 张卡一直跑。视频多模态输入的 token 数会多一个数量级以上，账要按实际 token 数重算。

对比一下：线上方案一天要 $4.1\times10^{18} \times 86400 = 3.5\times10^{23}$ FLOPs，离线生成特征一天 $1.8\times10^{20}$，差约 2000 倍。离线这一天的活，线上方案 44 秒就用完了。

### 4. 蒸馏：Pinterest 的相关性系统

([arXiv 2410.17152](https://arxiv.org/abs/2410.17152))

**教师**：cross-encoder，把 query 和 Pin 的文本拼起来送进语言模型，取句向量过几层全连接，输出 5 档相关性（L1 完全无关 … L5 高度相关），交叉熵训练。

1. Pin 的文本表示是拼出来的：标题和描述、BLIP 生成的图片 caption（人工抽检 1 万张，88% 质量好且相关）、链接页的标题和描述、用户收藏它的画板名、近两年在这个 Pin 上高质量互动的 query 词。逐项加入准确率一路涨（表 4）。
2. 模型越大越好：5 档准确率 SearchSAGE 基线 0.503、mBERT 0.535、Llama-3-8B 0.602。8B 用 qLoRA 在 8 张 40 GB A100 上微调。
3. 人工标注：训练集 28 万条；单个标注员准确率只有约 70%，所以用 3 人的软标签。

**学生**：线上实时服务的前馈网络（输入见第 2 节）。

1. 用教师给搜索日志里采样的 (query, Pin) 打 5 档软标签，分层采样保证各档平衡，数据量是人工标注的 100 倍以上，覆盖 45 种以上语言。
2. 学生准确率：只用 30 万人工标签 0.484；600 万教师标签 0.535；3000 万 0.548（表 5）。
3. 人工只标了美国英文数据，但多语言的教师把能力带到了其他国家：德国、法国、英国的 precision@8 也提升了。

**为什么蒸馏有效**：教师的能力没法直接上线，但它能把「贵的判断」变成大量便宜的标签。学生的容量远小于教师，靠的是：数据量大两个数量级、覆盖人工标注没有的语言和长尾 query、软标签比 70% 准的单人标签干净。

**线上结果**：nDCG@20 +2.18%（300 个 query 的人工评估）；搜索满足率美国 +0.7%、非美国 +2.0%（A/B，作者自测）。

### 5. listwise 重排：RankGPT

([arXiv 2304.09542](https://arxiv.org/abs/2304.09542)) 让 LLM 直接输出一组候选的排列（「[3] > [1] > [2] …」）。上下文放不下 100 个候选，就用滑窗从后往前排：

$$
\text{调用次数} = \frac{M - w}{s} + 1
$$

$M = 100$、$w = 20$、$s = 10$：$(100 - 20)/10 + 1 = 9$ 次，而且**串行**：每个窗口的前 $s$ 个要带进下一个窗口，后一次必须等前一次。一次调用的输入约 20 个候选 × 每个百来个 token，再生成一串排列，在 API 上是秒级，9 次串行就是十几秒，进不了搜索的在线链路。

它在离线很好用：

1. 效果：GPT-4 零样本在 TREC、BEIR 上的 nDCG@10 平均比监督的 monoT5（3B）高 2.7、2.3。
2. 蒸馏：1 万个 MS MARCO query、每个 BM25 取 20 个候选，让 ChatGPT 排好序，用 RankNet 式的 pairwise 损失训学生；435M 的学生在 BEIR 上平均 nDCG 比 monoT5（3B）高 1.67。

### 6. LLM 生成的 embedding / 特征怎么接进系统

1. **生产**：物品上传时进异步管线（caption、标签、内容向量），写进特征存储；模型升级时全量回刷。成本见第 3 节。
2. **使用**：作为排序模型的 dense 特征（直接用或接一个可训练的投影）、作为双塔物品塔的输入（新物品冷启动时尤其有用，见 [向量召回](/recsys/ann-retrieval)）、作为 semantic ID 的输入（TIGER 用 Sentence-T5 的 768 维向量，OneRec 用对齐过的多模态表示，见 [生成式推荐](/recsys/generative-rec)）。
3. **一致性**：特征存储里的向量要带版本号；换 LLM 或换 prompt 后，新旧向量不能混用，排序模型要用新向量重训或至少回刷训练数据。

### 7. 在线确实要用 LLM 时：query 侧 + 缓存

query 侧的 LLM（理解、改写、扩展）只和 query 有关，热门 query 高度重复：

$$
\text{LLM 调用 QPS} = Q \cdot (1 - h)
$$

$Q = 10^4$、命中率 $h = 0.7$：LLM 只需扛 3000 QPS。进一步：头部 query 离线预先算好写进 KV，在线只查表；长尾 query 用小模型或直接跳过改写。文档侧的 LLM 结果（摘要、标签）同理离线算好。真正和 (query, 文档) 成对相关的打分，才需要线上模型，这部分交给蒸馏出来的小模型。

## 面试追问

::: details Q：为什么不直接在线上用 LLM 做相关性精排？
算力和延迟。8B 模型 256 token 一次前向 $4.1\times10^{12}$ FLOPs，100 个候选、$10^4$ QPS 要约 1 万张 H100，单个请求在一张卡上要约 1 秒。离线打 3000 万条标签只要约 85 GPU 小时，所以用 LLM 当教师、蒸馏小模型上线。
:::

::: details Q：蒸馏出来的学生为什么能接近教师？
学生容量小，但训练数据是教师打的 100 倍以上的软标签，覆盖人工标注没有的语言和长尾 query；软标签也比单人约 70% 准的人工标签干净。Pinterest 学生的 5 档准确率从 0.484（30 万人工标签）提到 0.548（3000 万教师标签）。学生还用了教师没有的线上特征（互动率、embedding）。
:::

::: details Q：cross-encoder 和 bi-encoder 有什么区别？各用在哪？
bi-encoder 把 query 和文档各自编码成向量再算相似度，文档向量能离线算好，适合召回；cross-encoder 把两者拼在一起编码，能建模细粒度交互，更准，但每对都要现算，只能用在候选很少的精排 / 重排或离线当教师。
:::

::: details Q：RankGPT 对 100 个候选要调几次 LLM？为什么不能并行？
窗口 20、步长 10，从后往前滑：$(100 - 20)/10 + 1 = 9$ 次。每个窗口排好后，前 10 个要带进下一个窗口一起排，所以后一次依赖前一次的结果，只能串行。
:::

::: details Q：LLM 生成的物品 embedding 换版本时要注意什么？
新旧向量不在同一个空间，不能混用：特征存储带版本号，全量回刷所有物品，排序模型 / 双塔用新向量重训后再一起切换；新物品在切换前后都要按对应版本生成。
:::

::: details Q：query 改写这种在线 LLM 怎么省？
按 query 缓存：头部 query 离线预先算好，在线查表；在线 LLM 只处理未命中的长尾，QPS 降到 $Q(1-h)$；长尾再用小模型兜底，超时就跳过改写。
:::

## 手撕

RankGPT 式滑窗重排（LLM 用一个「按真实相关度排序」的函数代替），数一下调用次数：

```python
import random

def sliding_rerank(docs, llm_rank, w=20, s=10):
    """RankGPT 式滑窗：从列表末尾往前，每次让 LLM 重排一个长度 w 的窗口，窗口每次前移 s。"""
    docs, calls = list(docs), 0
    end = len(docs)
    while True:
        start = max(0, end - w)
        docs[start:end] = llm_rank(docs[start:end])   # 一次 LLM 调用：输出这个窗口的排列
        calls += 1
        if start == 0:
            return docs, calls
        end -= s

random.seed(0)
true_rel = {f"d{i}": random.random() for i in range(100)}
first_stage = sorted(true_rel, key=lambda d: true_rel[d] + random.gauss(0, 0.3), reverse=True)
oracle = lambda window: sorted(window, key=lambda d: -true_rel[d])   # 假装 LLM 排得完全对

out, calls = sliding_rerank(first_stage, oracle)
top10 = set(sorted(true_rel, key=lambda d: -true_rel[d])[:10])
print("LLM calls:", calls)
print("true top-10 in first-stage top-10:", len(top10 & set(first_stage[:10])))
print("true top-10 in reranked top-10:", len(top10 & set(out[:10])))
# LLM calls: 9
# true top-10 in first-stage top-10: 2
# true top-10 in reranked top-10: 10
```

读输出：一趟从后往前的滑窗，每个窗口把最好的 10 个带到下一个窗口，所以真正的 top-10（$\le s$ 个）一定能被送到最前面；但它是 9 次串行调用。窗口内排得不准时，可以再滑一趟，代价翻倍。

## 参考

- [Improving Pinterest Search Relevance Using Large Language Models](https://arxiv.org/abs/2410.17152)：cross-encoder 教师、Pin 文本特征、各模型准确率、28 万人工标签与 70% 单人准确率、3000 万蒸馏标签、学生特征、nDCG@20 +2.18%、满足率
- [Is ChatGPT Good at Search? (RankGPT)](https://arxiv.org/abs/2304.09542)：permutation 生成、滑窗 $w = 20$、$s = 10$、top-100 重排、GPT-4 对 monoT5 的提升、435M 学生
- [TWIN](https://arxiv.org/abs/2302.02352)：快手每天新发布约 4500 万个视频
- [TIGER](https://arxiv.org/abs/2305.05065)：用 Sentence-T5 的内容向量做 semantic ID
- [OneRec Technical Report](https://arxiv.org/abs/2506.13695)：多模态表示对齐后做 tokenizer
- [prefill 的算力账（姊妹站）](https://scatyf3.github.io/ai-infra-inferview/inference/prefill-decode-roofline)：每 token $2N$ FLOPs
