---
title: 搜推 infra
---

# 搜推 infra

搜推系统和 LLM 的系统问题长得很不一样：参数大头是 TB 级的稀疏 embedding 表，每步只碰其中极小一部分；线上要在几十毫秒里把亿级候选筛到几十个；模型要按分钟级追新数据。这里从这三件事展开。

::: tip 面试官真正在考什么
能不能把账算清楚：embedding 表多大、一步要查多少行、all-to-all 传多少字节、一次请求的延迟花在哪一级漏斗；以及模型多久更新一次、怎么不出事地推上线。
:::

点左边的层名看这一层在管什么，点格子看介绍和相关文章；右上角的数字是挂在这个格子上的文章数，虚线格是还没有文章的主题。

<StackFigure stack="recsys" />

## 先看全景

- [搜推系统全景](./funnel-overview) — 召回、粗排、精排、重排，每级候选数、QPS 与延迟预算
- [系统设计：短视频推荐系统](./system-design) — 从量级估算到召回、排序、特征、训练、更新全链路

## Embedding：稀疏大表

- [Embedding 表的账](./embedding-tables) — 参数量、哈希冲突、动态 embedding、准入与淘汰、压缩
- [Embedding 查表](./embedding-lookup) — EmbeddingBag / TBE kernel、访存账、HBM / DRAM / SSD 多级缓存

## 训练：并行、算力、样本、更新

- [Parameter Server](./parameter-server) — pull / push、同步与异步、staleness、PS 与 AllReduce 混合
- [Embedding 分片与 all-to-all](./embedding-sharding) — table / row / column-wise、TorchRec planner、训练流水
- [排序模型的算力账](./ranking-model-compute) — DLRM / DCN、为什么 MFU 低、dense 部分怎么提速
- [样本与特征管线](./sample-pipeline) — 样本拼接、延迟标签、负采样、训练-服务一致性
- [在线学习与模型更新](./online-learning) — 流式训练、增量 checkpoint、参数同步、AUC / GAUC / NE 与 A/B

## 服务：召回与排序

- [用户长序列建模](./long-sequence) — DIN、SIM、TWIN 的计算与存储账
- [向量召回](./ann-retrieval) — 双塔、logQ 校正、HNSW 与 IVF-PQ
- [排序服务](./ranking-serving) — 取特征、batching、CPU vs GPU、embedding 服务、缓存与降级

## 新范式

- [生成式推荐](./generative-rec) — semantic ID、HSTU、OneRec 的训练与推理
- [LLM 用于搜推](./llm-for-search) — 生成特征、相关性精排、离线蒸馏与成本账
