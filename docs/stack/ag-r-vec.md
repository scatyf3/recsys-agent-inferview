---
title: "向量检索 / rerank"
---

# 向量检索 / rerank

第一阶段用 bi-encoder：查询和文档各自编码成向量，用 ANN 索引找内积最大的；再加一路 BM25 补关键词的精确匹配，两路用 RRF 按名次融合。第二阶段用 cross-encoder 把查询和每个候选拼在一起打分，准但每个候选都要过一次模型，所以只对几十个候选做。ColBERT 的 late interaction 介于两者之间。
