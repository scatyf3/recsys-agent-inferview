---
title: "GPU 检索 / 向量库"
---

# GPU 检索 / 向量库

GPU 擅长距离计算（就是矩阵乘）和 IVF 的顺序扫描，难点是 top-$k$ 选择，Faiss 为此专门写了 k-selection kernel；HNSW 的图遍历是串行的、延迟受限，在 GPU 上要换成 CAGRA 这类专为 GPU 设计的图。线上向量库还要解决分片与 top-$k$ 合并、新物品实时插入、属性过滤和定期重建。
