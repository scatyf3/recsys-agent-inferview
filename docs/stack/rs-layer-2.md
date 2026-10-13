---
title: "第 2 层：算子"
---

# 第 2 层：算子

搜推模型的算子分两类：稀疏的查表和稠密的特征交叉 / MLP。查表是 gather 加 pooling，算术强度极低，优化全在少搬字节和少启动 kernel（TBE 把上千张表合成一次、反向和优化器融合）；交叉和 MLP 是常规矩阵乘，但矩阵窄、形状碎，很难吃满 tensor core。用户行为序列这类变长输入还要用 jagged tensor 表示，避免 padding 的浪费。

- **EmbeddingBag / TBE**：一次 kernel 查完所有表，反向原地更新。
- **特征交叉**：DLRM 的点积、DCN 的交叉层，算力账和 MFU。
- **长序列 / jagged tensor**：变长序列的存储和计算。
