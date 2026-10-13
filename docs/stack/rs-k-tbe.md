---
title: "EmbeddingBag / TBE"
---

# EmbeddingBag / TBE

EmbeddingBag 是按 ID 取行再按 bag 求和，每读 4 字节只做 1 次加法，完全被带宽卡住。FBGEMM 的 TBE（Table Batched Embedding）把所有表拼成一块、一次 kernel 查完，上千张表时省下几千次 launch；反向不生成梯度张量，按行排序、合并重复行后直接原地做优化器更新。
