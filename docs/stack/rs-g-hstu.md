---
title: "HSTU / scaling"
---

# HSTU / scaling

Meta 的 HSTU 把用户的物品与行为交错成一条序列，用 pointwise（SiLU，不做 softmax）注意力加门控替代 DLRM 的特征交叉，能放大到 1.5 万亿参数，指标随算力呈幂律提升。训练按「每个用户一条序列、所有位置都当目标」组织，比按曝光逐条训练省一个序列长度的因子；变长序列用 jagged kernel 和随机截短来省算力。
