---
title: "embedding 表与哈希"
---

# embedding 表与哈希

参数量 = 各表行数 × 维度之和，训练时还要加优化器状态：MLPerf DLRMv2 的 26 张表 2.04 亿行、128 维，fp32 加 Adagrad 状态共 209 GB。ID 空间远大于能存的行数时，用哈希把 ID 映射到固定行数，N 个 ID 放进 m 行的冲突比例约 N/(2m)；商余（QR）分解用两张小表给每个 ID 拼出唯一的组合。
