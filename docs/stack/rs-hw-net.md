---
title: "网络与 all-to-all 带宽"
---

# 网络与 all-to-all 带宽

embedding 模型并行以后，每步要两次 all-to-all：ID 发给持有表的卡，pooled 向量发回来，而且在关键路径上。节点内 NVLink 每卡约 300 GB/s，跨节点每卡一块 200 Gbps（25 GB/s）的网卡，差一个数量级；所以从一个节点扩到多个节点时，通信时间往往不降反升，切表时要把通信量大的部分留在节点内。
