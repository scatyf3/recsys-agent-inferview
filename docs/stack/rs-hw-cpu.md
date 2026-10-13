---
title: "CPU 侧特征处理"
---

# CPU 侧特征处理

CPU 在搜推训练里干两份活：一是读样本、解析、对 ID 哈希、去重、按目标机器分桶；二是 Parameter Server 方案里的 server 本身，本质是大内存的分布式 key-value 存储。样本是 PB 级的，Meta 专门用一层 reader 节点预处理再喂给 GPU，否则 GPU 节点的主机网络和 CPU 先被打满。
