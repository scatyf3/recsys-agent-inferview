---
title: "dense 部分 DP"
---

# dense 部分 DP

dense 部分（MLP、交叉层）参数只有千万级，每张卡一份完整副本，各算各的样本，反向时梯度 AllReduce。和 output all-to-all 不同，梯度是逐层产生的，可以边反向边通信、藏在计算后面；早期把 dense 也放在 PS 上，每步每个 worker 都要推拉全部 dense 参数，server 很快成为热点。
