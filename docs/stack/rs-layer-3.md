---
title: "第 3 层：训练并行"
---

# 第 3 层：训练并行

搜推训练的并行方式是混合的：embedding 表太大，按表、按行、按列切到多台机器（模型并行）；dense 部分很小，每张卡一份、梯度 AllReduce（数据并行）；两边之间每步两次 all-to-all。历史上先有 CPU 上的 Parameter Server（异步、容量大），后来变成 GPU 上的全同步（快几十倍）。这一层的取舍是一致性对速度：异步不用等，但用的是旧参数；同步结果确定，但最慢的那张卡和关键路径上的通信决定步长。

- **Parameter Server**：pull / push，稀疏部分适合异步。
- **embedding 分片**：table / row / column-wise，规划器负责切得均衡。
- **all-to-all 与流水**：通信在关键路径上，靠多 batch 流水重叠。
- **dense 部分 DP**：AllReduce 同步，可以和反向重叠。
