---
title: "排序推理 / batching"
---

# 排序推理 / batching

一个请求里的几百个候选天然是一个 batch，用户侧的计算（用户塔、行为序列编码）只做一次再广播给所有候选。embedding 查表是低算术强度的访存操作，适合 CPU 或独立的 embedding 服务；dense 部分和 attention 放 GPU，batch 太小时 GPU 赢不了 CPU，所以要跨请求攒 batch，用排队延迟换吞吐。
