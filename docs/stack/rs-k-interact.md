---
title: "特征交叉 / dense 部分"
---

# 特征交叉 / dense 部分

DLRM 把底层 MLP 的输出和各个 embedding 两两做点积，再进顶层 MLP；DCN-v2 用交叉层 x0 ⊙ (W x + b) + x 做高阶交叉，满秩每层 2D² FLOPs，低秩降到 4Dr。DLRMv2 每条样本训练约 96 MFLOPs，参数却有 261 亿，所以 GPU 大半时间在等查表和通信，MFU 远低于 LLM 训练。
