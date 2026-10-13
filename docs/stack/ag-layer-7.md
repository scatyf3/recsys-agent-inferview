---
title: "第 7 层：评测与可观测"
---

# 第 7 层：评测与可观测

Agent 的输出是多步、带随机性的，跑一次说明不了好坏。这一层管离线 benchmark 怎么跑、pass@k 怎么算才无偏，线上一个任务的 trace 怎么记（每次模型调用、工具调用一个 span，带 token 数和耗时），以及怎么把线上轨迹拿回来回放做回归。权衡是评测置信度对成本：题目和重复次数越多越准，但每跑一次都是真实的 token 和沙箱开销。

- **tracing**：span 树、token 计量、按任务汇总成本和延迟。
- **benchmark**：SWE-bench Verified、tau-bench，pass@k 和 pass^k。
- **线上评估**：采样打分、轨迹回放、回归门禁。
