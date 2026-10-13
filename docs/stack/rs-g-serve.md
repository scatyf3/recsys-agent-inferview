---
title: "生成式推荐的推理"
---

# 生成式推荐的推理

排序侧：$m$ 个候选共用同一段用户历史，M-FALCON 把候选分成 micro-batch 塞进同一次前向，并缓存历史的 KV，成本从 $O(m n^2 d)$ 降到约 $O(n^2 d + m n d)$。生成侧：在 semantic ID 上做 $L$ 步 beam search（OneRec 一次生成 512 个），encoder 的输出被所有 beam 共享，要用前缀树约束避免非法 ID。
