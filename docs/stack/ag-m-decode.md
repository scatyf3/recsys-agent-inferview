---
title: "约束解码 / 结构化输出"
---

# 约束解码 / 结构化输出

工具调用的参数必须能被程序解析。约束解码在每个解码步根据 grammar（JSON schema 编译成的下推自动机）算出一张 token mask，把不合法的 token 的 logit 设成负无穷。难点是 mask 的计算成本：词表 12.8 万个 token，逐个检查太慢；XGrammar 把绝大多数 token 的合法性预先算好，运行时只检查不到 1% 的 token，每步开销在几十微秒以内。
