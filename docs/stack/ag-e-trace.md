---
title: "tracing"
---

# tracing

一个 agent 任务记成一条 trace：根 span 是整个任务，下面每次模型调用、工具调用各一个子 span，带模型名、输入输出 token 数、缓存命中、耗时和错误。OpenTelemetry 的 GenAI 语义约定规定了这些字段的名字，按它打点，不同框架和后端的数据才能放在一起看「慢在哪、钱花在哪」。
