---
title: Agent infra
---

# Agent infra

Agent 对推理系统的要求和聊天不一样：一个任务是几十轮调用，每轮前缀几乎不变、只多几百个 token；中间夹着工具执行，GPU 那边要决定 KV 留不留；代码要在隔离的沙箱里跑；任务能跑几十分钟，中途挂了要能续。这里从模型推理一路往上讲到 Agentic RL。

::: tip 面试官真正在考什么
能不能把一个 agent 任务的账算清楚：多少轮、每轮多少输入输出 token、prefix cache 命中多少、钱和时间各花在哪；以及沙箱、编排、网关这些组件在规模上来以后怎么不挂。
:::

点左边的层名看这一层在管什么，点格子看介绍和相关文章；右上角的数字是挂在这个格子上的文章数，虚线格是还没有文章的主题。

<StackFigure stack="agent" />

## 先算账

- [Agent 负载画像](./agent-workload) — 一个任务多少轮、每轮多少 token，钱看输入和缓存命中，时间看输出和轮数
- [系统设计：coding agent 平台](./system-design) — 从量级估算到沙箱、编排、网关的整体架构

## 推理与上下文

- [多轮 agent 的 KV 复用](./prefix-reuse) — 前缀怎么命中、工具等待时 KV 留不留、session 亲和
- [Prompt caching 的账](./prompt-caching) — 写入 / 读取价格倍数、TTL、上下文怎么排才命中
- [上下文管理](./context-management) — token 预算、截断、压缩、sub-agent 隔离
- [Agent 延迟优化](./agent-latency) — 端到端时延怎么拆，长 prefill、并行与投机执行

## 工具与检索

- [Function calling 与约束解码](./function-calling) — 格式、解析、token mask 的成本、失败恢复
- [MCP 与工具协议](./tool-protocols) — 架构、传输、工具定义占多少 token、并行调用
- [RAG 与记忆](./rag-memory) — 切块、混合检索、rerank、长期记忆

## 运行时

- [代码执行沙箱](./sandbox) — 容器 / gVisor / microVM、冷启动、预热池、安全
- [编排与持久执行](./orchestration) — 多 agent 的 token 账、重试与幂等、断点续跑
- [LLM 网关](./llm-gateway) — 多租户限流、配额、背压、模型路由
- [评测与可观测](./eval-observability) — benchmark、pass@k、tracing、线上回放

## 训练

- [Agentic RL 的系统账](./agentic-rl) — 环境规模化、多轮长尾、异步、verifier 成本
