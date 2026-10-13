---
title: "RAG 管线"
---

# RAG 管线

离线：把文档切块，每块算 embedding，建向量索引和倒排索引。在线：查询（必要时先改写）走混合检索取几十个候选，rerank 后留下 top-k，拼进 prompt 交给模型。延迟大头通常是 rerank 和最后的生成，token 成本大头是拼进去的 top-k 块。
