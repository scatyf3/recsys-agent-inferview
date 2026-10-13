---
title: "prefix cache / KV 复用"
---

# prefix cache / KV 复用

Agent 每一轮的输入是上一轮输入加上新追加的几百到几千个 token，前缀几乎完全相同。推理引擎把算好的 KV 按 block 存起来、用前缀 hash（vLLM）或 radix tree（SGLang）查找，命中的部分不用重新 prefill。难点在工具执行的那几秒里：这条请求的 KV 是留在 HBM、换到 CPU 内存，还是丢掉等下一轮重算。
