---
title: "function calling"
---

# function calling

模型在输出里按约定格式写出「调哪个工具、参数是什么」，服务端解析出来交给应用执行，结果再作为下一轮的输入。格式各家不同：OpenAI / Anthropic API 返回结构化的 JSON 块，开源模型多是 `<tool_call>` 这类标签包 JSON 或专门的特殊 token。参数不合法时要么用约束解码从源头杜绝，要么把错误作为工具结果回给模型让它重试。
