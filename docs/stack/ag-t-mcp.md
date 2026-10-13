---
title: "MCP / 工具协议"
---

# MCP / 工具协议

MCP 把「工具的提供方」和「用工具的应用」解耦：应用（host）里的每个 client 连一个 server，server 用 JSON-RPC 暴露工具、资源和提示模板，传输走本地子进程的 stdio 或远程的 Streamable HTTP。系统上的问题是工具定义的 token 开销：几个 server 加起来几十个工具，定义就能占几万 token，所以要按需加载、做工具检索。
