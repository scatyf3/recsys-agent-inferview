# 搜推 & Agent Infra Inferview

搜推 infra 与 Agent infra 的面试知识库，框架沿用 [ai-infra-inferview](https://github.com/scatyf3/ai-infra-inferview)：两张分层图、文章、闪卡，批注和复习记录同步到 `data` 分支。

站点：<https://scatyf3.github.io/recsys-agent-inferview/>

```bash
npm install
npm run docs:dev      # http://localhost:5173/recsys-agent-inferview/
npm test
npm run docs:build
```

- 文章：`docs/recsys/`、`docs/agent/`，frontmatter 用 `recsys: [..]` / `agent: [..]` 挂到分层图上（见 `docs/.vitepress/layers.ts`）。
- 分层图每个格子的介绍：`docs/stack/<id>.md`。
- 闪卡：`src/data/cards-*.ts`。
