---
title: "第 8 层：Agentic RL"
---

# 第 8 层：Agentic RL

在真实环境里用强化学习训 agent：每条 rollout 是几十轮的交互，要一个独立沙箱，最后跑测试拿奖励。这一层管环境怎么规模化到成千上万个并发，多轮 rollout 的长尾怎么不拖住 GPU（异步、partial rollout），以及 verifier 本身的成本和被 hack 的风险。权衡是 GPU 利用率对策略新鲜度：越异步 GPU 越满，样本越 off-policy。

- **环境规模化**：镜像、启动、失败率，CPU 侧的沙箱集群和 GPU 一样要算。
- **多轮 rollout**：轨迹级异步、过采样、超长轨迹屏蔽。
- **verifier / reward**：能跑测试就跑测试，判不了才用模型打分。
