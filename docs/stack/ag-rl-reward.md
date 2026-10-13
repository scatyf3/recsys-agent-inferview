---
title: "verifier / reward"
---

# verifier / reward

能程序化判定的任务用 verifier 给奖励：跑单元测试、比对最终状态，便宜而且难 hack；判不了的才用奖励模型或 LLM 评审，贵而且会被 hack。SWE 类任务的 verifier 成本主要在沙箱的 CPU 时间：每条 rollout 都要把测试跑一遍。
