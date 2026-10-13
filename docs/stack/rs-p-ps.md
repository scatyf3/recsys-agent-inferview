---
title: "Parameter Server"
---

# Parameter Server

参数按 key 切到一组 server 上，worker 每步 pull 这个 batch 用到的行、push 梯度，优化器在 server 上跑。稀疏的行很少被两个 worker 同时改，可以异步；dense 参数每步都被所有人改，异步的 staleness 约等于 worker 数。PERSIA 用 embedding 异步加 dense 同步，AUC 和全同步差不到 0.1%。
