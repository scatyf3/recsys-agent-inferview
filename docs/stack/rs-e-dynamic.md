---
title: "动态 embedding / 准入淘汰"
---

# 动态 embedding / 准入淘汰

动态 embedding 是一张 key → 向量的哈希表，一个 ID 一行、没有冲突，表随数据增长（Monolith 用 cuckoo hashmap）。为了不让表无限涨，新 ID 要出现够 k 次（或按概率）才建行，长期不出现的按过期时间删掉；稳态行数约等于每天新准入的 ID 数乘以过期时间。
