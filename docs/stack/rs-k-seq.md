---
title: "用户长序列 / jagged tensor"
---

# 用户长序列 / jagged tensor

target attention 要对每个候选和每条历史行为都算一次，成本是 $m \times L$；序列从几百拉到上万，就要先用便宜的检索（GSU）挑出 top-100 再做精确注意力（ESU），或者像 TWIN 那样把物品侧的投影预计算缓存。用户之间序列长度差几个数量级，补齐到最长会浪费大半算力，所以用 jagged tensor（values + offsets）和专门的 kernel。
