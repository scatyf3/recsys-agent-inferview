---
title: "embedding 分片"
---

# embedding 分片

GPU 同步训练里 embedding 表分到各张卡上：整张表放一张卡（table-wise）、按行切（row-wise）、按维度切（column-wise）、小表直接复制（data parallel），以及节点内按行、节点间按表的层次组合。切得不均衡时最慢的卡决定步长，TorchRec 的规划器在显存约束下枚举切法、估耗时，挑最均衡的方案。
