---
title: "all-to-all 与流水"
---

# all-to-all 与流水

每步两次 all-to-all：ID 发到持有表的卡（input dist），查完的 pooled 向量发回样本所在的卡（output dist），后者是 dense 前向的输入，在关键路径上。TorchRec 的 TrainPipelineSparseDist 让 3 个 batch 同时在飞：一个在拷到卡上、一个在做 input dist、一个在查表和算 dense，结果和不流水完全一样。
