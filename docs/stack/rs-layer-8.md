---
title: "第 8 层：生成式推荐"
---

# 第 8 层：生成式推荐

把推荐写成序列生成：物品先编码成几个 token（semantic ID），模型读用户的行为序列，自回归地生成下一个物品；或者用一个大的序列模型替代传统 DLRM 的特征交叉。好处和代价都来自「更像 LLM」：算力密度和 MFU 上去了，级联里大量的通信和存储开销没了；但要处理码本、beam search、非法 ID 和长序列推理。

- **semantic ID**：用残差量化把十亿级物品压成 3 层、每层几千个码字的 token 序列。
- **HSTU / scaling**：pointwise 注意力、生成式训练和 jagged kernel，让序列模型能放大到万亿参数。
- **推理**：排序侧用 M-FALCON 共享用户历史的计算，生成侧是 SID 上的 beam search。
- **LLM 用于搜推**：LLM 多半放在离线（标注、embedding），线上用蒸馏出来的小模型。
