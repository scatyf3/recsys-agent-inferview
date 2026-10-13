---
title: "多级缓存"
---

# 多级缓存

表在 DRAM（或 UVM）里，HBM 放最热的行。FBGEMM 的 MANAGED_CACHING 默认缓存总行数的 20%，替换策略 LRU 或 LFU；Meta 的软件缓存是 32 路组相联，比直接用 UVM 快约 15%。ID 访问接近幂律，缓存两成的行就能命中九成左右，再配合提前预取下一个 batch 的行。
