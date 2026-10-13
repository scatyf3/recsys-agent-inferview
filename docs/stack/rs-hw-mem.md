---
title: "HBM / DRAM / SSD 层次"
---

# HBM / DRAM / SSD 层次

embedding 表按容量和带宽分层放：一个 ZionEX 节点有 320 GB HBM（12.4 TB/s）、1.5 TB DRAM（320 GB/s），再往下是 SSD。GPU 经 PCIe 读主机内存比读 HBM 慢约 50 倍，所以表放不进 HBM 时用 HBM 当缓存，有效带宽由命中率决定：命中 90% 只剩约 220 GB/s，99% 约 960 GB/s。
