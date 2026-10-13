---
title: "benchmark / pass@k"
---

# benchmark / pass@k

离线评测用能自动判定的任务集：SWE-bench Verified 跑仓库的测试判 patch 对不对，tau-bench 比对对话结束时数据库的状态。模型有随机性，一题采 $n$ 次、对了 $c$ 次，用 Chen et al. 的无偏估计算 pass@k；tau-bench 的 pass^k 反过来问 $k$ 次是不是全对，衡量可靠性。
