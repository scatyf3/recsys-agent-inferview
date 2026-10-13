---
title: "隔离：容器 / microVM"
---

# 隔离：容器 / microVM

容器共享宿主内核，靠 namespace、cgroup、seccomp 隔离，攻击面是整个 Linux syscall 接口。gVisor 在用户态重新实现一个内核，拦截并处理沙箱里的 syscall。Firecracker 这类 microVM 给每个沙箱一个独立的客户机内核，靠硬件虚拟化隔离。跑模型生成的不可信代码，一般至少用 gVisor 或 microVM。
