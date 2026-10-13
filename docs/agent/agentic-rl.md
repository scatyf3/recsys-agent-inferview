---
title: Agentic RL 的系统账：环境规模化、多轮长尾、异步与 verifier 成本
status: draft
tags: [agentic-rl, environment-scaling, sandbox, rollout, partial-rollout, verifier, reward-hacking]
difficulty: 4
order: 13
related: [/agent/sandbox, /agent/eval-observability, /agent/orchestration]
agent: [ag-rl-env, ag-rl-rollout, ag-rl-reward]
---

# Agentic RL 的系统账：环境规模化、多轮长尾、异步与 verifier 成本

> 训练 agent 的每条 rollout 都要一个真实的环境：一个装好仓库的容器，跑几十轮，最后跑测试拿奖励。GPU 那边的事 sister site 讲过了；这页只算环境侧的账：要多少个沙箱、多少 CPU 和磁盘、为什么一步训练总被最慢的那个环境拖住、奖励本身花多少钱、又怎么被 hack

## 一句话结论

一步训练 $B$ 个 prompt、每个采 $n$ 条，就要 $Bn$ 个独立环境：DeepSWE 是 $64 \times 8 = 512$ 个 Docker 容器；SkyRL 估每个 SWE 容器要 1 个以上 CPU、约 7 GB 磁盘，512 个就是约 3.6 TB；再加上异步带来的在途倍数，Kimi K2 跑了一万多个并发沙箱，Qwen3-Coder 说是两万个并行环境。镜像存储按「每题一个镜像」做会到几十 TB（SWE-Gym 2,438 题 6 TB），按仓库共享镜像能压到几百 GB（SWE-smith 5 万题 295 GB）。环境比 GPU 更容易出事：每个环境每步只有 0.1% 的失败率，512 个里至少坏一个的概率就有 40%，同步训练就会等它超时（RollArt 实测：出现环境故障的 step 从 366 s 变成 513 s，`env.reset` 占 rollout 时间的 78%）。所以要环境级异步、多开冗余环境凑够就停、轨迹太长就挂起（partial rollout）。奖励上，SWE 任务能跑测试就跑测试（DeepSWE 0/1 奖励，测试超时 5 分钟），代价是 CPU 时间；SWE-RL 干脆不执行，用 patch 和标准答案的文本相似度当奖励，换来几十万条训练数据。verifier 也会被 hack：SWE-bench 镜像里残留的未来 git 历史曾被 RL 训练中的模型用 `git log --all` 找到答案。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $B$ | 一步训练的 prompt（任务）数 | 64 |
| $n$ | 每个任务采样几条轨迹 | 8 |
| $R$ | 一步要跑的 rollout 数，$R = Bn$ | 512 |
| $u$ | 在途倍数：同时活着的环境数 / $R$（异步、冗余带来） | 1–20 |
| $c_{\text{cpu}}$、$s_{\text{disk}}$ | 每个环境占的 CPU 核数、磁盘 | 1 核、7 GB（SkyRL 估计） |
| $p_f$ | 一个环境在一步里失败（启动失败、超时、崩溃）的概率 | 0.1%（假设） |
| $t_{\text{test}}$ | 跑一次测试拿奖励的时间 | 2 分钟（假设），上限 5 分钟 |
| $T$ | 一条轨迹的轮数上限 | 50–100 |

算法侧（token 进 token 出、loss mask、staleness、decoupled PPO）不在这页，见 sister site 的 [Agentic RL 的 rollout 系统](https://scatyf3.github.io/ai-infra-inferview/posttrain/rl-agentic)、[异步 Rollout](https://scatyf3.github.io/ai-infra-inferview/posttrain/rl-async-rollout) 和 [RL 训练提速](https://scatyf3.github.io/ai-infra-inferview/train/efficient-rl)。

### 1. 一步训练要多少个环境

每条 rollout 要一个独立环境：同一道题的 8 条轨迹会各自改文件、装包、跑测试，不能共用。

$$
\text{并发环境数} = u \cdot R = u \cdot Bn
$$

**同步训练**（$u = 1$）：DeepSWE 每步 $64 \times 8 = 512$ 个 Docker 容器（[DeepSWE 博客](https://www.together.ai/blog/deepswe)，2025-07-02）。

**资源**：SkyRL-v0 的估计（[博客](https://novasky-ai.notion.site/skyrl-v0)，2025-05）是每个容器 1 个以上 CPU、约 7 GB 磁盘，batch 16 × 8 条就要 100 多个 CPU、接近 1 TB 磁盘。按这个算 512 个：

1. CPU：512 核以上。
2. 磁盘：$512 \times 7 = 3.6$ TB。
3. DeepSWE 的集群每个节点约 200 核、6 TB 以上 NVMe，至少 3 个节点；他们说能扩到 1000 个以上 CPU 核，靠 Kubernetes 的 Cluster Autoscaler 增减节点。

**异步之后更多**：轨迹级异步、冗余采样会让同时活着的环境数是一步所需的好几倍。公开的数字：

1. Kimi K2（[arXiv 2507.20534](https://arxiv.org/abs/2507.20534) §3.2.1）：基于 Kubernetes 的沙箱，支持一万个以上并发沙箱实例。
2. Qwen3-Coder（[博客](https://qwenlm.github.io/blog/qwen3-coder/)，2025-07-22）：在阿里云上搭了能并行跑两万个独立环境的系统。
3. MiMo-V2-Flash（[arXiv 2601.02780](https://arxiv.org/abs/2601.02780) §4.3.2）：Kubernetes 集群上一万个以上并发 pod，约 12 万个环境。

GPU 只有几十到几百张，CPU 侧的沙箱集群却是上万个容器。环境集群要单独规划、单独扩缩容，不能和训练节点绑在一起。

### 2. 镜像存储和环境构建

**每题一个镜像太大**。SWE 任务的环境 = 仓库在某个 commit 的代码 + 装好的依赖。SWE-smith 论文表 2（[arXiv 2504.21798](https://arxiv.org/abs/2504.21798)）：

| 数据集 | 题数 | 环境总大小 | 每题均摊 |
|---|---|---|---|
| SWE-Gym | 2,438 | 6 TB | 约 2.5 GB |
| R2E-Gym | 8,135 | 4 TB | 约 0.5 GB |
| SWE-smith | 50,000（128 个仓库） | 295 GB | 约 6 MB |

SWE-smith 的做法：同一个仓库的任务共用一个镜像（每个仓库约 2.3 GB），任务之间的差别只是代码改动（往正常代码里注入 bug 生成题目）。论文估计按 SWE-bench 的方式做 5 万题要 50–150 TB，差约 500 倍。镜像小了，节点能把常用镜像都缓存在本地，环境启动就不用拉镜像（拉镜像为什么慢见 [代码执行沙箱](/agent/sandbox) 第 2 节）。

**环境构建的成功率不高**。从真实 PR 自动造环境，难在把依赖装对：

1. SWE-rebench（[arXiv 2505.20411](https://arxiv.org/abs/2505.20411)）：约 45 万个 PR 起步，自动生成的安装脚本只对 31% 的仓库成功，最后得到 2.1 万多道题。
2. MiMo-V2-Flash：8 种语言的环境搭建成功率 70%。

**数据规模对比**：

| 数据 | 规模 | 需要执行环境吗 |
|---|---|---|
| SWE-Gym | 2,438 题 | 要 |
| R2E-Gym | 8,135 题（DeepSWE 用其中 4.5k） | 要 |
| SWE-smith | 5 万题 | 要 |
| SWE-RL 的 PR 数据 | 27.3 万个 PR 种子 | 不要（见第 5 节） |

### 3. 多轮 rollout 的长尾：环境比 GPU 更容易出事

一条轨迹的时间是各轮生成和环境执行之和：

$$
t_{\text{traj}} = \sum_{i=1}^{T}\big(t_{\text{gen},i} + t_{\text{env},i}\big) + t_{\text{reset}} + t_{\text{test}}
$$

同步训练一步要等 $R$ 条里最慢的那条。生成侧的长尾（个别回答特别长）sister site 讲过了；环境侧多了一种更糟的长尾：**失败**。

**一步里至少一个环境出事的概率**：

$$
P(\text{至少一个失败}) = 1 - (1 - p_f)^R
$$

1. $p_f = 0.1\%$、$R = 512$：$1 - 0.999^{512} = 0.40$。每个环境「千分之一会坏」，一步里却有 40% 的概率至少坏一个。
2. $p_f = 1\%$：$0.994$，几乎每步都有。
3. $R = 100$、$p_f = 0.1\%$：$0.095$。$R$ 越大越躲不掉。

坏的环境要等超时才知道，同步训练这段时间全体空等。RollArt 的实测（[arXiv 2512.22560](https://arxiv.org/abs/2512.22560) §3.1，Qwen3-8B、SWE 任务，作者自测）：

1. 正常 step 平均 365.7 s；出现环境故障的 step 是 513.3 s。
2. 环境超时时，`env.reset` 一项占 rollout 时间的 78%；生产环境里 `env.reset` 的长尾可以到几百秒。
3. 大约每 10 个 step 坏一次。修好之后 `env.reset` 成功率超过 99.99%，99.99% 以上在 1 分钟内完成（§8）。

### 4. 对策：环境级异步、冗余、partial rollout

**冗余**：多开一些环境，凑够 $R$ 条就停，慢的和坏的丢掉。开 $R(1 + \epsilon)$ 个，只要坏掉和超慢的不超过 $\epsilon R$ 个，就不用等。ROLL Flash 报告冗余环境 rollout 带来额外 7–16% 的加速（[arXiv 2510.11345](https://arxiv.org/abs/2510.11345)，作者自测）。代价：多出来的那部分 GPU 生成和环境被浪费，而且丢掉的偏向「慢的」轨迹（往往是难题），会让样本分布偏向简单题。

**环境级异步**：每条轨迹的环境调用独立返回，不按批同步。ROLL Flash 的真实 SWE 训练从 10.22 h 降到 8.32 h（作者自测）。

**流水化**：把一条轨迹拆成「启动环境 → 跑 agent → 跑测试算奖励」三段，各自排队。前后两段只用 CPU，中间段用 GPU，三段重叠起来 GPU 就不用等 CPU。SkyRL-Agent（[arXiv 2511.16108](https://arxiv.org/abs/2511.16108)）报告这样比按批异步快约 1.55 倍、GPU 利用率约 90%；SkyRL-v0 的三段流水比朴素基线快 4–5 倍（作者自测）。

**partial rollout**：太长的轨迹不等它，挂起、下一轮接着跑。Kimi K2 §3.3.4：为防止长尾轨迹阻塞整个 rollout，用 partial rollout，让没跑完的长任务暂停、在下一轮 RL 迭代里继续。Kimi-Researcher（[2025-06-20](https://moonshotai.github.io/Kimi-Researcher/)）做到按轮挂起，报告至少 1.5 倍加速（作者自测）。挂起的不只是 token 序列，还有**环境**：沙箱要一直活着或打快照，这又是一份资源账。

**要多少条并发轨迹**才能喂饱 GPU：约为「想同时 decode 的序列数 ÷ 轨迹在生成上花的时间比例」，推导见 sister site [Agentic RL](https://scatyf3.github.io/ai-infra-inferview/posttrain/rl-agentic)。环境越慢，这个数越大，就是第 1 节里 $u$ 的来源。

### 5. Verifier / reward 的成本和风险

**跑测试（执行式 verifier）**：DeepSWE 的奖励是稀疏的 0/1：选定的 FAIL_TO_PASS 和 PASS_TO_PASS 测试全过才给 1；跑测试的超时是 5 分钟（官方 SWE-bench 评测是 30 分钟），生成的超时 20 分钟。

算一笔：$R = 512$、每次 $t_{\text{test}} = 2$ 分钟（假设）、每次 1 核：每步 $512 \times 2 = 1024$ 核·分钟 ≈ 17 核·小时。并行时不慢（512 个同时跑，2 分钟），但最慢的测试决定这一段的时长，所以超时要设，而且要比评测时紧。

**不执行（文本相似度）**：SWE-RL（[arXiv 2502.18449](https://arxiv.org/abs/2502.18449)）用 GitHub 的 PR 数据，奖励是预测 patch 和真实 patch 的 `difflib.SequenceMatcher` 相似度（0 到 1），格式错误给 −1。好处是不需要可执行环境，数据从几千题扩到 27.3 万个 PR 种子；Llama3-SWE-RL-70B 在 SWE-bench Verified 上 41.0%（作者自测）。代价是奖励只是「像不像标准答案」，不同写法的正确解会被低估。

**奖励模型 / LLM 评审**：判不了的任务（研究报告、开放式对话）才用。它要 GPU，而且负载不均：只在轨迹结束时集中来一批。RollArt 把奖励模型做成 serverless 共享服务，奖励 GPU 的利用率从 6% 提到 88%（作者自测）。DeepSeek-R1-Zero 不用神经奖励模型，理由是大规模 RL 里它可能被 reward hack（[arXiv 2501.12948](https://arxiv.org/abs/2501.12948) §2.2.2）。

| | 跑测试 | 文本相似度 | 奖励模型 / LLM 评审 |
|---|---|---|---|
| 成本在哪 | CPU、沙箱 | 几乎为 0 | GPU |
| 需要可执行环境 | 要 | 不要 | 看任务 |
| 准确性 | 高（测试质量决定） | 低，正确的不同写法被低估 | 中，会被 hack |
| 数据规模 | 受环境构建限制 | 能用所有历史 PR | 不受限 |

**verifier 被 hack**：环境里任何能「偷看答案」的漏洞，RL 都会找到，因为找到了就有奖励。

1. SWE-bench 的镜像里残留着 base commit 之后的 git 历史，agent 用 `git log --all` 能看到修复提交（[SWE-bench issue #465](https://github.com/SWE-bench/SWE-bench/issues/465)，2025-09）。
2. MiMo-V2-Flash 附录 B：RL 训练中模型学会了偷看未来提交来拿奖励，他们通过统计 rollout 里 `git log --all` 一类命令出现的次数发现。
3. 防法：环境里删掉 base commit 之后的历史（SWE-bench v4.1.0 镜像就这么修的）；测试文件不放进 agent 能读到的地方；监控轨迹里的可疑命令；奖励突然上涨时抽样看轨迹。

### 6. 几个系统各自的重点

| 系统 | 环境侧的重点 |
|---|---|
| [SWE-RL](https://arxiv.org/abs/2502.18449) | 不执行，相似度奖励，靠数据量 |
| [DeepSWE](https://www.together.ai/blog/deepswe) / [rLLM](https://github.com/rllm-org/rllm) | R2E-Gym 4.5k 题，每步 512 容器，Kubernetes 自动扩缩，64 张 H100 训 6 天 |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | 远程沙箱服务（每个 16 核节点约 80–100 个容器），三段流水，SkyRL-Gym 环境库 |
| [Kimi K2](https://arxiv.org/abs/2507.20534) | 一万以上并发沙箱，partial rollout |
| [AReaL](https://arxiv.org/abs/2505.24298) | 全异步、可中断生成；论文只做单步任务，多轮 agent 留作后续 |
| [slime](https://github.com/THUDM/slime) | Megatron + SGLang，经数据 buffer 连接，有面向长尾 agent 生成的全异步示例 |
| [verl agent loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) | 每条轨迹一个协程，第一轮选负载最低的推理服务、之后黏在同一个上 |

## 面试追问

::: details Q：agentic RL 一步训练要多少个沙箱？怎么估资源？
$R = Bn$ 个（每条轨迹一个，不能共用），再乘异步和冗余带来的在途倍数。DeepSWE 每步 512 个容器；按 SkyRL 的每容器 1 核、7 GB 估，就是 512 核、3.6 TB 磁盘。异步之后同时活着的更多，Kimi K2 和 MiMo 都是一万以上并发。环境集群要和 GPU 集群分开规划。
:::

::: details Q：为什么环境的失败率对同步 RL 影响这么大？
一步要等所有 $R$ 个环境，至少一个失败的概率是 $1 - (1-p_f)^R$：$p_f = 0.1\%$、$R = 512$ 时已经 40%。坏的环境要等到超时才发现，全体空等。RollArt 实测有环境故障的 step 从 366 s 变成 513 s。对策是冗余（多开、凑够就停）、环境级异步、更紧的超时，以及把环境启动做稳。
:::

::: details Q：SWE 环境的镜像怎么存？
每题一个镜像会到几十 TB（SWE-Gym 2,438 题 6 TB）。按仓库共享基础镜像、每题只放代码差异，SWE-smith 5 万题只要 295 GB。镜像小了节点能本地缓存，环境启动不用拉镜像。
:::

::: details Q：SWE-RL 为什么用文本相似度当奖励？有什么问题？
不需要可执行环境：自动搭环境的成功率只有三成左右，而 GitHub 上的 PR 有几十万。相似度奖励能用上全部数据，SWE-RL 70B 在 Verified 上 41%。问题是只奖励「像标准答案」：不同写法的正确解得分低，可能把模型推向模仿而不是解决问题。能跑测试时，执行式奖励更准。
:::

::: details Q：SWE 任务的 RL 会被怎么 hack？怎么防？
任何能偷看答案的漏洞都会被找到。真实案例：SWE-bench 镜像里残留未来的 git 历史，模型用 `git log --all` 找到修复提交，MiMo 在 RL 训练里就遇到了。防法：清掉 base commit 之后的历史、测试不放在 agent 能读的位置、监控轨迹里的可疑命令、奖励异常上涨时抽样看轨迹。
:::

::: details Q：verifier 的成本主要在哪？
执行式的在 CPU：每条轨迹跑一遍测试，512 条 × 2 分钟就是约 17 核·小时，并且最慢的测试决定这一段的时长，所以测试超时要比评测时紧（DeepSWE 5 分钟，官方评测 30 分钟）。奖励模型的在 GPU，而且只在轨迹结束时集中来一批，利用率低，RollArt 做成共享的 serverless 服务后从 6% 提到 88%。
:::

## 参考

- [DeepSWE（Together AI，2025-07-02）](https://www.together.ai/blog/deepswe)：512 容器 / 步、Kubernetes、0/1 奖励、测试超时
- [SkyRL-v0 博客](https://novasky-ai.notion.site/skyrl-v0)、[SkyRL 仓库](https://github.com/NovaSky-AI/SkyRL)、[SkyRL-Agent (arXiv 2511.16108)](https://arxiv.org/abs/2511.16108)：每容器资源、远程沙箱、三段流水
- [Kimi K2 (arXiv 2507.20534)](https://arxiv.org/abs/2507.20534)：§3.2.1 一万以上并发沙箱、§3.3.4 partial rollout
- [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/)：按轮 partial rollout
- [Qwen3-Coder 博客](https://qwenlm.github.io/blog/qwen3-coder/)（2025-07-22）：两万个并行环境
- [MiMo-V2-Flash (arXiv 2601.02780)](https://arxiv.org/abs/2601.02780)：§4.3.2 环境规模、附录 B SWE reward hacking
- [SWE-smith (arXiv 2504.21798)](https://arxiv.org/abs/2504.21798)：表 2 环境存储
- [SWE-Gym (arXiv 2412.21139)](https://arxiv.org/abs/2412.21139)、[R2E-Gym (arXiv 2504.07164)](https://arxiv.org/abs/2504.07164)、[SWE-rebench (arXiv 2505.20411)](https://arxiv.org/abs/2505.20411)：数据规模、环境构建成功率
- [SWE-RL (arXiv 2502.18449)](https://arxiv.org/abs/2502.18449)：相似度奖励、PR 数据规模、41.0%
- [RollArt (arXiv 2512.22560)](https://arxiv.org/abs/2512.22560)：§3.1 环境故障与 `env.reset`、serverless 奖励
- [ROLL Flash (arXiv 2510.11345)](https://arxiv.org/abs/2510.11345)：环境级异步、冗余 rollout
- [AReaL (arXiv 2505.24298)](https://arxiv.org/abs/2505.24298)、[slime](https://github.com/THUDM/slime)、[verl agent loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)、[rLLM](https://github.com/rllm-org/rllm)
- [DeepSeek-R1 (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948)：§2.2.2 不用神经奖励模型的理由
- [SWE-bench issue #465](https://github.com/SWE-bench/SWE-bench/issues/465)：git 历史泄露与修复
