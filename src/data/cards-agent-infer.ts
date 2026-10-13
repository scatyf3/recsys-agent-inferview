import type { Card } from '@lib/flashcards'
import { lines } from '@lib/flashcards'

// Agent 推理、上下文、工具：对应 docs/agent/ 下 agent-workload 到 agent-latency 七篇。
// id 前缀：ag-wl- 负载、ag-kv- KV 复用、ag-cache- prompt caching、ag-ctx- 上下文管理、
// ag-fc- function calling、ag-mcp- 工具协议、ag-lat- 延迟。

const WL = '/agent/agent-workload'
const KV = '/agent/prefix-reuse'
const PC = '/agent/prompt-caching'
const CTX = '/agent/context-management'
const FC = '/agent/function-calling'
const MCP = '/agent/tool-protocols'
const LAT = '/agent/agent-latency'

const c = (x: Omit<Card, 'deck'>): Card => ({ deck: 'agent', ...x })

export const agentInferCards: Card[] = [
  // ---------------- 负载画像 ----------------
  c({
    id: 'ag-wl-total-input',
    topic: 'Agent 负载',
    q: '一个 agent 任务跑 $T = 50$ 轮。第 $i$ 轮的输入 $L_i = S + (i-1)a$：$S = 10\\text{K}$ 是固定前缀（系统提示 + 工具定义 + 任务，单位 token），$a = 1.5\\text{K}$ 是每轮追加的 token（模型输出 + 工具结果）。整个任务的总输入 $\\sum_i L_i$ 是多少？最后一轮的上下文多长？',
    a: '等差数列求和：\n$$\\sum_{i=1}^{T} L_i = TS + a\\,\\frac{T(T-1)}{2}$$\n1. $TS = 50 \\times 10\\text{K} = 500\\text{K}$。\n2. $a \\cdot T(T-1)/2 = 1.5\\text{K} \\times 1225 = 1837.5\\text{K}$。\n3. 合计 **2,337.5K ≈ 234 万 token**。\n4. 最后一轮 $L_{50} = 10\\text{K} + 49 \\times 1.5\\text{K} = 83.5\\text{K}$。\n为什么第二项大：每轮都要把整个历史再发一次，历史线性增长，总量就是 $T^2$ 量级。轮数翻倍，总输入接近翻四倍。',
    ref: WL,
  }),
  c({
    id: 'ag-wl-io-ratio',
    topic: 'Agent 负载',
    q: '接上一张卡：50 轮、总输入 2,337.5K token。每轮模型输出 $o = 300$ token。这个任务的输入 : 输出是多少？为什么 agent 的这个比值远高于聊天？',
    a: '总输出 $To = 50 \\times 300 = 15\\text{K}$，输入 : 输出 $= 2337.5 / 15 \\approx$ **156 : 1**。Manus 实测约 100 : 1，同一量级。\n原因：\n1. 每轮输出只是一个工具调用加几句话（几百 token），输入却是整个历史。\n2. 工具结果（文件内容、搜索结果、命令输出）作为输入进来，通常比模型自己写的长。\n3. 轮数多，历史反复计入输入。\n聊天每轮回答几百 token、历史短，大约是个位数比一。所以 agent 的 prefill 量和账单都由输入决定。',
    ref: WL,
  }),
  c({
    id: 'ag-wl-reuse',
    topic: 'Agent 负载',
    q: '前缀复用率上限 $h = 1 - \\dfrac{S + (T-1)a}{\\sum_i L_i}$：分子是整个任务里第一次出现的 token（第 1 轮的固定前缀 $S$，加之后每轮新追加的 $a$），分母是总输入。$S = 10\\text{K}$、$a = 1.5\\text{K}$、$T = 50$、总输入 2,337.5K，$h$ 是多少？为什么说它是上限？',
    a: '1. 第一次出现的 token：$10\\text{K} + 49 \\times 1.5\\text{K} = 83.5\\text{K}$。\n2. $h = 1 - 83.5 / 2337.5 = $ **96.4%**。\n是上限，因为它假设：\n1. 前缀 token 级完全不变（没有时间戳、没有改写历史）；\n2. 缓存在两轮之间没被淘汰、没过期；\n3. 每轮请求都回到存着 KV 的那台机器。\n任何一条不满足，实际命中率都会低于它。',
    ref: WL,
  }),
  c({
    id: 'ag-wl-money',
    topic: 'Agent 负载',
    q: '一个任务总输入 2,337.5K token、总输出 15K token。输入价 $p_{\\text{in}} = 3$ USD / 百万 token，输出价 $p_{\\text{out}} = 15$ USD / 百万 token（输出是输入的 5 倍）。不开缓存时账单多少？输入占几成？',
    a: '1. 输入：$2.3375 \\times 3 = 7.01$ USD。\n2. 输出：$0.015 \\times 15 = 0.225$ USD。\n3. 合计 7.24 USD，输入占 **97%**。\n输出单价高 5 倍也扭不过来，因为量差了 156 倍。所以 agent 的成本优化第一件事是让输入命中缓存（缓存读取约 0.1 倍价），其次才是少调几轮、压短上下文。',
    ref: WL,
  }),
  c({
    id: 'ag-wl-time',
    topic: 'Agent 负载',
    q: '每轮时间 = TTFT + 输出 token 数 × TPOT + 工具时间。设 TTFT = 0.6 s（前缀命中时的首 token 延迟），每轮输出 $o = 300$ token，TPOT = 20 ms（每个输出 token 的解码时间），工具 1.5 s，共 50 轮。任务多久？哪一项最大？',
    a: '每轮 $0.6 + 300 \\times 0.02 + 1.5 = 8.1$ s，50 轮 **405 s**（约 6.8 分钟）。\n1. 解码：$6.0 \\times 50 = 300$ s，**74%**。\n2. 工具：75 s，19%。\n3. TTFT：30 s，7%。\n结论：**钱花在输入（prefill），时间花在输出（decode）**。缩短时延先砍输出 token 和轮数；TTFT 小的前提是缓存命中，失效时它会跳到好几秒。',
    ref: WL,
  }),

  // ---------------- KV 复用 ----------------
  c({
    id: 'ag-kv-block-match',
    topic: 'KV 复用',
    q: 'vLLM 的 prefix cache 按 $B = 16$ token 一块，只缓存满块；匹配时至少留最后 1 个 token 去算（要拿它的 logits），所以最多匹配 $\\lfloor (L-1)/B \\rfloor$ 块。第 1 轮整条序列（输入 + 输出）10,300 token 已缓存；第 2 轮输入 $L = 11{,}500$（前 10,300 和第 1 轮相同）。第 2 轮命中多少 token、要算多少？',
    a: '1. 第 1 轮缓存的满块：$\\lfloor 10300/16 \\rfloor = 643$ 块，即 $643 \\times 16 = 10{,}288$ token。最后 12 个 token 在未满的块里，没缓存。\n2. 第 2 轮最多可匹配 $\\lfloor 11499/16 \\rfloor = 718$ 块，表里只有前 643 块。\n3. 命中 **10,288**，要算 $11500 - 10288 =$ **1,212** token，命中率 89.5%。\n只缓存满块是因为未满的块还会被追加写，不能共享；轮数越多、上下文越长，命中率越接近上限。',
    ref: KV,
  }),
  c({
    id: 'ag-kv-chain-hash',
    topic: 'KV 复用',
    q: 'vLLM 第 $j$ 块的 hash 是 $H_j = \\text{hash}(H_{j-1}, \\text{tokens}_j, \\text{extra})$，带上了前一块的 hash。为什么要这样？它带来什么后果？',
    a: '因为一个位置的 KV 取决于它之前的**所有** token，单看本块 16 个 token 相同不代表 KV 相同。带上 $H_{j-1}$ 后，第 $j$ 块 hash 相同就等价于前 $(j+1)B$ 个 token 全部相同，查表从第 0 块往后逐块比，遇到第一个不命中就停。\n后果：前缀中间改 1 个 token（比如开头的时间戳），从那一块往后**全部不命中**。\nextra 里放 LoRA id、多模态输入 hash、租户的 `cache_salt`：salt 不同第 0 块就不同，KV 不跨租户共享。',
    ref: KV,
  }),
  c({
    id: 'ag-kv-size',
    topic: 'KV 复用',
    q: '每个 token 的 KV 字节数 $m_{kv} = 2\\,\\ell\\,h_{kv}\\,d_h\\,b$：2 是 K 和 V 各一份，$\\ell$ 层数，$h_{kv}$ 是 KV head 数，$d_h$ 每个 head 的维度，$b$ 每元素字节数。Llama-3.1-8B：$\\ell = 32$、$h_{kv} = 8$、$d_h = 128$、bf16（$b = 2$）。每 token 多大？一个 5 万 token 的 agent 上下文多大？',
    a: '1. $m_{kv} = 2 \\times 32 \\times 8 \\times 128 \\times 2 = 131{,}072$ B = **128 KiB**。\n2. 5 万 token：$5\\times10^4 \\times 131072 \\approx$ **6.55 GB**。\n含义：一张 80 GB H100 放 8B 权重（约 16 GB）后，假设留 55 GB 给 KV，只能放约 42 万 token，也就是 8 个这样的 session。agent 上下文长，并发数被显存卡死，所以「等工具时 KV 还占不占显存」直接影响吞吐。',
    ref: KV,
  }),
  c({
    id: 'ag-kv-tool-wait',
    topic: 'KV 复用',
    q: 'agent 发出工具调用后请求结束，等工具 $t_{\\text{tool}} = 1.5$ s 后才有下一轮。这条 session 的 KV 有 6.55 GB（8B 模型、$L = 5\\times10^4$ token）。三种处理：留在 HBM；换到 CPU（有效带宽 50 GB/s）；丢掉，下一轮重算。重算 FLOPs $\\approx 2NL + 2\\ell L^2 d$（$N = 8.03\\times10^9$ 参数，$\\ell = 32$ 层，$d = 4096$ 是 hidden 维），有效算力 $4.9\\times10^{14}$ FLOP/s。换出一趟、重算各要多久？该怎么选？',
    a: '1. 换出一趟：$6.55 / 50 = 0.13$ s，来回 0.26 s，可以和别的请求的计算重叠。\n2. 重算：线性层 $2 \\times 8.03\\times10^9 \\times 5\\times10^4 = 8.0\\times10^{14}$；attention $2 \\times 32 \\times (5\\times10^4)^2 \\times 4096 = 6.6\\times10^{14}$；合计 $1.46\\times10^{15}$，除以 $4.9\\times10^{14}$ 约 **3.0 s**，还占着 GPU。\n3. 留在 HBM：不花时间，但 6.55 GB 被占 1.5 s。\n选法：工具短、显存够就留；显存紧就换出；只有工具要跑几分钟或上下文很短时才丢。InferCept 测得把每次工具调用当请求结束、下一轮重算，重算占前向时间 37–40%；Continuum 按工具时长分布给 KV 设 TTL。',
    ref: KV,
  }),
  c({
    id: 'ag-kv-route-random',
    topic: 'KV 复用',
    q: '推理服务有 $R = 8$ 个副本，KV 只在处理过这条 session 的那个副本的显存里。如果请求随机路由，agent 的下一轮回到原副本（能命中前缀）的概率是多少？没命中的轮次要付出什么？',
    a: '概率 $1/R = 1/8 =$ **12.5%**。\n其余 87.5% 的轮次落在新副本上，要把整段上下文重新 prefill：8B 模型 5 万 token 约 3 s（MFU 50% 假设），70B、8 卡、83.5K 上下文约 6.6 s；同时 API 侧也会按写入价重付。所以 agent 服务必须做**亲和路由**：按 session id 或前缀把同一任务的请求发回同一个副本（Manus 自部署 vLLM 时就按 session id 路由）。',
    ref: KV,
  }),
  c({
    id: 'ag-kv-route-balance',
    topic: 'KV 复用',
    q: '纯按 session 固定路由会让个别副本过载（session 长短、轮数差别大，迁走又要重算上下文）。SGLang 的 cache-aware 路由怎么在「命中」和「均衡」之间取舍？',
    a: '1. router 给每个 worker 维护一棵**近似 radix tree**，记录发过去的请求前缀，不用去问 worker 真实的缓存。\n2. 先判断是否失衡：最忙和最闲的 worker 负载差超过绝对阈值、且超过相对倍数。失衡就直接发给**最闲的**。\n3. 不失衡时找前缀匹配最长的 worker，匹配比例超过 `cache_threshold` 就发给它，否则发给缓存空间最多的。\n思路：均衡时追求命中，失衡时先保负载，用一次重算换掉一次排队。SGLang v0.4 报告命中率 20% → 75%、吞吐 1.9 倍（作者自测）。再进一步是集群级 KV 池（Mooncake），跨机器拉 KV 比重算便宜，迁移代价变小。',
    ref: KV,
  }),

  // ---------------- Prompt caching ----------------
  c({
    id: 'ag-cache-mult',
    topic: 'Prompt caching',
    q: '一轮的有效输入单价（相对基础输入价的倍数）$m(h) = h\\,r + (1-h)\\,w$：$h$ 是从缓存读的比例，$r = 0.1$ 是缓存读取倍数，$w = 1.25$ 是写入 5 分钟缓存的倍数（Anthropic，假设没命中的部分都写入缓存）。算 $h = 0$、50%、90% 时的 $m$。为什么写入比普通输入还贵？',
    a: '1. $h = 0$：$m = 1.25$，比不开缓存**还贵 25%**。\n2. $h = 50\\%$：$0.05 + 0.625 = 0.675$。\n3. $h = 90\\%$：$0.09 + 0.125 = 0.215$。\n写入贵：服务方要把这段前缀的 KV 存在显存或内存里，存得越久越贵（1 小时缓存是 2 倍）。读取省掉了 prefill 计算，只收 0.1 倍。所以只用一次的前缀开缓存是亏的，agent 这种每轮重复发送前缀的负载才赚。',
    ref: PC,
  }),
  c({
    id: 'ag-cache-breakeven',
    topic: 'Prompt caching',
    q: '一段前缀写入缓存一次、之后被读 $k$ 次。开缓存总价 $w + kr$，不开是 $1 + k$（单位：这段前缀按基础价算一次）。$r = 0.1$。5 分钟缓存 $w = 1.25$、1 小时缓存 $w = 2$ 时，各要读几次才回本？',
    a: '划算条件 $w + kr < 1 + k$，即\n$$k > \\frac{w - 1}{1 - r}$$\n1. 5 分钟：$k > 0.25 / 0.9 = 0.28$，**读 1 次**就回本。\n2. 1 小时：$k > 1 / 0.9 = 1.11$，要**读 2 次**。\n怎么选：工具都在几秒内返回就用 5 分钟（每次命中免费刷新 TTL）；有超过 5 分钟的工具时，过期一次就要整段重写，1 小时更便宜。',
    ref: PC,
  }),
  c({
    id: 'ag-cache-task',
    topic: 'Prompt caching',
    q: '一个 agent 任务总输入 2,337.5K token，其中第一次出现、需要写入缓存的 83.5K，其余都能从缓存读。基础输入价 $p = 3$ USD / 百万 token，读取 $r = 0.1$ 倍、写入 $w = 1.25$ 倍。开缓存前后输入费各多少？',
    a: '1. 不开：$2.3375 \\times 3 = 7.01$ USD。\n2. 写入：$0.0835 \\times 3 \\times 1.25 = 0.313$ USD。\n3. 读取：$(2337.5 - 83.5)\\text{K} = 2254\\text{K}$，$2.254 \\times 3 \\times 0.1 = 0.676$ USD。\n4. 开缓存合计 **0.989 USD**，是原来的 $0.141$ 倍（约 1/7），正好等于 $m(h) = 0.964 \\times 0.1 + 0.036 \\times 1.25$。\n所以 Manus 把 KV-cache 命中率当作生产 agent 最重要的单一指标。',
    ref: PC,
  }),
  c({
    id: 'ag-cache-ttl',
    topic: 'Prompt caching',
    q: '5 分钟缓存过期一次，下一轮整段前缀 $L$ 要重新写入，多付 $\\Delta = (w - r)\\,L\\,p$。一个任务里有 5 轮的工具超过 5 分钟，这几轮上下文平均 $L = 60\\text{K}$；$w = 1.25$、$r = 0.1$、$p = 3$ USD / 百万 token。过期多付多少？换成 1 小时缓存（写入 2 倍，任务共写入 83.5K token）多付多少？',
    a: '1. 每次过期：$1.15 \\times 0.06 \\times 3 = 0.207$ USD，5 次 **1.04 USD**。全命中时整个任务输入费才 0.99 USD，等于翻倍。\n2. 1 小时缓存：所有写入从 1.25 倍变 2 倍，多付 $0.0835 \\times 0.75 \\times 3 =$ **0.188 USD**，但不再过期。\n结论：有长耗时工具（长测试、等人审批）的任务用 1 小时缓存。',
    ref: PC,
  }),
  c({
    id: 'ag-cache-timestamp',
    topic: 'Prompt caching',
    q: '系统提示开头放了一个精确到秒的时间戳。一个任务总输入 2,337.5K token，基础价 3 USD / 百万 token，缓存写入 1.25 倍。这个任务的输入费是多少？和不开缓存（7.01 USD）比呢？',
    a: '时间戳每轮都变，前缀从第一个 token 起就不同，每轮 $h = 0$，整个输入都按写入价计费：\n$2.3375 \\times 3 \\times 1.25 =$ **8.77 USD**，比不开缓存的 7.01 还贵 25%，比正常命中的 0.99 贵近 9 倍。\n修法：时间戳、请求 id 这类会变的东西放到最后的用户消息里，或者不放。同类问题：JSON key 顺序不稳定、工具列表顺序不稳定、改写历史。',
    ref: PC,
  }),
  c({
    id: 'ag-cache-order',
    topic: 'Prompt caching',
    q: 'Anthropic 的缓存前缀按 `tools` → `system` → `messages` 的顺序拼，改了某一层，这一层和之后的层都失效。据此，agent 的上下文应该怎么排？为什么任务中途不该增删工具？',
    a: '按「多久变一次」从前往后排（见图）：\n1. 工具定义、系统提示放最前，整个任务不改。\n2. 会变的东西（时间戳、随机内容）放最后或不放。\n3. 历史只追加，不回头改。\n4. 序列化确定：JSON key 顺序、工具列表顺序固定。\n增删工具：`tools` 在最前面，改了它三层全失效，下一轮整段上下文按 1.25 倍重写；历史里还有对旧工具的调用，模型会困惑。要限制工具就用解码时屏蔽（logits masking）或 allowed tools 参数，定义本身不动。',
    fig: lines`
      [tools][system][task][t1][res][t2]...
       fixed for the whole task | append only
      change tools -> all layers invalid
      change a message -> rest invalid
    `,
    ref: PC,
  }),

  // ---------------- 上下文管理 ----------------
  c({
    id: 'ag-ctx-tmax',
    topic: '上下文管理',
    q: '窗口要装下这一轮的输入和输出：$L_i + O_{\\max} \\le W$，其中 $L_i = S + (i-1)a$。所以最多能跑 $T_{\\max} = \\lfloor (W - O_{\\max} - S)/a \\rfloor + 1$ 轮。窗口 $W = 200\\text{K}$，预留输出 $O_{\\max} = 32\\text{K}$，固定前缀 $S = 10\\text{K}$。每轮追加 $a = 1.5\\text{K}$ 和 $a = 5\\text{K}$ 时各能跑几轮？',
    a: '1. $a = 1.5\\text{K}$：$\\lfloor 158/1.5 \\rfloor + 1 = 105 + 1 =$ **106** 轮。\n2. $a = 5\\text{K}$：$\\lfloor 158/5 \\rfloor + 1 = 31 + 1 =$ **32** 轮。\n工具结果一变长，能跑的轮数成比例下降，所以单个工具结果要设上限。而且不能等撞墙才管：上下文越长模型越容易漏信息（context rot），每轮的读取费也正比于上下文长度。',
    ref: CTX,
  }),
  c({
    id: 'ag-ctx-truncate',
    topic: '上下文管理',
    q: '一个 $x$ token 的工具结果进入上下文后，之后还要跑 $n$ 轮，它的总成本是写一次加读 $n$ 次：$x(w + nr)$（单位：按基础输入价计的 token；$w = 1.25$ 写入倍数，$r = 0.1$ 读取倍数）。第 10 轮一次 `grep` 返回 80K token，后面还有 40 轮。不截断和截断到 10K 各花多少？',
    a: '1. 不截断：$80\\text{K} \\times (1.25 + 0.1 \\times 40) = 80\\text{K} \\times 5.25 =$ **420K**，按 3 USD / 百万约 1.26 USD；还占掉 200K 窗口的 40%。\n2. 截断到 10K：$10\\text{K} \\times 5.25 =$ **52.5K**，约 0.16 USD。\n截断要可恢复：完整输出写文件，上下文放开头、结尾和路径。截断只作用于新追加的内容，**不改已有前缀，缓存不受影响**，是最便宜的一层。Claude Code 对 MCP 工具输出默认上限 25,000 token。',
    ref: CTX,
  }),
  c({
    id: 'ag-ctx-clear',
    topic: '上下文管理',
    q: '清掉旧工具结果：上下文长 $L$，从位置 $P$ 起清掉 $x$ 个 token。前缀从 $P$ 处变了，之后的部分要重新写入，一次性代价 $w(L - x - P)$；之后每轮少读 $rx$（$w = 1.25$、$r = 0.1$，单位同前）。$L = 100\\text{K}$、$P = 15\\text{K}$，清 $x = 60\\text{K}$ 和 $x = 2\\text{K}$ 各几轮回本？',
    a: '1. $x = 60\\text{K}$：一次性 $1.25 \\times 25\\text{K} = 31.25\\text{K}$，每轮省 $6\\text{K}$，约 **5.2 轮**回本。\n2. $x = 2\\text{K}$：一次性 $1.25 \\times 83\\text{K} = 103.75\\text{K}$，每轮省 $0.2\\text{K}$，要 **519 轮**。\n一次性代价主要由 $P$ 之后剩下多少决定，和清多少关系不大，所以**少清几次、每次清一大块**。这就是 Anthropic context editing 有 `clear_at_least` 参数的原因：清不够这么多就不清。',
    ref: CTX,
  }),
  c({
    id: 'ag-ctx-compact',
    topic: '上下文管理',
    q: '上下文 $L = 100\\text{K}$ 时做一次摘要（compaction）：压缩请求读一遍历史（命中缓存，0.1 倍），输出 2K token 的摘要（输出价是基础输入价的 $\\rho = 5$ 倍），之后上下文变成固定前缀 10K + 摘要 2K，下一轮要按 1.25 倍写入。一次性代价多少？之后每轮省多少？几轮回本？',
    a: '一次性（单位：按基础输入价计的 token）：\n1. 读历史：$0.1 \\times 100\\text{K} = 10\\text{K}$。\n2. 输出摘要：$5 \\times 2\\text{K} = 10\\text{K}$。\n3. 重写新前缀：$1.25 \\times 12\\text{K} = 15\\text{K}$。\n合计 **35K**。每轮省 $0.1 \\times (100 - 12)\\text{K} = 8.8\\text{K}$，约 **4 轮**回本。\n另外两个代价：解码 2K 摘要按 20 ms / token 要 40 s，任务停住（可以后台压缩）；摘要会丢细节，关键内容要落盘、摘要里留路径。',
    ref: CTX,
  }),
  c({
    id: 'ag-ctx-subagent',
    topic: '上下文管理',
    q: '一段探索要 10 轮、每轮追加 3K（共 30K）。交给 sub-agent 做，只回传 1.5K 的结论；主 agent 之后还要跑 $n = 20$ 轮，缓存读取 0.1 倍。主上下文能省多少读取费（单位：按基础输入价计的 token）？sub-agent 的主要收益在哪？',
    a: '主上下文每轮少带 $30\\text{K} - 1.5\\text{K} = 28.5\\text{K}$，20 轮省 $0.1 \\times 28.5\\text{K} \\times 20 =$ **57K**。\n但 sub-agent 自己也要付钱：固定前缀 5K（假设）、10 轮输入合计 185K，大部分命中缓存，约 55K。钱上大致打平。\n真正的收益：\n1. 主上下文干净，不被探索过程的中间结果干扰（减少 context rot）。\n2. 几个 sub-agent 可以并行，时延更短。\n代价是总 token 变多：Anthropic 报告多 agent 约是聊天的 15 倍。',
    ref: CTX,
  }),

  // ---------------- Function calling ----------------
  c({
    id: 'ag-fc-naive-mask',
    topic: 'Function calling',
    q: '约束解码每一步要算一张 token mask：词表里哪些 token 能被 grammar 接受。朴素做法对每个 token 都跑一遍 grammar 自动机，耗时 $t = V c$。$V = 128{,}256$（Llama 3 词表），每个 token 检查 $c = 0.5\\ \\mu\\text{s}$。一步要多久？和一个 15 ms 的解码步比呢？',
    a: '$t = 128256 \\times 0.5\\ \\mu\\text{s} \\approx$ **64 ms**，比一个解码步（15 ms）还慢 4 倍以上。batch 里 64 条序列都要约束时，每步要算 64 张 mask，CPU 跟不上。\n为什么不能只查「可能出现」的 token：一个 token 是一串字节，它能否被接受取决于自动机当前状态和栈，事先不知道该查哪些，只能整个词表都过一遍。XGrammar 的解决办法见下一张卡。',
    ref: FC,
  }),
  c({
    id: 'ag-fc-xgrammar',
    topic: 'Function calling',
    q: 'XGrammar 把词表分成两类：只看自动机当前节点就能判定的 token（预先算好、缓存起来），和要看栈才能判定的 context-dependent token（运行时检查）。Llama-3.1 的 JSON grammar 里后者约 1,134 个，做 context expansion 后剩约 120 个。每个检查 $c = 0.5\\ \\mu\\text{s}$，运行时一步要多久？举一个 context-dependent token 的例子。',
    a: '运行时只查约 120 个：$120 \\times 0.5\\ \\mu\\text{s} =$ **60 µs**，比朴素的 64 ms 快约 1000 倍，和论文实测「JSON 每 token 不到 40 µs」同一量级。mask 在 CPU 上算，和 GPU 前向重叠，端到端几乎没有开销。\n例子：`}` 能不能出现，取决于当前嵌套在第几层对象里，要看栈；字符串内部的普通字母只看「在字符串里」这个节点就能判定，属于可预计算的一类。超过 99% 的 token 是可预计算的。',
    ref: FC,
  }),
  c({
    id: 'ag-fc-fail-prob',
    topic: 'Function calling',
    q: '单次工具调用出错（参数不合法或执行失败）的概率 $p$，一个任务 $K$ 次调用，至少出错一次的概率 $P = 1 - (1-p)^K$。$K = 50$，$p = 1\\%$ 和 $p = 2\\%$ 时各是多少？这说明什么？',
    a: '1. $p = 1\\%$：$1 - 0.99^{50} = 1 - 0.605 =$ **39.5%**。\n2. $p = 2\\%$：$1 - 0.98^{50} = 1 - 0.364 =$ **63.6%**。\n长任务里出错是常态，不能一出错就终止任务。必须有恢复路径：把错误作为工具结果回给模型，让它下一轮改正（见下一张卡）。约束解码能消灭语法和 schema 错误，但消灭不了语义错误（参数值不对、文件不存在）。',
    ref: FC,
  }),
  c({
    id: 'ag-fc-recovery',
    topic: 'Function calling',
    q: '模型发出的工具调用解析失败或参数不合法，agent 框架应该怎么处理？',
    a: '1. 执行前按 JSON schema 校验，不合法就不执行。\n2. 把具体错误作为工具结果回给模型（Anthropic `is_error: true`，MCP `isError: true`），写清错在哪，模型下一轮自己改。MCP 把这类叫 tool execution error；未知工具这类协议错误模型较难自救。\n3. 设重试上限，同一调用连续失败几次就换策略或终止，避免反复犯同一个错。\n4. `stop_reason` 是 `max_tokens` 说明调用被截断，放宽输出上限重来，别当语法错误回给模型。\n5. 失败记录留在上下文里，模型看到后不容易再犯（Manus 的经验）。',
    ref: FC,
  }),
  c({
    id: 'ag-fc-auto-required',
    topic: 'Function calling',
    q: '在 vLLM 里，`tool_choice="auto"` 和 `tool_choice="required"`（或指定工具名）在服务端的实现有什么不同？各自的代价？',
    a: '1. **auto**：模型自由生成，可以先说话、再按自己的格式（如 `<tool_call>{...}</tool_call>`）写调用；服务端用 `--tool-call-parser` 指定的 parser 事后从文本里抽出来。不做约束，文档原话是参数偶尔可能格式错误。\n2. **required / 指定工具**：服务端按工具参数的 JSON schema 做约束解码（structured outputs），保证能解析出调用。代价是输出被 grammar 锁住，模型没有自由文本的地方先推理；而且保证的只是「可解析」，不是「调用得好」。\n折中做法：先自由输出，遇到工具调用的起始标记再启用 grammar。',
    ref: FC,
  }),

  // ---------------- MCP 与工具协议 ----------------
  c({
    id: 'ag-mcp-roles',
    topic: 'MCP / 工具',
    q: 'MCP 里 host、client、server 各是什么？server 能提供哪几类东西？',
    a: '1. **host**：发起连接的 LLM 应用（如 Claude Code、IDE 插件），管理多个 client，决定把哪些工具给模型、调用前是否征求用户同意。\n2. **client**：host 里的连接器，和一个 server 一对一。\n3. **server**：提供能力的服务（GitHub、数据库、浏览器）。\nserver 提供三类：**tools**（模型决定调用的函数）、**resources**（给模型或用户的数据）、**prompts**（给用户用的模板）。消息是 JSON-RPC 2.0，用 `tools/list` 发现工具、`tools/call` 调用。一对一的好处是隔离：每个 server 的权限、生命周期、故障互不影响。',
    ref: MCP,
  }),
  c({
    id: 'ag-mcp-transport',
    topic: 'MCP / 工具',
    q: 'MCP 的两种标准传输 stdio 和 Streamable HTTP 有什么区别？各适合什么场景？',
    a: '1. **stdio**：client 把 server 作为子进程启动，stdin / stdout 上按行分隔的 JSON-RPC，stderr 打日志。适合本地工具（文件、命令），没有网络和鉴权问题；stdout 只能写合法 MCP 消息。\n2. **Streamable HTTP**：server 是独立进程，只有一个端点；client 每条消息一个 POST，回复是一个 JSON 或一个 SSE 流（中间可以推进度）。适合远程、多用户服务，要鉴权、校验 `Origin` 头。\n2025-06-18 版有 `Mcp-Session-Id`，负载均衡要按 session 粘住；2026-07-28 版去掉了协议层 session，每个请求自带版本和能力，HTTP server 可以无状态扩容。',
    ref: MCP,
  }),
  c({
    id: 'ag-mcp-schema-cost',
    topic: 'MCP / 工具',
    q: '工具定义占 $D$ token，放在缓存前缀里跑 $T$ 轮：第一轮写入、之后每轮读取，成本 $D\\,(w + (T-1)\\,r)$（$w = 1.25$ 写入倍数，$r = 0.1$ 读取倍数，单位：按基础输入价计的 token）。接了 58 个工具，$D = 55\\text{K}$，$T = 50$。成本多少？按 3 USD / 百万 token 是多少钱？',
    a: '$55\\text{K} \\times (1.25 + 49 \\times 0.1) = 55\\text{K} \\times 6.15 \\approx$ **338K**，约 **1.01 USD**。\n对比：一个 50 轮任务其余输入全命中缓存时约 0.99 USD，工具定义让输入账单翻倍；不开缓存是 $55\\text{K} \\times 50 = 2.75\\text{M}$，8.25 USD。另外 55K 占了 200K 窗口的 27.5%，工具多了模型也更容易选错。\n办法：tool search（前缀只放约 500 token 的搜索工具，Anthropic 报告总 token 少 85%）、合并工具、把多步调用写成代码。',
    ref: MCP,
  }),
  c({
    id: 'ag-mcp-dynamic-tools',
    topic: 'MCP / 工具',
    q: '为什么不能每轮按当前步骤语义检索 top-k 个工具放进 prompt？tool search 为什么就没这个问题？',
    a: '工具定义在缓存前缀的**最前面**。每轮换一次工具集，整个前缀失效，每轮都按 1.25 倍写入价重付全部上下文，比不开缓存还贵；历史里调用过的工具突然不见了，模型也会出错。\ntool search 不破坏缓存：延迟加载的工具一开始根本不在前缀里，模型搜到之后，工具定义是**追加**在后面的，前面一个 token 都没变。\n其他合规做法：任务开始时选定工具集，整个任务不变；或保留全部工具，用解码屏蔽 / allowed tools 限制本轮可选的。',
    ref: MCP,
  }),
  c({
    id: 'ag-mcp-parallel',
    topic: 'MCP / 工具',
    q: '$K$ 个互不依赖的工具调用，串行要 $\\sum_k t_k + (K-1)\\,t_{\\text{turn}}$（每多一轮模型调用花 $t_{\\text{turn}}$），并行一轮发出只要 $\\max_k t_k$。三个工具耗时 2、1.5、3 s，多一轮模型调用 $t_{\\text{turn}} = 0.6 + 100 \\times 0.02 = 2.6$ s（TTFT 加 100 个输出 token）。串行、并行各多久？',
    a: '1. 串行：$6.5 + 2 \\times 2.6 =$ **11.7 s**。\n2. 并行：$\\max(2, 1.5, 3) =$ **3 s**。\n还省钱：少的 2 轮每轮都要读一遍上下文。\n限制：\n1. 只有互不依赖、没有冲突副作用的调用能并行（两个写同一文件的不行）。\n2. 工具服务有并发上限。\n3. 有依赖的要先规划成 DAG 再按依赖调度：LLMCompiler 对比 ReAct 时延最多快 3.7 倍（作者自测）。',
    ref: MCP,
  }),

  // ---------------- Agent 延迟 ----------------
  c({
    id: 'ag-lat-prefill',
    topic: 'Agent 延迟',
    q: 'prefill FLOPs：命中缓存时只算新 token，$2Nx + 4\\ell xLd$；全部失效时 $2NL + 2\\ell L^2 d$。$N = 70.6\\times10^9$ 参数，$\\ell = 80$ 层，$d = 8192$ hidden 维，已有上下文 $L = 83.5\\text{K}$，新追加 $x = 1.5\\text{K}$，8 张 H100 有效算力 $F = 3.17\\times10^{15}$ FLOP/s（MFU 40% 假设）。两种情况的 prefill 各要多久？',
    a: '1. 全失效：线性 $2 \\times 70.6\\times10^9 \\times 83500 = 1.18\\times10^{16}$；attention $2 \\times 80 \\times 83500^2 \\times 8192 = 9.14\\times10^{15}$；合计 $2.09\\times10^{16}$，**约 6.6 s**。\n2. 命中：线性 $2.12\\times10^{14}$；attention $4 \\times 80 \\times 1500 \\times 83500 \\times 8192 = 3.28\\times10^{14}$；合计 $5.4\\times10^{14}$，**约 0.17 s**。\n差 39 倍。attention 项：每个新 token 要和整个上下文做 $QK^\\top$ 和 $PV$，每层 $4Ld$；全部重算时 causal 平均看一半，是 $2L^2d$。所以缓存失效首先是时延事故。',
    ref: LAT,
  }),
  c({
    id: 'ag-lat-output',
    topic: 'Agent 延迟',
    q: '一个 50 轮的 agent 任务，每轮 TTFT 0.6 s、输出 300 token、TPOT 20 ms（每个输出 token 的解码时间）、工具 1.5 s，共 405 s。如果让模型每轮少说 150 token（不复述计划），任务变成多久？为什么先砍输出？',
    a: '每轮省 $150 \\times 0.02 = 3$ s，50 轮省 **150 s**，任务从 405 s 变成 **255 s**（少 37%）。\n先砍输出是因为解码是大头：每轮 6 s 的解码占 74%，TTFT 只占 7%。办法：提示词要求直接调用工具、不复述；按步骤难度给思考预算或换小模型；用上下文做草稿的投机解码（agent 输出里大量内容可以从上下文抄，接受率高）。',
    ref: LAT,
  }),
  c({
    id: 'ag-lat-spec-tool',
    topic: 'Agent 延迟',
    q: '投机执行工具：模型还在生成时，用快的预测器猜下一个工具调用并提前执行；猜对（概率 $p$）就直接用结果，工具和生成重叠，每轮期望省 $\\Delta t = p \\cdot \\min(t_{\\text{tool}}, t_{\\text{gen}})$。$t_{\\text{gen}} = 6.6$ s（模型生成这一轮的时间），$t_{\\text{tool}} = 1.5$ s，$p = 0.5$。每轮省多少？哪些工具不能投机？',
    a: '$\\Delta t = 0.5 \\times \\min(1.5, 6.6) =$ **0.75 s**，占一轮 8.1 s 的约 9%。工具越慢、越可预测省得越多：Speculative Actions 报告预测准确率最高 55%、时延最多降 20%；PASTE 按调用模式预测，平均任务完成时间降 43.5%（作者自测）。\n不能投机的：有副作用的调用（写文件、发消息、付款、改数据库），猜错无法撤销；除非在隔离的环境快照上执行。猜错的只读调用也要付工具本身的成本（API 费、限流配额）。',
    ref: LAT,
  }),
  c({
    id: 'ag-lat-early-start',
    topic: 'Agent 延迟',
    q: '一轮输出里有 3 个并行工具调用，每个 100 token，TPOT 20 ms，按输出顺序工具耗时 3、2、1.5 s。方案 A：等整段输出完再一起执行；方案 B：流式解析，某个调用的 JSON 一闭合就开始执行。这一轮各多久？',
    a: '三个调用在 2、4、6 s 闭合（每个 $100 \\times 0.02 = 2$ s）。\n1. A：$6 + \\max(3, 2, 1.5) =$ **9 s**。\n2. B：分别在 $2 + 3 = 5$、$4 + 2 = 6$、$6 + 1.5 = 7.5$ s 跑完，这一轮 **7.5 s**。\n第 $k$ 个调用提前 $(K-k)\\,c\\,t_{\\text{tpot}}$ 开始（$c$ 是每个调用的 token 数），越早输出提前越多，所以**慢的工具应该先输出**。流式不只是给用户看进度。',
    ref: LAT,
  }),
]
