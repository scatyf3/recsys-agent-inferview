/**
 * 两张分层图：搜推栈（frontmatter `recsys`）和 Agent 栈（frontmatter `agent`），都是 0 在最底、8 在最顶，
 * 每层再拆成几个小主题。文章用 frontmatter 挂上来：
 *   - 字符串 = 小主题 id（搜推 rs- 开头，Agent ag- 开头），文章出现在那个格子里
 *   - 数字   = 整层，用于讲一整层的综述文章
 * 一篇都没挂的小主题在图上显示成占位格。
 */
export interface StackTopic {
  id: string
  name: string
  /** 同层内的分组，如「训练侧 / 服务侧」 */
  group?: string
}

export interface Layer {
  id: number
  name: string
  /** 论文风格插图上用的英文名 */
  en: string
  short: string
  scope: string
  tag?: string
  /** 图上的强调色：从底层的冷色渐变到顶层的暖色 */
  color: string
  topics: StackTopic[]
}

export const recsysLayers: Layer[] = [
  {
    id: 0, name: '硬件与存储', en: 'Hardware & Storage', short: '硬件', color: '#0ea5e9',
    scope: '大表放在哪、样本从哪来',
    topics: [
      { id: 'rs-hw-mem', name: 'HBM / DRAM / SSD 层次' },
      { id: 'rs-hw-net', name: '网络与 all-to-all 带宽' },
      { id: 'rs-hw-cpu', name: 'CPU 侧特征处理' },
    ],
  },
  {
    id: 1, name: 'Embedding 存储', en: 'Sparse Parameters', short: 'Embedding', color: '#06b6d4',
    scope: 'TB 级稀疏参数怎么存',
    topics: [
      { id: 'rs-e-table', name: 'embedding 表与哈希' },
      { id: 'rs-e-dynamic', name: '动态 embedding / 准入淘汰' },
      { id: 'rs-e-cache', name: '多级缓存' },
      { id: 'rs-e-compress', name: '压缩（量化 / 混合维度）' },
    ],
  },
  {
    id: 2, name: '算子', en: 'Kernels', short: 'Kernel', color: '#14b8a6',
    scope: '稀疏查表和特征交叉',
    topics: [
      { id: 'rs-k-tbe', name: 'EmbeddingBag / TBE' },
      { id: 'rs-k-interact', name: '特征交叉 / dense 部分' },
      { id: 'rs-k-seq', name: '用户长序列 / jagged tensor' },
    ],
  },
  {
    id: 3, name: '训练并行', en: 'Training Parallelism', short: '并行', color: '#22c55e',
    scope: '稀疏模型并行 + 稠密数据并行',
    topics: [
      { id: 'rs-p-ps', name: 'Parameter Server' },
      { id: 'rs-p-shard', name: 'embedding 分片' },
      { id: 'rs-p-a2a', name: 'all-to-all 与流水' },
      { id: 'rs-p-dense', name: 'dense 部分 DP' },
    ],
  },
  {
    id: 4, name: '样本与特征', en: 'Samples & Features', short: '样本', color: '#84cc16',
    scope: '从曝光日志到训练样本',
    topics: [
      { id: 'rs-d-join', name: '样本拼接 / 延迟标签' },
      { id: 'rs-d-store', name: 'feature store / 一致性' },
      { id: 'rs-d-neg', name: '负采样' },
      { id: 'rs-d-stream', name: '流式管线' },
    ],
  },
  {
    id: 5, name: '模型更新', en: 'Model Updates', short: '更新', color: '#eab308',
    scope: '模型多新鲜、怎么上线',
    topics: [
      { id: 'rs-u-online', name: '在线 / 增量训练' },
      { id: 'rs-u-ckpt', name: 'checkpoint 与增量导出' },
      { id: 'rs-u-sync', name: '参数同步到线上' },
      { id: 'rs-u-ab', name: '指标 / A/B / 回滚' },
    ],
  },
  {
    id: 6, name: '召回', en: 'Retrieval', short: '召回', color: '#f97316',
    scope: '从亿级候选里捞出几千',
    topics: [
      { id: 'rs-r-tower', name: '双塔' },
      { id: 'rs-r-ann', name: 'ANN 索引' },
      { id: 'rs-r-gpu', name: 'GPU 检索 / 向量库' },
    ],
  },
  {
    id: 7, name: '排序服务', en: 'Ranking Serving', short: '排序', color: '#ef4444',
    scope: '几十毫秒内给几千个候选打分',
    topics: [
      { id: 'rs-s-pipeline', name: '多阶段漏斗 / 延迟预算' },
      { id: 'rs-s-feature', name: '在线特征获取' },
      { id: 'rs-s-infer', name: '排序推理 / batching' },
      { id: 'rs-s-cache', name: '缓存 / 降级' },
    ],
  },
  {
    id: 8, name: '生成式推荐', en: 'Generative Rec', short: '生成式', color: '#ec4899', tag: '新范式',
    scope: '把推荐写成序列生成',
    topics: [
      { id: 'rs-g-sid', name: 'semantic ID' },
      { id: 'rs-g-hstu', name: 'HSTU / scaling' },
      { id: 'rs-g-serve', name: '生成式推荐的推理' },
      { id: 'rs-g-llm', name: 'LLM 用于搜推' },
    ],
  },
]

export const agentLayers: Layer[] = [
  {
    id: 0, name: '模型推理', en: 'Model Serving', short: '推理', color: '#0ea5e9',
    scope: '多轮、长前缀、带工具调用的推理',
    topics: [
      { id: 'ag-m-prefix', name: 'prefix cache / KV 复用' },
      { id: 'ag-m-ttft', name: '延迟：TTFT 与长 prefill' },
      { id: 'ag-m-decode', name: '约束解码 / 结构化输出' },
      { id: 'ag-m-route', name: 'session 亲和路由' },
    ],
  },
  {
    id: 1, name: '上下文', en: 'Context', short: '上下文', color: '#06b6d4',
    scope: '窗口里放什么、花多少钱',
    topics: [
      { id: 'ag-c-budget', name: 'token 预算' },
      { id: 'ag-c-cache', name: 'prompt caching' },
      { id: 'ag-c-compact', name: '压缩 / 摘要' },
    ],
  },
  {
    id: 2, name: '记忆与检索', en: 'Memory & RAG', short: '记忆', color: '#14b8a6',
    scope: '窗口外的知识怎么取回来',
    topics: [
      { id: 'ag-r-rag', name: 'RAG 管线' },
      { id: 'ag-r-vec', name: '向量检索 / rerank' },
      { id: 'ag-r-mem', name: '长期记忆' },
    ],
  },
  {
    id: 3, name: '工具', en: 'Tools', short: '工具', color: '#22c55e',
    scope: '模型怎么调外部能力',
    topics: [
      { id: 'ag-t-call', name: 'function calling' },
      { id: 'ag-t-mcp', name: 'MCP / 工具协议' },
      { id: 'ag-t-parallel', name: '并行工具调用' },
    ],
  },
  {
    id: 4, name: '执行环境', en: 'Sandbox', short: '沙箱', color: '#84cc16',
    scope: '代码和命令在哪跑',
    topics: [
      { id: 'ag-x-sandbox', name: '隔离：容器 / microVM' },
      { id: 'ag-x-cold', name: '冷启动 / 快照' },
      { id: 'ag-x-sec', name: '安全 / 权限' },
    ],
  },
  {
    id: 5, name: '编排', en: 'Orchestration', short: '编排', color: '#eab308',
    scope: '一个任务怎么走完',
    topics: [
      { id: 'ag-o-loop', name: 'agent loop' },
      { id: 'ag-o-multi', name: '多 agent / sub-agent' },
      { id: 'ag-o-durable', name: '持久执行 / 重试' },
    ],
  },
  {
    id: 6, name: '网关', en: 'Gateway', short: '网关', color: '#f97316',
    scope: '多租户、多模型、有预算',
    topics: [
      { id: 'ag-g-rate', name: '限流 / 配额' },
      { id: 'ag-g-model', name: '模型路由 / 降级' },
      { id: 'ag-g-cost', name: '成本账' },
    ],
  },
  {
    id: 7, name: '评测与可观测', en: 'Eval & Observability', short: '评测', color: '#ef4444',
    scope: '做得对不对、慢在哪',
    topics: [
      { id: 'ag-e-trace', name: 'tracing' },
      { id: 'ag-e-bench', name: 'benchmark / pass@k' },
      { id: 'ag-e-online', name: '线上评估 / 回放' },
    ],
  },
  {
    id: 8, name: 'Agentic RL', en: 'Agentic RL', short: 'RL', color: '#ec4899',
    scope: '在环境里训 agent',
    topics: [
      { id: 'ag-rl-env', name: '环境规模化' },
      { id: 'ag-rl-rollout', name: '多轮 rollout / 异步' },
      { id: 'ag-rl-reward', name: 'verifier / reward' },
    ],
  },
]

/** 两张分层图：搜推栈挂在 frontmatter 的 `recsys`，Agent 栈挂在 `agent` */
export type StackId = 'recsys' | 'agent'
export interface StackDef {
  id: StackId
  name: string
  /** 图所在的页面，独立页的面包屑链回这里 */
  home: string
  layers: Layer[]
  /** 整层综述的 key，也是 docs/stack/ 下层总述页的文件名 */
  layerKey: (n: number) => string
  /** 层号前缀：搜推栈 R4，Agent 栈 A4 */
  prefix: string
}
export const stacks: Record<StackId, StackDef> = {
  recsys: { id: 'recsys', name: '搜推栈', home: '/recsys/', layers: recsysLayers, layerKey: (n) => `rs-layer-${n}`, prefix: 'R' },
  agent: { id: 'agent', name: 'Agent 栈', home: '/agent/', layers: agentLayers, layerKey: (n) => `ag-layer-${n}`, prefix: 'A' },
}
export const STACK_IDS = Object.keys(stacks) as StackId[]

/** frontmatter 里一个取值是否合法：这张图的层号，或这张图上的小主题 id */
export function isValidEntry(stack: StackId, s: unknown): boolean {
  const def = stacks[stack]
  if (typeof s === 'number') return def.layers.some((l) => l.id === s)
  return typeof s === 'string' && def.layers.some((l) => l.topics.some((t) => t.id === s))
}

/**
 * 分层图上的一个格子 key（小主题 id，或 rs-layer-N / ag-layer-N）→ 它在哪张图、哪一层。
 * 两张图的小主题 id 不重名，所以 key 在全站唯一。
 */
export function locateKey(key: string): { stack: StackDef; layer: Layer; topic?: StackTopic } | undefined {
  for (const s of Object.values(stacks)) {
    for (const l of s.layers) {
      if (s.layerKey(l.id) === key) return { stack: s, layer: l }
      const t = l.topics.find((x) => x.id === key)
      if (t) return { stack: s, layer: l, topic: t }
    }
  }
  return undefined
}

/** 一篇文章的 frontmatter（recsys / agent 两个字段）→ 它挂在哪些格子上 */
export function keysOfArticle(t: Record<StackId, (string | number)[]>): string[] {
  return STACK_IDS.flatMap((id) => t[id].map((s) => (typeof s === 'number' ? stacks[id].layerKey(s) : s)))
}
