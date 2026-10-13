/**
 * 侧边栏分组：每个目录按分层图拆成几组。只影响侧边栏，文件不动、URL 不变。
 *
 * 归组规则（见 sidebar.ts）：
 *   1. frontmatter 写了 `section` 的，按它来（全景、系统设计这类跨层的文章）；
 *   2. 其余按这个目录对应的那张图（recsys / agent 字段）的第一项所在的层，落到 layers 覆盖它的组；
 *   3. 都没有的放进这个目录的第一组。
 * 组内按 frontmatter `order`，再按文件名。
 */
import type { StackId } from './layers'

export interface Section {
  id: string
  label: string
  stack: StackId
  /** 这一组收哪几层的文章 */
  layers: number[]
  /** 组标题的链接 */
  link?: string
}

export const sections: Section[] = [
  { id: 'rs-overview', label: '搜推：全景与系统设计', stack: 'recsys', layers: [], link: '/recsys/' },
  { id: 'rs-train', label: '搜推训练：embedding、并行、样本、更新', stack: 'recsys', layers: [0, 1, 2, 3, 4, 5] },
  { id: 'rs-serve', label: '搜推服务：召回、排序、生成式', stack: 'recsys', layers: [6, 7, 8] },
  { id: 'ag-overview', label: 'Agent：全景与系统设计', stack: 'agent', layers: [], link: '/agent/' },
  { id: 'ag-infer', label: 'Agent 推理、上下文与工具', stack: 'agent', layers: [0, 1, 2, 3] },
  { id: 'ag-run', label: 'Agent 运行时：沙箱、编排、网关、评测', stack: 'agent', layers: [4, 5, 6, 7] },
  { id: 'ag-rl', label: 'Agentic RL', stack: 'agent', layers: [8] },
]

/** 目录 → 它对应哪张图 */
export const dirStack: Record<string, StackId> = { recsys: 'recsys', agent: 'agent' }
