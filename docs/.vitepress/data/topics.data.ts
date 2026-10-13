import { createContentLoader } from 'vitepress'
// data loader 在 vite 别名之外单独打包，这里只能走相对路径
import { isFamiliarity } from '../../../src/lib/fam'
import { STACK_IDS, isValidEntry, type StackId } from '../layers'

export type Status = 'todo' | 'draft' | 'reviewed'
export const STATUSES: Status[] = ['todo', 'draft', 'reviewed']

export interface Topic extends Record<StackId, (string | number)[]> {
  url: string
  domain: string
  slug: string
  title: string
  status: Status
  tags: string[]
  difficulty: number
  related: string[]
  order: number
  isIndex: boolean
  /** 熟练度 0–4（0 最熟），null = 未评；语义见 src/lib/fam.ts */
  familiarity: number | null
}

declare const data: Topic[]
export { data }

export default createContentLoader('*/*.md', {
  transform(raw): Topic[] {
    return raw
      // docs/stack/ 是分层图小主题的介绍页，由 stack.data.ts 收集，不算知识库文章
      .filter((p) => !p.url.startsWith('/stack/'))
      .map((p) => {
        // url 形如 /recsys/embedding-tables 或 /recsys/
        const parts = p.url.split('/').filter(Boolean)
        const domain = parts[0]
        const slug = parts[1] ?? 'index'
        const isIndex = slug === 'index'
        const fm = p.frontmatter
        const status: Status = fm.status ?? 'todo'
        if (!isIndex && !STATUSES.includes(status)) {
          throw new Error(`[topics] invalid status "${status}" in ${p.url} (expected ${STATUSES.join(' | ')})`)
        }
        if (!fm.title) throw new Error(`[topics] missing title in ${p.url}`)
        if (!isFamiliarity(fm.familiarity)) {
          throw new Error(`[topics] invalid familiarity "${fm.familiarity}" in ${p.url} (expected 0 | 1 | 1.5 | 2 | 3 | 3.5 | 4)`)
        }
        const mounts = {} as Record<StackId, (string | number)[]>
        for (const id of STACK_IDS) {
          const list: (string | number)[] = fm[id] ?? []
          for (const s of list) {
            if (!isValidEntry(id, s)) throw new Error(`[topics] unknown ${id} entry "${s}" in ${p.url} (expected a layer 0–8 or a topic id from layers.ts)`)
          }
          mounts[id] = list
        }
        return {
          url: p.url,
          domain,
          slug,
          title: fm.title,
          status,
          tags: fm.tags ?? [],
          difficulty: fm.difficulty ?? 2,
          related: fm.related ?? [],
          order: fm.order ?? 99,
          isIndex,
          familiarity: fm.familiarity ?? null,
          ...mounts,
        }
      })
      .sort((a, b) => a.domain.localeCompare(b.domain) || a.order - b.order || a.slug.localeCompare(b.slug))
  },
})
