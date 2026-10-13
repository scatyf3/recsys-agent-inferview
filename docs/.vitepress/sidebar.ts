import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import matter from 'gray-matter'
import type { DefaultTheme } from 'vitepress'
import { domains } from './domains'
import { stacks } from './layers'
import { dirStack, sections } from './sections'

const docsRoot = fileURLToPath(new URL('..', import.meta.url))

interface Entry {
  /** 相对 docs/ 的路径，不带 .md，如 recsys/embedding-tables */
  key: string
  dir: string
  title: string
  order: number
  section?: string
  /** 这个目录对应那张图上的挂载点 */
  entries: (string | number)[]
}

function readDomain(dir: string): Entry[] {
  const abs = path.join(docsRoot, dir)
  if (!fs.existsSync(abs)) return []
  const stack = dirStack[dir]
  return fs
    .readdirSync(abs)
    .filter((f) => f.endsWith('.md') && f !== 'index.md')
    .map((f) => {
      const key = `${dir}/${f.replace(/\.md$/, '')}`
      const fm = matter(fs.readFileSync(path.join(abs, f), 'utf8')).data
      return {
        key,
        dir,
        title: fm.title ?? key,
        order: fm.order ?? 99,
        section: fm.section,
        entries: stack && Array.isArray(fm[stack]) ? fm[stack] : [],
      }
    })
}

/** 文章归哪一组，规则见 sections.ts */
function sectionOf(e: Entry): string | undefined {
  if (e.section) {
    if (!sections.some((s) => s.id === e.section)) throw new Error(`[sidebar] ${e.key} 的 section "${e.section}" 不在 sections.ts 里`)
    return e.section
  }
  const stack = dirStack[e.dir]
  const own = sections.filter((s) => s.stack === stack)
  const first = e.entries[0]
  const layer =
    typeof first === 'number' ? first : stacks[stack]?.layers.find((l) => l.topics.some((t) => t.id === first))?.id
  return (layer !== undefined ? own.find((s) => s.layers.includes(layer)) : undefined)?.id ?? own[0]?.id
}

/** 按分层图生成 sidebar；空组不显示。 */
export function buildSidebar(): DefaultTheme.Sidebar {
  const entries = domains.flatMap((d) => readDomain(d.dir))
  return sections
    .map((s) => {
      const items = entries
        .filter((e) => sectionOf(e) === s.id)
        .sort((a, b) => a.order - b.order || a.key.localeCompare(b.key))
        .map((e) => ({ text: e.title, link: `/${e.key}` }))
      return { text: s.label, link: s.link, collapsed: false, items }
    })
    .filter((g) => g.items.length)
}
