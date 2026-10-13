/** 领域元数据：目录名 → 展示信息。这是唯一需要手工维护的结构性元数据。 */
export interface Domain {
  dir: string
  label: string
  short: string
  color: string
  order: number
}

export const domains: Domain[] = [
  { dir: 'recsys', label: '搜推 infra', short: '搜推', color: '#0e7490', order: 1 },
  { dir: 'agent', label: 'Agent infra', short: 'Agent', color: '#7c3aed', order: 2 },
]

export const domainByDir = Object.fromEntries(domains.map((d) => [d.dir, d])) as Record<string, Domain>
