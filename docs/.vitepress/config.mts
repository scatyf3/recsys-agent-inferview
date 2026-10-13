import { defineConfig } from 'vitepress'
import { fileURLToPath } from 'node:url'
import { buildSidebar } from './sidebar'
import { markPlugin } from './markdown/mark'
import { dataStore } from './dataStore'

const BASE = '/recsys-agent-inferview/'

export default defineConfig({
  lang: 'zh-CN',
  title: '搜推 & Agent Infra Inferview',
  description: '搜推 infra 与 Agent infra 面试知识库：embedding、并行、召回排序、生成式推荐；上下文、工具、沙箱、编排、Agentic RL',
  base: BASE,
  // 手机上「添加到主屏幕」后像 app 一样全屏打开、离线可用（manifest 和 sw.js 在 docs/public）
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: `${BASE}icons/favicon-64.png` }],
    ['link', { rel: 'manifest', href: `${BASE}manifest.webmanifest` }],
    ['link', { rel: 'apple-touch-icon', href: `${BASE}icons/apple-touch-icon.png` }],
    ['meta', { name: 'theme-color', content: '#ffffff', media: '(prefers-color-scheme: light)' }],
    ['meta', { name: 'theme-color', content: '#1b1b1f', media: '(prefers-color-scheme: dark)' }],
    ['meta', { name: 'mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-title', content: 'RecAgent' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'default' }],
  ],
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: false,
  markdown: {
    math: true,
    lineNumbers: false,
    config: (md) => {
      md.use(markPlugin)
    },
  },
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '搜推', link: '/recsys/' },
      { text: 'Agent', link: '/agent/' },
      { text: '闪卡', link: '/flashcards' },
    ],
    sidebar: buildSidebar(),
    search: { provider: 'local', options: { detailedView: true } },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    socialLinks: [{ icon: 'github', link: 'https://github.com/scatyf3/recsys-agent-inferview' }],
  },
  vite: {
    plugins: [dataStore()],
    resolve: {
      alias: {
        '@lib': fileURLToPath(new URL('../../src/lib', import.meta.url)),
        '@data': fileURLToPath(new URL('../../src/data', import.meta.url)),
      },
    },
  },
})
