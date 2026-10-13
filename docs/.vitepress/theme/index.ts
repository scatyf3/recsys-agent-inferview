import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import StackFigure from './components/StackFigure.vue'
import Flashcards from './components/Flashcards.vue'
import StackPageExtras from './components/StackPageExtras.vue'
import ReaderNotes from './components/ReaderNotes.vue'
import SyncControl from './components/SyncControl.vue'
import { registerServiceWorker } from './pwa'
import 'katex/dist/katex.min.css'
import './custom.css'

export default {
  extends: DefaultTheme,
  // 全站挂载读者划词高亮 / 批注和导航栏上的同步状态；/stack/<id> 页额外加上所在层和相关文章（其他页面渲染为空）
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(ReaderNotes),
      'nav-bar-content-after': () => h(SyncControl),
      'doc-before': () => h(StackPageExtras, { where: 'before' }),
      'doc-after': () => h(StackPageExtras, { where: 'after' }),
    }),
  enhanceApp({ app }) {
    registerServiceWorker()
    app.component('StackFigure', StackFigure)
    app.component('Flashcards', Flashcards)
  },
} satisfies Theme
