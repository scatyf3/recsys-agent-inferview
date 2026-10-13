// 闪卡：搜推（recsys）和 Agent（agent）两个牌组，按文章分到四个文件里写，这里合起来。
//
// id 是复习记录的 key（src/data/flashcard-progress.json），改题面可以，改 id 会丢掉这张卡的记录；
// 要删卡就把 id 加进 retiredIds。
// a 支持行内 `code`、**加粗**、$公式$ 和 $$单独一行的公式$$（KaTeX），\n 换行；
// fig 是答案下面的示意图（等宽字符画），code 是代码块；ref 是出处（站内链接，不含 base）。

import type { Card } from '@lib/flashcards'
import { recsysTrainCards } from './cards-recsys-train'
import { recsysServeCards } from './cards-recsys-serve'
import { agentInferCards } from './cards-agent-infer'
import { agentRunCards } from './cards-agent-run'

export const retiredIds: string[] = []

export const cards: Card[] = [...recsysTrainCards, ...recsysServeCards, ...agentInferCards, ...agentRunCards]
