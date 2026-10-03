import { base64ToArrayBuffer } from './base64'
import { buildParseMessages } from './parse-messages'
import type { AppLanguage } from './i18n'
import { SAMPLE_FILES } from './samples.generated'
import { parseWorkbook, type ColumnKind, type Problem } from './solver'

export const samplesFor = (lang: AppLanguage) => SAMPLE_FILES[lang].map(({ id, label }) => ({ id, label }))

/** サンプルの .xlsx そのもの（base64。共有して書き換え・読み込みに使う） */
export function sampleFile(lang: AppLanguage, id: string): { base64: string; label: string } {
  const s = SAMPLE_FILES[lang].find((x) => x.id === id)
  if (!s) throw new Error(`sample ${id} not found`)
  return { base64: s.base64, label: s.label }
}

/** その言語のサンプル（その国らしい氏名・項目名）を読む。label はサンプルの表示名 */
export function loadSample(lang: AppLanguage, id: string): { problem: Problem; label: string } {
  const s = SAMPLE_FILES[lang].find((x) => x.id === id)
  if (!s) throw new Error(`sample ${id} not found`)
  return { problem: parseWorkbook(base64ToArrayBuffer(s.base64), buildParseMessages(lang)), label: s.label }
}

// 新しい名簿・ひな形に入れる項目の例（docs/i18n-glossary.md）
// 性別はリストの選択肢を最初から入れておく（♂ ♀ ？）
const G = (name: string, unknown = '?') => ({ name, kind: 'category' as const, levels: ['♂', '♀', unknown] })
const STARTER_COLUMNS: Record<AppLanguage, { name: string; kind: ColumnKind; levels?: string[] }[]> = {
  ja: [G('性別', '？'), { name: '学力', kind: 'degree' }, { name: '学習支援', kind: 'flag' }],
  en: [G('Gender'), { name: 'Academics', kind: 'degree' }, { name: 'Learning support', kind: 'flag' }],
  ko: [G('성별'), { name: '학업', kind: 'degree' }, { name: '학습 지원', kind: 'flag' }],
  es: [G('Género'), { name: 'Desempeño académico', kind: 'degree' }, { name: 'Apoyo en el aprendizaje', kind: 'flag' }],
  de: [G('Geschlecht'), { name: 'Leistung', kind: 'degree' }, { name: 'Lernförderung', kind: 'flag' }],
  'pt-BR': [G('Gênero'), { name: 'Desempenho', kind: 'degree' }, { name: 'Apoio à aprendizagem', kind: 'flag' }],
}

/** 新しい名簿（空）。項目の例と、生徒 count 名（NO だけ）から始める */
export function blankProblem(lang: AppLanguage, count = 3): Problem {
  const cols = STARTER_COLUMNS[lang]
  return {
    students: Array.from({ length: count }, (_, i) => ({ no: i + 1, name: '', values: Object.fromEntries(cols.map((c) => [c.name, ''])) })),
    columns: cols.map((c) => ({ name: c.name, weight: 1, kind: c.kind, levels: c.levels ?? [], enabled: !!c.levels?.length })),
    numClasses: 2,
    maxPerClass: null,
    wantedGroups: [],
    unwantedGroups: [],
    warnings: [],
  }
}
