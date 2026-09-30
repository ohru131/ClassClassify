import { base64ToArrayBuffer } from './base64'
import { buildParseMessages } from './parse-messages'
import type { AppLanguage } from './i18n'
import { SAMPLE_FILES } from './samples.generated'
import { parseWorkbook, type ColumnKind, type Problem } from './solver'

export const samplesFor = (lang: AppLanguage) => SAMPLE_FILES[lang].map(({ id, label }) => ({ id, label }))

/** その言語のサンプル（その国らしい氏名・項目名）を読む。label はサンプルの表示名 */
export function loadSample(lang: AppLanguage, id: string): { problem: Problem; label: string } {
  const s = SAMPLE_FILES[lang].find((x) => x.id === id)
  if (!s) throw new Error(`sample ${id} not found`)
  return { problem: parseWorkbook(base64ToArrayBuffer(s.base64), buildParseMessages(lang)), label: s.label }
}

// 新しい名簿・ひな形に入れる項目の例（docs/i18n-glossary.md）
const STARTER_COLUMNS: Record<AppLanguage, { name: string; kind: ColumnKind }[]> = {
  ja: [{ name: '性別', kind: 'category' }, { name: '学力', kind: 'category' }, { name: '学習支援', kind: 'flag' }],
  en: [{ name: 'Gender', kind: 'category' }, { name: 'Academics', kind: 'category' }, { name: 'Learning support', kind: 'flag' }],
  ko: [{ name: '성별', kind: 'category' }, { name: '학업', kind: 'category' }, { name: '학습 지원', kind: 'flag' }],
  es: [{ name: 'Género', kind: 'category' }, { name: 'Desempeño académico', kind: 'category' }, { name: 'Apoyo en el aprendizaje', kind: 'flag' }],
  de: [{ name: 'Geschlecht', kind: 'category' }, { name: 'Leistung', kind: 'category' }, { name: 'Lernförderung', kind: 'flag' }],
  'pt-BR': [{ name: 'Gênero', kind: 'category' }, { name: 'Desempenho', kind: 'category' }, { name: 'Apoio à aprendizagem', kind: 'flag' }],
}

/** 新しい名簿（空）。項目の例と、生徒 count 名（NO だけ）から始める */
export function blankProblem(lang: AppLanguage, count = 3): Problem {
  const cols = STARTER_COLUMNS[lang]
  return {
    students: Array.from({ length: count }, (_, i) => ({ no: i + 1, name: '', values: Object.fromEntries(cols.map((c) => [c.name, ''])) })),
    columns: cols.map((c) => ({ name: c.name, weight: 1, kind: c.kind, levels: [], enabled: false })),
    numClasses: 2,
    maxPerClass: null,
    wantedGroups: [],
    unwantedGroups: [],
    warnings: [],
  }
}
