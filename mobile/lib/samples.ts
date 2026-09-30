import { base64ToArrayBuffer } from './base64'
import { SAMPLE_FILES } from './samples.generated'
import { parseWorkbook, type Problem } from './solver'

export const SAMPLES = SAMPLE_FILES.map(({ id, label }) => ({ id, label }))

export function loadSample(id: string): { problem: Problem; name: string } {
  const s = SAMPLE_FILES.find((x) => x.id === id)
  if (!s) throw new Error(`サンプル ${id} がありません`)
  return { problem: parseWorkbook(base64ToArrayBuffer(s.base64)), name: `サンプル: ${s.label}` }
}

/** 新しい名簿（空）。性別・学力の2項目と、生徒3名から始める */
export function blankProblem(): Problem {
  const columns = ['性別', '学力']
  return {
    students: [1, 2, 3].map((no) => ({ no, name: '', values: Object.fromEntries(columns.map((c) => [c, ''])) })),
    columns: [
      { name: '性別', weight: 1, kind: 'category', levels: [], enabled: false },
      { name: '学力', weight: 1, kind: 'category', levels: [], enabled: false },
    ],
    numClasses: 2,
    maxPerClass: null,
    wantedGroups: [],
    unwantedGroups: [],
    warnings: [],
  }
}
