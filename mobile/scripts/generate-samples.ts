// 言語別のサンプル名簿を作り、base64 でバンドルへ埋め込む（lib/samples.generated.ts）。
//   npm run samples:generate   （リポジトリ直下で npm install 済みであること。共有ソルバーを読むため）
//
// - ja: public/sample*.xlsx をそのまま埋め込む（test/stored-project.test.ts が一致を確認する）
// - 他の言語: 日本語のサンプルを読み、構造（人数・組数・項目の分布・ペア指定）はそのままに、
//   氏名をその国らしい名前へ、項目名を docs/i18n-glossary.md の「配慮の必要な項目名」へ置き換え、
//   その言語のシート名・見出しで書き出す（読み込み側の別名の検証も兼ねる）。
import { readFileSync, writeFileSync } from 'node:fs'

import { parseWorkbook } from '../../src/solver/parse'
import * as XLSX from 'xlsx-js-style'

import { rosterWorkbook } from '../../src/solver/export'
import type { FileLanguage } from '../../src/solver/labels'
import type { Problem } from '../../src/solver/types'

const FILES = [
  { id: 'sample1', file: 'sample1.xlsx' },
  { id: 'sample2', file: 'sample2.xlsx' },
  { id: 'sample-group', file: 'sample-group.xlsx' },
] as const

const LABELS: Record<FileLanguage, Record<(typeof FILES)[number]['id'], string>> = {
  ja: { sample1: 'クラス分け（80名・4組）', sample2: 'クラス分け（80名・シンプル）', 'sample-group': 'グループ分け（30名・6班）' },
  en: { sample1: 'Class placement (80 students, 4 classes)', sample2: 'Class placement (80 students, simple)', 'sample-group': 'Group work (30 students, 6 groups)' },
  ko: { sample1: '반 편성(학생 80명·4개 반)', sample2: '반 편성(학생 80명·간단)', 'sample-group': '모둠 편성(30명·6모둠)' },
  es: { sample1: 'Grupos (80 estudiantes, 4 grupos)', sample2: 'Grupos (80 estudiantes, simple)', 'sample-group': 'Equipos de trabajo (30, 6 equipos)' },
  de: { sample1: 'Klassen (80 Schüler, 4 Klassen)', sample2: 'Klassen (80 Schüler, einfach)', 'sample-group': 'Gruppenarbeit (30 Schüler, 6 Gruppen)' },
  'pt-BR': { sample1: 'Turmas (80 alunos, 4 turmas)', sample2: 'Turmas (80 alunos, simples)', 'sample-group': 'Grupos de trabalho (30 alunos, 6 grupos)' },
}

// 項目名（docs/i18n-glossary.md 第3節）。性別と支援の必要性は配慮のある表現にする
const COLUMNS: Record<Exclude<FileLanguage, 'ja'>, Record<string, string>> = {
  en: { '性別♀': 'Girl', 学習支援: 'Learning support', 登校支援: 'Attendance support', 視覚配慮: 'Vision support', 情緒面の配慮: 'Emotional support', 走力: 'Running', ピアノ: 'Piano', 学習: 'Academics', 体育: 'PE', PTA: 'Parent committee', 協調性: 'Teamwork', 前回の組: 'Previous group' },
  ko: { '性別♀': '여학생', 学習支援: '학습 지원', 登校支援: '등교 지원', 視覚配慮: '시각 지원', 情緒面の配慮: '정서 지원', 走力: '달리기', ピアノ: '피아노', 学習: '학업', 体育: '체육', PTA: '학부모회', 協調性: '협동심', 前回の組: '이전 모둠' },
  es: { '性別♀': 'Niña', 学習支援: 'Apoyo en el aprendizaje', 登校支援: 'Apoyo a la asistencia', 視覚配慮: 'Apoyo visual', 情緒面の配慮: 'Apoyo emocional', 走力: 'Velocidad', ピアノ: 'Piano', 学習: 'Desempeño académico', 体育: 'Educación física', PTA: 'Comité de familias', 協調性: 'Trabajo en equipo', 前回の組: 'Grupo anterior' },
  de: { '性別♀': 'Mädchen', 学習支援: 'Lernförderung', 登校支援: 'Unterstützung Anwesenheit', 視覚配慮: 'Unterstützung Sehen', 情緒面の配慮: 'Emotionale Unterstützung', 走力: 'Laufen', ピアノ: 'Klavier', 学習: 'Leistung', 体育: 'Sport', PTA: 'Elternbeirat', 協調性: 'Teamfähigkeit', 前回の組: 'Vorherige Gruppe' },
  'pt-BR': { '性別♀': 'Menina', 学習支援: 'Apoio à aprendizagem', 登校支援: 'Apoio à frequência', 視覚配慮: 'Apoio visual', 情緒面の配慮: 'Apoio emocional', 走力: 'Corrida', ピアノ: 'Piano', 学習: 'Desempenho', 体育: 'Educação física', PTA: 'Conselho de pais', 協調性: 'Cooperação', 前回の組: 'Grupo anterior' },
}

// 名（女・男）と姓。ありふれた名前を選び、特定の人を指さないよう組み合わせで作る
const NAMES: Record<Exclude<FileLanguage, 'ja'>, { f: string[]; m: string[]; last: string[]; join: (first: string, last: string) => string }> = {
  en: {
    f: ['Emma', 'Olivia', 'Ava', 'Sophia', 'Mia', 'Isla', 'Grace', 'Chloe', 'Lily', 'Zoe', 'Harper', 'Ella', 'Ruby', 'Aria', 'Nora', 'Maya', 'Leah', 'Ivy', 'Hannah', 'Amelia'],
    m: ['Liam', 'Noah', 'Oliver', 'Ethan', 'Lucas', 'Mason', 'Jack', 'Leo', 'Henry', 'Owen', 'Caleb', 'Ryan', 'Isaac', 'Samuel', 'Daniel', 'Aiden', 'Max', 'Theo', 'Jacob', 'Eli'],
    last: ['Smith', 'Johnson', 'Brown', 'Taylor', 'Wilson', 'Martin', 'Anderson', 'Thompson', 'White', 'Walker', 'Young', 'King', 'Wright', 'Scott', 'Green', 'Baker', 'Adams', 'Nelson', 'Hill', 'Campbell'],
    join: (f, l) => `${f} ${l}`,
  },
  ko: {
    f: ['서연', '지우', '서윤', '하은', '민서', '지유', '윤서', '채원', '수아', '지민', '예은', '다은', '하린', '소율', '유나', '시은', '예린', '지안', '서아', '나은'],
    m: ['민준', '서준', '도윤', '예준', '시우', '하준', '주원', '지호', '지후', '준우', '건우', '우진', '선우', '현우', '연우', '유준', '정우', '승현', '민재', '지환'],
    last: ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임', '한', '오', '서', '신', '권', '황', '안', '송', '류', '홍'],
    join: (f, l) => `${l}${f}`,
  },
  es: {
    f: ['Sofía', 'Valentina', 'Isabella', 'Camila', 'Mariana', 'Lucía', 'Martina', 'Daniela', 'Gabriela', 'Victoria', 'Renata', 'Emilia', 'Florencia', 'Paula', 'Antonella', 'Josefa', 'Catalina', 'Ximena', 'Julieta', 'Agustina'],
    m: ['Mateo', 'Santiago', 'Sebastián', 'Matías', 'Benjamín', 'Nicolás', 'Tomás', 'Joaquín', 'Diego', 'Emiliano', 'Gabriel', 'Lucas', 'Martín', 'Felipe', 'Alejandro', 'Samuel', 'Agustín', 'Daniel', 'Maximiliano', 'Vicente'],
    last: ['González', 'Rodríguez', 'Pérez', 'López', 'Martínez', 'Sánchez', 'Ramírez', 'Torres', 'Flores', 'Rojas', 'Díaz', 'Morales', 'Silva', 'Castro', 'Vargas', 'Herrera', 'Muñoz', 'Gutiérrez', 'Jiménez', 'Romero'],
    join: (f, l) => `${f} ${l}`,
  },
  de: {
    f: ['Emilia', 'Mia', 'Sophia', 'Hannah', 'Lina', 'Emma', 'Mila', 'Clara', 'Lea', 'Marie', 'Ella', 'Leni', 'Frieda', 'Ida', 'Lara', 'Anna', 'Luisa', 'Paula', 'Greta', 'Johanna'],
    m: ['Noah', 'Leon', 'Paul', 'Ben', 'Finn', 'Elias', 'Felix', 'Henry', 'Jonas', 'Luis', 'Emil', 'Anton', 'Theo', 'Maximilian', 'Jakob', 'Moritz', 'Oskar', 'David', 'Julian', 'Lukas'],
    last: ['Müller', 'Schmidt', 'Schneider', 'Fischer', 'Weber', 'Meyer', 'Wagner', 'Becker', 'Schulz', 'Hoffmann', 'Koch', 'Richter', 'Klein', 'Wolf', 'Schröder', 'Neumann', 'Braun', 'Zimmermann', 'Hartmann', 'Krüger'],
    join: (f, l) => `${f} ${l}`,
  },
  'pt-BR': {
    f: ['Helena', 'Alice', 'Laura', 'Maria Alice', 'Valentina', 'Heloísa', 'Maria Clara', 'Cecília', 'Júlia', 'Sophia', 'Manuela', 'Isabela', 'Lívia', 'Beatriz', 'Luiza', 'Lorena', 'Giovanna', 'Mariana', 'Antonella', 'Clara'],
    m: ['Miguel', 'Arthur', 'Gael', 'Heitor', 'Theo', 'Davi', 'Gabriel', 'Bernardo', 'Samuel', 'João Miguel', 'Pedro', 'Lucas', 'Rafael', 'Enzo', 'Matheus', 'Benício', 'Nicolas', 'Gustavo', 'Guilherme', 'Joaquim'],
    last: ['Silva', 'Santos', 'Oliveira', 'Souza', 'Rodrigues', 'Ferreira', 'Alves', 'Pereira', 'Lima', 'Gomes', 'Costa', 'Ribeiro', 'Martins', 'Carvalho', 'Almeida', 'Lopes', 'Soares', 'Fernandes', 'Vieira', 'Barbosa'],
    join: (f, l) => `${f} ${l}`,
  },
}

const CATEGORY_MARKS: Record<string, string> = { '×': '1', '○': '2', '◎': '3' }

function localize(p: Problem, lang: Exclude<FileLanguage, 'ja'>): Problem {
  const colName = (c: string) => COLUMNS[lang][c] ?? c
  const names = NAMES[lang]
  const used = new Set<string>()
  const counters = { f: 0, m: 0 }
  const kinds = new Map(p.columns.map((c) => [c.name, c.kind]))
  const students = p.students.map((s, i) => {
    const g: 'f' | 'm' = s.values['性別♀'] ? 'f' : 'm'
    let name = ''
    for (let tries = 0; !name || used.has(name); tries++) {
      const n = counters[g]++
      name = names.join(names[g][n % names[g].length], names.last[(i * 7 + tries * 3 + Math.floor(n / names[g].length) * 11) % names.last.length])
    }
    used.add(name)
    const values = Object.fromEntries(
      Object.entries(s.values).map(([k, v]) => {
        const kind = kinds.get(k)
        const nv = v === '' ? '' : kind === 'flag' ? '✓' : kind === 'category' ? (CATEGORY_MARKS[v] ?? v) : v
        return [colName(k), nv]
      }),
    )
    return { ...s, name, values }
  })
  const columns = p.columns.map((c) => ({
    ...c,
    name: colName(c.name),
    levels: c.kind === 'flag' ? ['✓'] : c.kind === 'category' ? c.levels.map((l) => CATEGORY_MARKS[l] ?? l).sort() : c.levels,
  }))
  return { ...p, students, columns }
}

const toBuf = (b: Buffer) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer
const out: Record<string, { id: string; label: string; base64: string }[]> = {}
for (const lang of Object.keys(LABELS) as FileLanguage[]) {
  out[lang] = FILES.map(({ id, file }) => {
    const raw = readFileSync(new URL(`../../public/${file}`, import.meta.url))
    if (lang === 'ja') return { id, label: LABELS.ja[id], base64: raw.toString('base64') }
    const jp = parseWorkbook(toBuf(raw))
    const p = localize(jp, lang)
    return { id, label: LABELS[lang][id], base64: XLSX.write(rosterWorkbook(p, jp.numClasses, lang), { bookType: 'xlsx', type: 'base64', compression: true }) as string }
  })
}

const body = Object.entries(out)
  .map(([lang, list]) => `  ${JSON.stringify(lang)}: [\n${list.map((s) => `    { id: ${JSON.stringify(s.id)}, label: ${JSON.stringify(s.label)}, base64: ${JSON.stringify(s.base64)} },`).join('\n')}\n  ],`)
  .join('\n')
writeFileSync(
  new URL('../lib/samples.generated.ts', import.meta.url),
  `// 自動生成（scripts/generate-samples.ts）。手で編集しないこと。\nimport type { AppLanguage } from './i18n'\n\nexport const SAMPLE_FILES: Record<AppLanguage, { id: string; label: string; base64: string }[]> = {\n${body}\n}\n`,
)
console.log('lib/samples.generated.ts を生成しました')
