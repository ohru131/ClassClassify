// 言語別のサンプル名簿（Web 版・スマホ版で共通）を作る。
//   npm run samples:generate
// 出力: public/samples/<lang>/{sample1,sample2,sample-group}.xlsx と public/samples/<lang>/template.zip
// （スマホ版は mobile/ の `npm run samples:embed` でこの出力を base64 にして埋め込む）
//
// - ja: 既存の public/sample*.xlsx（ピアノ伴奏者など日本固有の配慮を含む）を基本にし、
//   点数の項目を1つだけ足して「該当／カテゴリ／数値（平均）」の3種類がすべて使われるようにする。
//   Web 版の従来の URL（public/sample*.xlsx・public/template.zip）は変えずに残す。
// - 他の言語: その国の学校でクラス分けに実際に使われる項目で作り直す（根拠は docs/i18n-glossary.md 第3節）。
//   人数・組数・ペア指定の数は日本語のサンプルに揃える。値は固定の乱数で作るので毎回同じ出力になる。
// 氏名は、その国で一般的な名と姓を機械的に組み合わせた架空のもの。
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import type { WorkBook } from 'xlsx-js-style'
import XLSX from '../src/solver/xlsx'

import { rosterWorkbook } from '../src/solver/export'
import type { FileLanguage } from '../src/solver/labels'
import type { ColumnSpec, Problem, Student } from '../src/solver/types'
import { detectKind } from '../src/solver/columns'

type Lang = Exclude<FileLanguage, 'ja'>
type SampleId = 'sample1' | 'sample2' | 'sample-group'
const IDS: SampleId[] = ['sample1', 'sample2', 'sample-group']
const root = new URL('../public/', import.meta.url)

// ---------- 乱数（固定の種） ----------
function rng(seed: number) {
  let s = seed >>> 0 || 1
  return () => {
    s ^= s << 13
    s >>>= 0
    s ^= s >>> 17
    s ^= s << 5
    s >>>= 0
    return s / 4294967296
  }
}

// ---------- 項目の定義 ----------
type Col =
  | { kind: 'flag'; name: string; rate: number; mark?: string }
  | { kind: 'category'; name: string; values: string[]; weights?: number[] }
  | { kind: 'score'; name: string; min: number; max: number; step: number; decimals?: number }
  | { kind: 'gender'; name: string; values: [string, string] }

interface SampleDef {
  n: number
  k: number
  max: number
  columns: Col[]
  wanted: number
  unwanted: number
}

const TICK = '✓'

// 国ごとの項目（docs/i18n-glossary.md 第3節に対訳と根拠）
const DEFS: Record<Lang, Record<SampleId, SampleDef>> = {
  en: {
    sample1: {
      n: 80, k: 4, max: 25, wanted: 4, unwanted: 4,
      columns: [
        { kind: 'gender', name: 'Gender', values: ['F', 'M'] },
        { kind: 'score', name: 'Reading score', min: 55, max: 100, step: 1 },
        { kind: 'category', name: 'Math level', values: ['1', '2', '3'], weights: [1, 2, 1] },
        { kind: 'flag', name: 'IEP/504 plan', rate: 0.12 },
        { kind: 'flag', name: 'English learner', rate: 0.1 },
        { kind: 'flag', name: 'Behavior support', rate: 0.08 },
        { kind: 'flag', name: 'Leadership', rate: 0.12 },
      ],
    },
    sample2: {
      n: 80, k: 4, max: 25, wanted: 1, unwanted: 2,
      columns: [
        { kind: 'gender', name: 'Gender', values: ['F', 'M'] },
        { kind: 'score', name: 'Reading score', min: 55, max: 100, step: 1 },
        { kind: 'flag', name: 'IEP/504 plan', rate: 0.12 },
        { kind: 'category', name: 'Previous class', values: ['A', 'B', 'C', 'D'] },
      ],
    },
    'sample-group': {
      n: 30, k: 6, max: 5, wanted: 0, unwanted: 4,
      columns: [
        { kind: 'gender', name: 'Gender', values: ['F', 'M'] },
        { kind: 'score', name: 'Reading score', min: 55, max: 100, step: 1 },
        { kind: 'flag', name: 'Leadership', rate: 0.2 },
        { kind: 'category', name: 'Previous group', values: ['1', '2', '3', '4', '5', '6'] },
      ],
    },
  },
  ko: {
    sample1: {
      n: 80, k: 4, max: 25, wanted: 4, unwanted: 4,
      columns: [
        { kind: 'gender', name: '성별', values: ['여', '남'] },
        { kind: 'score', name: '학업 성취도', min: 55, max: 100, step: 1 },
        { kind: 'category', name: '교우 관계', values: ['원만', '보통', '지원 필요'], weights: [3, 4, 1] },
        { kind: 'flag', name: '특수교육 대상', rate: 0.06 },
        { kind: 'flag', name: '한국어 지원', rate: 0.06 },
        { kind: 'flag', name: '리더십', rate: 0.12 },
        { kind: 'category', name: '출신 초등학교', values: ['가람초', '나래초', '다솜초', '라온초'] },
      ],
    },
    sample2: {
      n: 80, k: 4, max: 25, wanted: 1, unwanted: 2,
      columns: [
        { kind: 'gender', name: '성별', values: ['여', '남'] },
        { kind: 'score', name: '학업 성취도', min: 55, max: 100, step: 1 },
        { kind: 'flag', name: '특수교육 대상', rate: 0.06 },
        { kind: 'category', name: '이전 반', values: ['1반', '2반', '3반', '4반'] },
      ],
    },
    'sample-group': {
      n: 30, k: 6, max: 5, wanted: 0, unwanted: 4,
      columns: [
        { kind: 'gender', name: '성별', values: ['여', '남'] },
        { kind: 'score', name: '학업 성취도', min: 55, max: 100, step: 1 },
        { kind: 'flag', name: '리더십', rate: 0.2 },
        { kind: 'category', name: '이전 모둠', values: ['1', '2', '3', '4', '5', '6'] },
      ],
    },
  },
  es: {
    sample1: {
      n: 80, k: 4, max: 25, wanted: 4, unwanted: 4,
      columns: [
        { kind: 'gender', name: 'Género', values: ['F', 'M'] },
        { kind: 'score', name: 'Promedio de notas', min: 4.0, max: 7.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'NEE (PIE)', rate: 0.1 },
        { kind: 'category', name: 'Convivencia escolar', values: ['Sin observaciones', 'Seguimiento'], weights: [7, 1] },
        { kind: 'flag', name: 'Liderazgo', rate: 0.12 },
        { kind: 'category', name: 'Grupo de origen', values: ['A', 'B', 'C', 'D'] },
      ],
    },
    sample2: {
      n: 80, k: 4, max: 25, wanted: 1, unwanted: 2,
      columns: [
        { kind: 'gender', name: 'Género', values: ['F', 'M'] },
        { kind: 'score', name: 'Promedio de notas', min: 4.0, max: 7.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'NEE (PIE)', rate: 0.1 },
        { kind: 'category', name: 'Grupo de origen', values: ['A', 'B', 'C', 'D'] },
      ],
    },
    'sample-group': {
      n: 30, k: 6, max: 5, wanted: 0, unwanted: 4,
      columns: [
        { kind: 'gender', name: 'Género', values: ['F', 'M'] },
        { kind: 'score', name: 'Promedio de notas', min: 4.0, max: 7.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'Liderazgo', rate: 0.2 },
        { kind: 'category', name: 'Equipo anterior', values: ['1', '2', '3', '4', '5', '6'] },
      ],
    },
  },
  de: {
    sample1: {
      n: 80, k: 4, max: 25, wanted: 4, unwanted: 4,
      columns: [
        { kind: 'gender', name: 'Geschlecht', values: ['w', 'm'] },
        { kind: 'score', name: 'Notenschnitt', min: 1.0, max: 4.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'Förderbedarf', rate: 0.08 },
        { kind: 'flag', name: 'DaZ', rate: 0.1 },
        { kind: 'category', name: 'Verhalten', values: ['unauffällig', 'Unterstützung'], weights: [7, 1] },
        { kind: 'category', name: 'Herkunftsgrundschule', values: ['GS Am Park', 'GS Lindenweg', 'GS Nord', 'GS Süd'] },
      ],
    },
    sample2: {
      n: 80, k: 4, max: 25, wanted: 1, unwanted: 2,
      columns: [
        { kind: 'gender', name: 'Geschlecht', values: ['w', 'm'] },
        { kind: 'score', name: 'Notenschnitt', min: 1.0, max: 4.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'Förderbedarf', rate: 0.08 },
        { kind: 'category', name: 'Herkunftsgrundschule', values: ['GS Am Park', 'GS Lindenweg', 'GS Nord', 'GS Süd'] },
      ],
    },
    'sample-group': {
      n: 30, k: 6, max: 5, wanted: 0, unwanted: 4,
      columns: [
        { kind: 'gender', name: 'Geschlecht', values: ['w', 'm'] },
        { kind: 'score', name: 'Notenschnitt', min: 1.0, max: 4.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'Teamfähigkeit', rate: 0.2 },
        { kind: 'category', name: 'Vorherige Gruppe', values: ['1', '2', '3', '4', '5', '6'] },
      ],
    },
  },
  'pt-BR': {
    sample1: {
      n: 80, k: 4, max: 25, wanted: 4, unwanted: 4,
      columns: [
        { kind: 'gender', name: 'Gênero', values: ['F', 'M'] },
        { kind: 'score', name: 'Média', min: 5.0, max: 10.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'AEE', rate: 0.08 },
        { kind: 'category', name: 'Convivência', values: ['Tranquila', 'Acompanhamento'], weights: [7, 1] },
        { kind: 'flag', name: 'Liderança', rate: 0.12 },
        { kind: 'category', name: 'Turma de origem', values: ['A', 'B', 'C', 'D'] },
      ],
    },
    sample2: {
      n: 80, k: 4, max: 25, wanted: 1, unwanted: 2,
      columns: [
        { kind: 'gender', name: 'Gênero', values: ['F', 'M'] },
        { kind: 'score', name: 'Média', min: 5.0, max: 10.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'AEE', rate: 0.08 },
        { kind: 'category', name: 'Turma de origem', values: ['A', 'B', 'C', 'D'] },
      ],
    },
    'sample-group': {
      n: 30, k: 6, max: 5, wanted: 0, unwanted: 4,
      columns: [
        { kind: 'gender', name: 'Gênero', values: ['F', 'M'] },
        { kind: 'score', name: 'Média', min: 5.0, max: 10.0, step: 0.1, decimals: 1 },
        { kind: 'flag', name: 'Liderança', rate: 0.2 },
        { kind: 'category', name: 'Grupo anterior', values: ['1', '2', '3', '4', '5', '6'] },
      ],
    },
  },
}

// 名（女・男）と姓。ありふれた名前を機械的に組み合わせた架空の氏名
const NAMES: Record<Lang, { f: string[]; m: string[]; last: string[]; join: (first: string, last: string) => string }> = {
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

const pick = (r: () => number, values: string[], weights?: number[]) => {
  const w = weights ?? values.map(() => 1)
  let x = r() * w.reduce((a, b) => a + b, 0)
  for (let i = 0; i < values.length; i++) if ((x -= w[i]) < 0) return values[i]
  return values[values.length - 1]
}

function build(lang: Lang, id: SampleId): Problem {
  const def = DEFS[lang][id]
  const r = rng([...`${lang}/${id}`].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7))
  const names = NAMES[lang]
  const used = new Set<string>()
  const counter = { f: 0, m: 0 }
  const students: Student[] = []
  for (let i = 0; i < def.n; i++) {
    const g: 'f' | 'm' = i % 2 === 0 ? 'f' : 'm' // 男女は半々
    const values: Record<string, string> = {}
    for (const c of def.columns) {
      if (c.kind === 'gender') values[c.name] = g === 'f' ? c.values[0] : c.values[1]
      else if (c.kind === 'flag') values[c.name] = r() < c.rate ? (c.mark ?? TICK) : ''
      else if (c.kind === 'category') values[c.name] = pick(r, c.values, c.weights)
      else {
        const steps = Math.round((c.max - c.min) / c.step)
        // 中央寄りの分布（3つの一様乱数の平均）
        const u = (r() + r() + r()) / 3
        const v = c.min + Math.round(u * steps) * c.step
        values[c.name] = v.toFixed(c.decimals ?? 0)
      }
    }
    let name = ''
    for (let tries = 0; !name || used.has(name); tries++) {
      const k = counter[g]++
      name = names.join(names[g][k % names[g].length], names.last[Math.floor(r() * names.last.length)])
    }
    used.add(name)
    students.push({ no: i + 1, name, values })
  }
  // 組み合わせたい・離したい生徒（同じ組は2〜3人、別の組は2人）。互いに重ならない生徒から選ぶ
  const taken = new Set<number>()
  const draw = () => {
    for (;;) {
      const i = Math.floor(r() * def.n)
      if (!taken.has(i)) {
        taken.add(i)
        return i
      }
    }
  }
  const wantedGroups = Array.from({ length: def.wanted }, (_, gi) => Array.from({ length: gi === 0 ? 3 : 2 }, draw))
  const unwantedGroups = Array.from({ length: def.unwanted }, (_, gi) => Array.from({ length: gi === 1 ? 3 : 2 }, draw))
  const columns: ColumnSpec[] = def.columns.map((c) => {
    const { kind, levels } = detectKind(students.map((s) => s.values[c.name]))
    return { name: c.name, weight: 1, kind, levels, enabled: levels.length > 0 }
  })
  return { students, columns, numClasses: def.k, maxPerClass: def.max, wantedGroups, unwantedGroups, warnings: [] }
}

// ---------- ja: 既存のサンプルに点数の項目を1つ足す ----------
const JA_SCORE: Record<SampleId, { name: string; min: number; max: number; decimals: number }> = {
  sample1: { name: 'テスト平均', min: 45, max: 100, decimals: 0 },
  sample2: { name: 'テスト平均', min: 45, max: 100, decimals: 0 },
  'sample-group': { name: '50m走（秒）', min: 7.6, max: 11.0, decimals: 1 },
}

function jaWorkbook(id: SampleId): WorkBook {
  const file = readFileSync(new URL(`${id}.xlsx`, root))
  const wb = XLSX.read(file, { type: 'buffer', cellStyles: true })
  const ws = wb.Sheets['生徒名簿']
  const range = XLSX.utils.decode_range(ws['!ref']!)
  const col = range.e.c + 1
  const r = rng(id.length * 97 + 13)
  const spec = JA_SCORE[id]
  const rows: (string | number)[][] = [[1], [spec.name]]
  for (let row = 2; row <= range.e.r; row++) {
    const hasNo = ws[XLSX.utils.encode_cell({ r: row, c: 0 })]
    const u = (r() + r() + r()) / 3
    rows.push(hasNo ? [Number((spec.min + u * (spec.max - spec.min)).toFixed(spec.decimals))] : [''])
  }
  XLSX.utils.sheet_add_aoa(ws, rows, { origin: { r: 0, c: col } })
  return wb
}

// ---------- 出力 ----------
const CRC = new Uint32Array(256).map((_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})
const crc32 = (b: Uint8Array) => {
  let c = 0xffffffff
  for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

// 更新日時は固定（毎回同じ出力にするため）: 2026-01-01 00:00
const DOS_DATE = ((2026 - 1980) << 9) | (1 << 5) | 1

/** 無圧縮の zip（ひな形の配布用。xlsx 自体が圧縮済みなので圧縮しない） */
function zip(files: { name: string; data: Uint8Array }[]): Buffer {
  const parts: Buffer[] = []
  const central: Buffer[] = []
  let offset = 0
  for (const f of files) {
    const name = Buffer.from(f.name, 'utf8')
    const crc = crc32(f.data)
    const local = Buffer.alloc(30)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(20, 4)
    local.writeUInt16LE(0x0800, 6) // UTF-8 のファイル名
    local.writeUInt16LE(DOS_DATE, 12)
    local.writeUInt32LE(crc, 14)
    local.writeUInt32LE(f.data.length, 18)
    local.writeUInt32LE(f.data.length, 22)
    local.writeUInt16LE(name.length, 26)
    parts.push(local, name, Buffer.from(f.data))
    const cd = Buffer.alloc(46)
    cd.writeUInt32LE(0x02014b50, 0)
    cd.writeUInt16LE(20, 4)
    cd.writeUInt16LE(20, 6)
    cd.writeUInt16LE(0x0800, 8)
    cd.writeUInt16LE(DOS_DATE, 14)
    cd.writeUInt32LE(crc, 16)
    cd.writeUInt32LE(f.data.length, 20)
    cd.writeUInt32LE(f.data.length, 24)
    cd.writeUInt16LE(name.length, 28)
    cd.writeUInt32LE(offset, 42)
    central.push(cd, name)
    offset += 30 + name.length + f.data.length
  }
  const cdBuf = Buffer.concat(central)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0)
  end.writeUInt16LE(files.length, 8)
  end.writeUInt16LE(files.length, 10)
  end.writeUInt32LE(cdBuf.length, 12)
  end.writeUInt32LE(offset, 16)
  return Buffer.concat([...parts, cdBuf, end])
}

// ひな形の zip に入れるファイル名（その言語で）
const ZIP_NAMES: Record<FileLanguage, Record<SampleId, string>> = {
  ja: { sample1: 'クラス分けサンプル1.xlsx', sample2: 'クラス分けサンプル2.xlsx', 'sample-group': 'グループ分けサンプル.xlsx' },
  en: { sample1: 'class-placement-sample-1.xlsx', sample2: 'class-placement-sample-2.xlsx', 'sample-group': 'group-work-sample.xlsx' },
  ko: { sample1: '반편성_예시1.xlsx', sample2: '반편성_예시2.xlsx', 'sample-group': '모둠편성_예시.xlsx' },
  es: { sample1: 'grupos-ejemplo-1.xlsx', sample2: 'grupos-ejemplo-2.xlsx', 'sample-group': 'equipos-ejemplo.xlsx' },
  de: { sample1: 'Klasseneinteilung-Beispiel-1.xlsx', sample2: 'Klasseneinteilung-Beispiel-2.xlsx', 'sample-group': 'Gruppeneinteilung-Beispiel.xlsx' },
  'pt-BR': { sample1: 'turmas-exemplo-1.xlsx', sample2: 'turmas-exemplo-2.xlsx', 'sample-group': 'grupos-exemplo.xlsx' },
}

const write = (wb: WorkBook) => new Uint8Array(XLSX.write(wb, { bookType: 'xlsx', type: 'array', compression: true }) as ArrayBuffer)

for (const lang of Object.keys(ZIP_NAMES) as FileLanguage[]) {
  const dir = new URL(`samples/${lang}/`, root)
  mkdirSync(dir, { recursive: true })
  const files = IDS.map((id) => {
    const wb = lang === 'ja' ? jaWorkbook(id) : rosterWorkbook(build(lang, id), DEFS[lang][id].k, lang)
    const data = write(wb)
    writeFileSync(new URL(`${id}.xlsx`, dir), data)
    return { name: ZIP_NAMES[lang][id], data }
  })
  writeFileSync(new URL('template.zip', dir), zip(files))
}
console.log('public/samples/<lang>/ を生成しました')
