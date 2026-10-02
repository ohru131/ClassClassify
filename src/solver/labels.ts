// Excel ファイルの語彙（シート名・見出し・設定の項目名・結果の見出し）を言語ごとに持つ。
// - 読み込み（parse.ts）は全言語のシート名・見出しを受け付ける（日本語を最優先に探すので、
//   日本語のファイルの読み方は従来と変わらない）
// - 書き出し（export.ts）は渡された言語の語彙で出す。省略時は日本語（Web 版の挙動のまま）
// 書き出した名前は必ず読み込みの別名にも含まれる（同じ表を両方が使うので、往復で壊れない）。

import type { Problem } from './types'
import type { Violation } from './evaluate'

export type FileLanguage = 'ja' | 'en' | 'ko' | 'es' | 'de' | 'pt-BR'

export interface FileLabels {
  /** シート名（Excel の制約: 31文字以内、: \ / ? * [ ] を含まない） */
  sheets: {
    settings: string
    roster: string
    wanted: string
    unwanted: string
    assign: string
    byClass: string
    pairs: string
    summary: string
    failed: string
  }
  /** 生徒名簿の見出し */
  weight: string
  no: string
  name: string
  /** 設定シートの項目名 */
  studentCount: string
  maxPerClass: string
  classCount: string
  /** 組の名前（0 始まりの番号から） */
  className: (c: number) => string
  /** ペア指定のラベルの接頭辞（「同1」「別2」） */
  tagPrefix: { wanted: string; unwanted: string }
  /** 結果の見出し */
  classCol: string
  pairCol: string
  pairLabel: string
  kind: string
  members: string
  placed: string
  judged: string
  item: string
  value: string
  ideal: string
  count: string
  average: string
  wantedKind: string
  unwantedKind: string
  noPairs: string
  none: string
  /** 名前の区切り（一覧の中）と、組の区切り */
  listSep: string
  joinSep: string
}

const who = (p: Problem, i: number) => `${p.students[i].no}:${p.students[i].name}`

export const FILE_LABELS: Record<FileLanguage, FileLabels> = {
  ja: {
    sheets: { settings: '設定', roster: '生徒名簿', wanted: '同じ組ペア', unwanted: '別の組ペア', assign: '組分け', byClass: 'クラス別名簿', pairs: 'ペア指定', summary: '集計', failed: '組み合わせ失敗' },
    weight: '重み',
    no: 'NO',
    name: '名前',
    studentCount: '生徒人数',
    maxPerClass: '1クラスの最大人数',
    classCount: 'クラス数',
    className: (c) => `${c + 1}組`,
    tagPrefix: { wanted: '同', unwanted: '別' },
    classCol: '組',
    pairCol: 'ペア指定',
    pairLabel: '指定',
    kind: '種類',
    members: 'メンバー',
    placed: '配置',
    judged: '判定',
    item: '項目',
    value: '値',
    ideal: '理想',
    count: '人数',
    average: '平均',
    wantedKind: '同じ組',
    unwantedKind: '別の組',
    noPairs: '指定なし',
    none: 'なし',
    listSep: '、',
    joinSep: '・',
  },
  en: {
    sheets: { settings: 'Settings', roster: 'Roster', wanted: 'Keep together', unwanted: 'Keep apart', assign: 'Placement', byClass: 'Class lists', pairs: 'Pairings', summary: 'Summary', failed: 'Unmet conditions' },
    weight: 'Weight',
    no: 'No.',
    name: 'Name',
    studentCount: 'Number of students',
    maxPerClass: 'Maximum class size',
    classCount: 'Number of classes',
    className: (c) => `Class ${c + 1}`,
    tagPrefix: { wanted: 'T', unwanted: 'A' },
    classCol: 'Class',
    pairCol: 'Pairings',
    pairLabel: 'ID',
    kind: 'Type',
    members: 'Students',
    placed: 'Placed in',
    judged: 'Met',
    item: 'Attribute',
    value: 'Value',
    ideal: 'Target',
    count: 'Students',
    average: 'Average',
    wantedKind: 'Keep together',
    unwantedKind: 'Keep apart',
    noPairs: 'None',
    none: 'None',
    listSep: ', ',
    joinSep: ' / ',
  },
  ko: {
    sheets: { settings: '설정', roster: '학생 명단', wanted: '같은 반 배정', unwanted: '분리 배정', assign: '반 편성', byClass: '반별 명단', pairs: '배정 조건', summary: '집계', failed: '충족하지 못한 조건' },
    weight: '가중치',
    no: '번호',
    name: '이름',
    studentCount: '학생 수',
    maxPerClass: '반별 최대 인원',
    classCount: '반 수',
    className: (c) => `${c + 1}반`,
    tagPrefix: { wanted: '같', unwanted: '분' },
    classCol: '반',
    pairCol: '배정 조건',
    pairLabel: '조건',
    kind: '종류',
    members: '학생',
    placed: '배정된 반',
    judged: '충족',
    item: '항목',
    value: '값',
    ideal: '목표',
    count: '인원',
    average: '평균',
    wantedKind: '같은 반',
    unwantedKind: '분리',
    noPairs: '조건 없음',
    none: '없음',
    listSep: ', ',
    joinSep: '·',
  },
  es: {
    sheets: { settings: 'Configuración', roster: 'Lista de estudiantes', wanted: 'Mantener juntos', unwanted: 'Separar', assign: 'Distribución', byClass: 'Listas por grupo', pairs: 'Condiciones', summary: 'Resumen', failed: 'Condiciones no cumplidas' },
    weight: 'Peso',
    no: 'N.º',
    name: 'Nombre',
    studentCount: 'Número de estudiantes',
    maxPerClass: 'Máximo por grupo',
    classCount: 'Número de grupos',
    className: (c) => `Grupo ${c + 1}`,
    tagPrefix: { wanted: 'J', unwanted: 'S' },
    classCol: 'Grupo',
    pairCol: 'Condiciones',
    pairLabel: 'ID',
    kind: 'Tipo',
    members: 'Estudiantes',
    placed: 'Grupo asignado',
    judged: 'Cumple',
    item: 'Criterio',
    value: 'Valor',
    ideal: 'Meta',
    count: 'Estudiantes',
    average: 'Promedio',
    wantedKind: 'Mantener juntos',
    unwantedKind: 'Separar',
    noPairs: 'Sin condiciones',
    none: 'Ninguna',
    listSep: ', ',
    joinSep: ' / ',
  },
  de: {
    sheets: { settings: 'Einstellungen', roster: 'Schülerliste', wanted: 'Zusammen', unwanted: 'Trennen', assign: 'Klasseneinteilung', byClass: 'Klassenlisten', pairs: 'Wünsche', summary: 'Auswertung', failed: 'Nicht erfüllt' },
    weight: 'Gewicht',
    no: 'Nr.',
    name: 'Name',
    studentCount: 'Anzahl Schüler',
    maxPerClass: 'Höchstzahl pro Klasse',
    classCount: 'Anzahl Klassen',
    className: (c) => `Klasse ${c + 1}`,
    tagPrefix: { wanted: 'Z', unwanted: 'G' },
    classCol: 'Klasse',
    pairCol: 'Wünsche',
    pairLabel: 'ID',
    kind: 'Art',
    members: 'Schüler',
    placed: 'Klasse',
    judged: 'Erfüllt',
    item: 'Merkmal',
    value: 'Wert',
    ideal: 'Soll',
    count: 'Anzahl',
    average: 'Mittelwert',
    wantedKind: 'Zusammen',
    unwantedKind: 'Trennen',
    noPairs: 'Keine',
    none: 'Keine',
    listSep: ', ',
    joinSep: ' / ',
  },
  'pt-BR': {
    sheets: { settings: 'Configurações', roster: 'Lista de alunos', wanted: 'Manter juntos', unwanted: 'Separar', assign: 'Enturmação', byClass: 'Listas por turma', pairs: 'Condições', summary: 'Resumo', failed: 'Condições não atendidas' },
    weight: 'Peso',
    no: 'Nº',
    name: 'Nome',
    studentCount: 'Número de alunos',
    maxPerClass: 'Máximo por turma',
    classCount: 'Número de turmas',
    className: (c) => `Turma ${c + 1}`,
    tagPrefix: { wanted: 'J', unwanted: 'S' },
    classCol: 'Turma',
    pairCol: 'Condições',
    pairLabel: 'ID',
    kind: 'Tipo',
    members: 'Alunos',
    placed: 'Turma',
    judged: 'Atendida',
    item: 'Critério',
    value: 'Valor',
    ideal: 'Meta',
    count: 'Alunos',
    average: 'Média',
    wantedKind: 'Manter juntos',
    unwantedKind: 'Separar',
    noPairs: 'Nenhuma',
    none: 'Nenhuma',
    listSep: ', ',
    joinSep: ' / ',
  },
}

/** 条件違反の説明（日本語以外）。evaluate() の Violation から組み立てる */
const VIOLATION_TEXT: Record<Exclude<FileLanguage, 'ja'>, { wanted: (names: string) => string; unwanted: (a: string, b: string, cls: string) => string }> = {
  en: { wanted: (n) => `Keep together: ${n} are in different classes`, unwanted: (a, b, c) => `Keep apart: ${a} and ${b} are both in ${c}` },
  ko: { wanted: (n) => `같은 반 배정: ${n} 학생이 서로 다른 반에 있습니다`, unwanted: (a, b, c) => `분리 배정: ${a}, ${b} 학생이 모두 ${c}에 있습니다` },
  es: { wanted: (n) => `Mantener juntos: ${n} quedaron en grupos distintos`, unwanted: (a, b, c) => `Separar: ${a} y ${b} quedaron juntos en ${c}` },
  de: { wanted: (n) => `Zusammen: ${n} sind in verschiedenen Klassen`, unwanted: (a, b, c) => `Trennen: ${a} und ${b} sind beide in ${c}` },
  'pt-BR': { wanted: (n) => `Manter juntos: ${n} ficaram em turmas diferentes`, unwanted: (a, b, c) => `Separar: ${a} e ${b} ficaram juntos na ${c}` },
}

/** classOf を使って違反の説明文を作る（組の名前が要るため） */
export function violationText(lang: FileLanguage, v: Violation, p: Problem, classOf: number[]): string {
  if (lang === 'ja') return v.message
  const l = FILE_LABELS[lang]
  const t = VIOLATION_TEXT[lang]
  if (v.type === 'wanted') return t.wanted(v.students.map((i) => who(p, i)).join(l.listSep))
  return t.unwanted(who(p, v.students[0]), who(p, v.students[1]), l.className(classOf[v.students[0]]))
}

// 「N.º」「Nº」「N°」はどれも nº に揃える。º まで消すと素の「N」（属性の列名としてありうる）と区別できなくなる
const norm = (s: string) => s.trim().toLowerCase().replace(/[.．]/g, '').replace(/°/g, 'º')
const all = <K extends keyof FileLabels>(key: K) => Object.values(FILE_LABELS).map((l) => l[key])

/** 読み込みで受け付けるシート名（日本語を最初に探す） */
export const SHEET_ALIASES = {
  settings: Object.values(FILE_LABELS).map((l) => l.sheets.settings),
  roster: Object.values(FILE_LABELS).map((l) => l.sheets.roster),
  wanted: Object.values(FILE_LABELS).map((l) => l.sheets.wanted),
  unwanted: Object.values(FILE_LABELS).map((l) => l.sheets.unwanted),
}

// 素の「N」「#」は入れない（属性の列名として使われうるので、番号の列と取り違える）
const NO_HEADERS = new Set([...all('no'), 'NO', 'NUM', 'NÚM', 'NUMERO', 'NÚMERO', 'NUMMER', 'NR'].map(norm))
const NAME_HEADERS = new Set([...all('name'), '名前', '氏名', 'Nombre', 'Nome', 'Name', '이름', '성명'].map(norm))
const CLASS_COUNT_KEYS = new Set([...all('classCount'), 'Number of groups', '학급 수', 'Anzahl der Klassen', 'Número de turmas'].map(norm))
const MAX_KEYS = new Set([...all('maxPerClass'), 'Max class size', 'Maximum per class'].map(norm))

export const isNoHeader = (h: string) => NO_HEADERS.has(norm(h))
export const isNameHeader = (h: string) => NAME_HEADERS.has(norm(h))
export const isClassCountKey = (k: string) => CLASS_COUNT_KEYS.has(norm(k))
export const isMaxPerClassKey = (k: string) => MAX_KEYS.has(norm(k))

/** 読み込み時の警告・エラーの文言。省略時は日本語（Web 版と同じ） */
export interface ParseMessages {
  rosterMissing: string
  noStudents: string
  noMissing: (row: number, name: string) => string
  duplicateNo: (no: number) => string
  unknownNo: (sheet: string, row: number, no: number) => string
  classCountUnknown: (k: number) => string
  maxTooSmall: (max: number, k: number) => string
}

export const JA_PARSE_MESSAGES: ParseMessages = {
  rosterMissing: '「生徒名簿」シートが見つからないか、データがありません',
  noStudents: '生徒のデータがありません（3行目から下に NO と名前を入れてください）',
  noMissing: (row, name) => `生徒名簿 ${row}行目: 「${name}」の NO が空欄か読み取れないため、読み込んでいません`,
  duplicateNo: (no) => `出席番号 ${no} が2人以上に付いています`,
  unknownNo: (sheet, row, no) => `「${sheet}」${row}行目: 出席番号 ${no} は名簿にありません`,
  classCountUnknown: (k) => `「設定」シートのクラス数が読み取れなかったので、${k} クラスにしました`,
  maxTooSmall: (max, k) => `1クラス最大 ${max} 人 × ${k} クラスでは全員が入りきらないため、最大人数の設定は使いませんでした`,
}
