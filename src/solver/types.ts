export type ColumnKind = 'flag' | 'category' | 'numeric'

export interface ColumnSpec {
  name: string
  weight: number
  kind: ColumnKind
  /** flag/category の水準（空欄以外の値） */
  levels: string[]
  enabled: boolean
}

export interface Student {
  no: number
  name: string
  /** 列名 → セル値（文字列化済み、空欄は ''） */
  values: Record<string, string>
}

export interface Problem {
  students: Student[]
  columns: ColumnSpec[]
  numClasses: number
  /** 1クラスの最大人数（未指定なら null） */
  maxPerClass: number | null
  /** 同じ組にしたいグループ（生徒 index） */
  wantedGroups: number[][]
  /** 別の組にしたいグループ（生徒 index、グループ内の全ペアが対象） */
  unwantedGroups: number[][]
  warnings: string[]
}

/** ソルバーに渡す数値化済みの問題 */
export interface CompiledProblem {
  n: number
  k: number
  minSize: number
  maxSize: number
  /** 属性ごとの重み */
  weights: number[]
  /** 属性ごとの目標値の下限/上限（クラス1つあたり） */
  lo: number[]
  hi: number[]
  target: number[]
  /** attr[a][i] = 生徒 i の属性 a の値 */
  attr: number[][]
  wantedPairs: [number, number][]
  unwantedPairs: [number, number][]
}

export interface SolveResult {
  classOf: number[]
  cost: number
  elapsedMs: number
  iterations: number
}
