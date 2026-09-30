import type { Problem } from './types'

export interface PairColor {
  /** 背景色（淡色） */
  bg: string
  /** 文字色（濃色） */
  fg: string
}

/** 同じ組グループの色（淡い背景で行を塗り分ける） */
export const WANTED_COLORS: PairColor[] = [
  { bg: '#E0F2FE', fg: '#0369A1' },
  { bg: '#DCFCE7', fg: '#15803D' },
  { bg: '#FEF3C7', fg: '#B45309' },
  { bg: '#EDE9FE', fg: '#6D28D9' },
  { bg: '#CCFBF1', fg: '#0F766E' },
  { bg: '#FFEDD5', fg: '#C2410C' },
  { bg: '#E0E7FF', fg: '#4338CA' },
  { bg: '#ECFCCB', fg: '#4D7C0F' },
  { bg: '#F3E8FF', fg: '#7E22CE' },
  { bg: '#CFFAFE', fg: '#0E7490' },
]
/** 別の組の指定（バッジ） */
export const UNWANTED_COLOR: PairColor = { bg: '#FFE4E6', fg: '#BE123C' }
/** 条件を満たせていない */
export const VIOLATION_COLOR: PairColor = { bg: '#FECACA', fg: '#B91C1C' }

export const wantedColor = (g: number) => WANTED_COLORS[g % WANTED_COLORS.length]

export interface PairTag {
  kind: 'wanted' | 'unwanted'
  group: number
  /** 「同1」「別2」 */
  label: string
  /** この生徒について条件を満たしているか */
  ok: boolean
  color: PairColor
}

export interface PairGroupStatus {
  kind: 'wanted' | 'unwanted'
  group: number
  label: string
  members: number[]
  ok: boolean
  color: PairColor
}

/** 編成結果に対する、生徒ごとのペア指定タグとグループごとの達成状況 */
/** prefix: ラベルの接頭辞（既定は日本語の「同」「別」） */
export function pairStatus(p: Problem, classOf: number[], prefix: { wanted: string; unwanted: string } = { wanted: '同', unwanted: '別' }) {
  const tags: PairTag[][] = p.students.map(() => [])
  const groups: PairGroupStatus[] = []

  p.wantedGroups.forEach((g, gi) => {
    const ok = new Set(g.map((i) => classOf[i])).size <= 1
    const color = wantedColor(gi)
    const label = `${prefix.wanted}${gi + 1}`
    groups.push({ kind: 'wanted', group: gi, label, members: g, ok, color })
    for (const i of g) tags[i]?.push({ kind: 'wanted', group: gi, label, ok, color })
  })
  p.unwantedGroups.forEach((g, gi) => {
    const label = `${prefix.unwanted}${gi + 1}`
    let groupOk = true
    for (const i of g) {
      const ok = !g.some((j) => j !== i && classOf[j] === classOf[i])
      if (!ok) groupOk = false
      tags[i]?.push({ kind: 'unwanted', group: gi, label, ok, color: UNWANTED_COLOR })
    }
    groups.push({ kind: 'unwanted', group: gi, label, members: g, ok: groupOk, color: UNWANTED_COLOR })
  })
  return { tags, groups }
}

/** 行の背景色: 最初の同じ組グループの色（なければ null） */
export const rowColor = (tags: PairTag[]) => tags.find((t) => t.kind === 'wanted')?.color ?? null

/** 出力用テキスト: 「同1, 別2(×)」 */
export const tagText = (tags: PairTag[]) => tags.map((t) => `${t.label}${t.ok ? '' : '(×)'}`).join(', ')
