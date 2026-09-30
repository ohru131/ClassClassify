export const C = {
  bg: '#F6F7FB',
  card: '#FFFFFF',
  border: '#E2E8F0',
  text: '#0F172A',
  sub: '#475569',
  muted: '#94A3B8',
  primary: '#4F46E5',
  primarySoft: '#EEF2FF',
  primaryText: '#3730A3',
  danger: '#E11D48',
  dangerSoft: '#FFF1F2',
  good: '#059669',
  goodSoft: '#ECFDF5',
  warn: '#B45309',
  warnSoft: '#FFFBEB',
  hover: '#F1F5F9',
  focus: '#818CF8',
}

/** クラスごとの色（Web 版の classColor と同じ系統） */
const CLASS_COLORS = [
  { dot: '#6366F1', soft: '#EEF2FF', fg: '#4338CA' },
  { dot: '#D946EF', soft: '#FDF4FF', fg: '#A21CAF' },
  { dot: '#10B981', soft: '#ECFDF5', fg: '#047857' },
  { dot: '#F59E0B', soft: '#FFFBEB', fg: '#B45309' },
  { dot: '#0EA5E9', soft: '#F0F9FF', fg: '#0369A1' },
  { dot: '#F43F5E', soft: '#FFF1F2', fg: '#BE123C' },
  { dot: '#14B8A6', soft: '#F0FDFA', fg: '#0F766E' },
  { dot: '#8B5CF6', soft: '#F5F3FF', fg: '#6D28D9' },
]
export const classColor = (c: number) => CLASS_COLORS[c % CLASS_COLORS.length]
