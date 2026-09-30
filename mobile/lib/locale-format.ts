import type { AppLanguage } from './i18n'
import { LANGUAGE_META } from './i18n'

// 日付・数値は Intl でロケールの書式にする（Hermes の Intl が古い場合に備えて失敗時は素朴な書式）。
export function formatNumber(lang: AppLanguage, v: number, digits = 0): string {
  try {
    return new Intl.NumberFormat(LANGUAGE_META[lang].intl, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v)
  } catch {
    return v.toFixed(digits)
  }
}

export function formatDateTime(lang: AppLanguage, d: Date): string {
  try {
    return new Intl.DateTimeFormat(LANGUAGE_META[lang].intl, { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(d)
  } catch {
    const p = (n: number) => String(n).padStart(2, '0')
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
  }
}
