import type { FileLanguage } from './solver'

/**
 * 対応言語。**この配列が唯一の情報源**で、型・端末ロケールの判定・言語名・Intl のロケールを
 * すべてここから導く（既存アプリ UnitCalc の lib/i18n.ts と同じ方式）。言語を足すときは
 * ここに足し、型エラーになった箇所（UI 文言・プライバシーポリシー・Excel の語彙）を埋める。
 * 訳語は docs/i18n-glossary.md に揃える。
 */
export const APP_LANGUAGES = ['ja', 'en', 'ko', 'es', 'de', 'pt-BR'] as const
export type AppLanguage = (typeof APP_LANGUAGES)[number]

export const DEFAULT_LANGUAGE: AppLanguage = 'en'

export const LANGUAGE_META: Record<AppLanguage, { endonym: string; intl: string; file: FileLanguage }> = {
  ja: { endonym: '日本語', intl: 'ja-JP', file: 'ja' },
  en: { endonym: 'English', intl: 'en-US', file: 'en' },
  ko: { endonym: '한국어', intl: 'ko-KR', file: 'ko' },
  // 中南米の語彙を基本にする（Intl も es-419）
  es: { endonym: 'Español', intl: 'es-419', file: 'es' },
  de: { endonym: 'Deutsch', intl: 'de-DE', file: 'de' },
  'pt-BR': { endonym: 'Português (Brasil)', intl: 'pt-BR', file: 'pt-BR' },
}

export const isAppLanguage = (v: unknown): v is AppLanguage => typeof v === 'string' && (APP_LANGUAGES as readonly string[]).includes(v)

/**
 * 端末の優先言語の並び（expo-localization の getLocales()）から最初に対応できるものを選ぶ。
 * ポルトガル語はブラジル以外（pt-PT など）でも pt-BR を使う。どれにも当たらなければ英語。
 */
export function resolveDeviceLanguage(locales: readonly { languageTag?: string | null; languageCode?: string | null }[]): AppLanguage {
  for (const l of locales) {
    const tag = (l.languageTag ?? '').toLowerCase()
    const code = (l.languageCode ?? tag.split('-')[0] ?? '').toLowerCase()
    if (code === 'pt') return 'pt-BR'
    const hit = APP_LANGUAGES.find((a) => a === code)
    if (hit) return hit
  }
  return DEFAULT_LANGUAGE
}

/** 「{n} 名」のような埋め込みを置き換える */
export const format = (s: string, params?: Record<string, string | number>) =>
  params ? s.replace(/\{(\w+)\}/g, (m, k: string) => (k in params ? String(params[k]) : m)) : s
