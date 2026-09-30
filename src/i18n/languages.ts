import type { FileLanguage } from '../solver/labels'

/**
 * 対応言語（Web 版・スマホ版で共通）。**この配列が唯一の情報源**で、型・言語の判定・言語名・
 * Intl のロケールをすべてここから導く。言語を足すときはここに足し、型エラーになった箇所
 * （両方の UI 文言・プライバシーポリシー・Excel の語彙 src/solver/labels.ts）を埋める。
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

/** 「ja-JP」「pt-PT」「es」のような言語タグ1つを対応言語にする（当たらなければ null） */
export function matchLanguageTag(tag: string | null | undefined): AppLanguage | null {
  if (!tag) return null
  if (isAppLanguage(tag)) return tag
  const code = tag.toLowerCase().split(/[-_]/)[0]
  // ポルトガル語はブラジル以外（pt-PT など）でも pt-BR を使う
  if (code === 'pt') return 'pt-BR'
  return APP_LANGUAGES.find((a) => a === code) ?? null
}

/** 優先順の言語タグの並びから最初に対応できるものを選ぶ。どれにも当たらなければ英語 */
export function resolveLanguageTags(tags: readonly (string | null | undefined)[]): AppLanguage {
  for (const t of tags) {
    const hit = matchLanguageTag(t)
    if (hit) return hit
  }
  return DEFAULT_LANGUAGE
}

/** 「{n} 名」のような埋め込みを置き換える */
export const format = (s: string, params?: Record<string, string | number>) =>
  params ? s.replace(/\{(\w+)\}/g, (m, k: string) => (k in params ? String(params[k]) : m)) : s
