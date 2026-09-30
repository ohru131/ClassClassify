// 言語の定義・判定は Web 版と共通（リポジトリ直下の src/i18n/languages.ts）
import { resolveLanguageTags, type AppLanguage } from '../../src/i18n/languages'

export { APP_LANGUAGES, DEFAULT_LANGUAGE, format, isAppLanguage, LANGUAGE_META, matchLanguageTag, type AppLanguage } from '../../src/i18n/languages'

/** 端末の優先言語の並び（expo-localization の getLocales()）から最初に対応できるものを選ぶ */
export function resolveDeviceLanguage(locales: readonly { languageTag?: string | null; languageCode?: string | null }[]): AppLanguage {
  return resolveLanguageTags(locales.map((l) => l.languageTag || l.languageCode))
}
