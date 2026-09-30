import { format, isAppLanguage, LANGUAGE_META, resolveLanguageTags, matchLanguageTag, type AppLanguage } from '../i18n/languages'
import { FILE_LABELS, JA_PARSE_MESSAGES, type FileLabels, type FileLanguage, type ParseMessages } from '../solver/labels'
import { WEB_COPY, type WebCopyKey } from './index'

// Web 版の言語まわりの純関数（React・DOM に依存しない。テストから使う）

export const LANGUAGE_STORAGE_KEY = 'mosaic.lang'

/** 既定の言語: URL の ?lang= → 保存済み → ブラウザの言語（navigator.languages）→ 英語 */
export function resolveInitialLanguage(search: string, stored: string | null, browser: readonly string[]): AppLanguage {
  const q = new URLSearchParams(search).get('lang')
  const fromQuery = matchLanguageTag(q)
  if (fromQuery) return fromQuery
  if (isAppLanguage(stored)) return stored
  return resolveLanguageTags(browser)
}

export type WebI18n = {
  lang: AppLanguage
  t: (key: WebCopyKey, params?: Record<string, string | number>) => string
  file: FileLabels
  fileLang: FileLanguage
  className: (c: number) => string
  num: (v: number, digits?: number) => string
  compare: (a: string, b: string) => number
  parseMessages: ParseMessages
}

export function makeWebI18n(lang: AppLanguage): WebI18n {
  const copy = WEB_COPY[lang]
  const fileLang = LANGUAGE_META[lang].file
  const file = FILE_LABELS[fileLang]
  const intl = LANGUAGE_META[lang].intl
  const t = (key: WebCopyKey, params?: Record<string, string | number>) => format(copy[key], params)
  return {
    lang,
    t,
    file,
    fileLang,
    className: file.className,
    num: (v, digits = 0) => {
      try {
        return new Intl.NumberFormat(intl, { minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: false }).format(v)
      } catch {
        return v.toFixed(digits)
      }
    },
    compare: (a, b) => a.localeCompare(b, intl),
    // 日本語は Web 版の従来の文言そのもの
    parseMessages:
      lang === 'ja'
        ? JA_PARSE_MESSAGES
        : {
            rosterMissing: t('parseRosterMissing', { sheet: file.sheets.roster }),
            noStudents: t('parseNoStudents'),
            noMissing: (row, name) => t('parseNoMissing', { sheet: file.sheets.roster, row, name }),
            duplicateNo: (no) => t('parseDuplicateNo', { no }),
            unknownNo: (sheet, row, no) => t('parseUnknownNo', { sheet, row, no }),
            classCountUnknown: (k) => t('parseClassCountUnknown', { sheet: file.sheets.settings, k }),
            maxTooSmall: (max, k) => t('parseMaxTooSmall', { max, k }),
          },
  }
}

/** サンプル・ひな形の URL（public/samples/<lang>/。`npm run samples:generate` の出力） */
export const sampleUrl = (lang: AppLanguage, id: 'sample1' | 'sample2' | 'sample-group') => `./samples/${lang}/${id}.xlsx`
export const templateZipUrl = (lang: AppLanguage) => `./samples/${lang}/template.zip`
