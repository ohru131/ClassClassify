import { COPY, type CopyKey } from './copy'
import { format, LANGUAGE_META, type AppLanguage } from './i18n'
import { formatDateTime, formatNumber } from './locale-format'
import { buildParseMessages } from './parse-messages'
import { FILE_LABELS, type FileLabels, type FileLanguage, type ParseMessages } from './solver'

/** 'device' = 端末の言語に合わせる（既定） */
export type LanguageChoice = 'device' | AppLanguage

// React に依存しない（テスト・印刷用 HTML の生成からも使う）
export type I18n = {
  lang: AppLanguage
  choice: LanguageChoice
  setChoice: (c: LanguageChoice) => void
  t: (key: CopyKey, params?: Record<string, string | number>) => string
  /** Excel の語彙（シート名・組の名前・ペア指定の接頭辞） */
  file: FileLabels
  fileLang: FileLanguage
  className: (c: number) => string
  num: (v: number, digits?: number) => string
  date: (d: Date) => string
  parseMessages: ParseMessages
}

export function makeI18n(lang: AppLanguage, choice: LanguageChoice = lang, setChoice: (c: LanguageChoice) => void = () => {}): I18n {
  const copy = COPY[lang]
  const file = FILE_LABELS[LANGUAGE_META[lang].file]
  return {
    lang,
    choice,
    setChoice,
    t: (key, params) => format(copy[key], params),
    file,
    fileLang: LANGUAGE_META[lang].file,
    className: file.className,
    num: (v, digits) => formatNumber(lang, v, digits),
    date: (d) => formatDateTime(lang, d),
    parseMessages: buildParseMessages(lang),
  }
}
