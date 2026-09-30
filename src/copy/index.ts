import type { AppLanguage } from '../i18n/languages'
import { DE_COPY } from './de'
import { EN_COPY, type WebCopyKey } from './en'
import { ES_COPY } from './es'
import { JA_COPY } from './ja'
import { KO_COPY } from './ko'
import { PT_BR_COPY } from './pt-BR'

export type { WebCopyKey }

// キーが欠けた言語はここで型エラーになる（EN_COPY のキー集合が正）
export const WEB_COPY: Record<AppLanguage, Record<WebCopyKey, string>> = {
  en: EN_COPY,
  ja: JA_COPY,
  ko: KO_COPY,
  es: ES_COPY,
  de: DE_COPY,
  'pt-BR': PT_BR_COPY,
}
