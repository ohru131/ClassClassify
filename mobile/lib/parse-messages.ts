import { COPY } from './copy'
import { format, LANGUAGE_META, type AppLanguage } from './i18n'
import { FILE_LABELS, type ParseMessages } from './solver'

/** 読み込み時の警告・エラーを選択中の言語で出す（共有ソルバーの parseWorkbook に渡す） */
export function buildParseMessages(lang: AppLanguage): ParseMessages {
  const c = COPY[lang]
  const f = FILE_LABELS[LANGUAGE_META[lang].file]
  return {
    rosterMissing: format(c.parseRosterMissing, { sheet: f.sheets.roster }),
    noStudents: c.parseNoStudents,
    noMissing: (row, name) => format(c.parseNoMissing, { sheet: f.sheets.roster, row, name }),
    duplicateNo: (no) => format(c.parseDuplicateNo, { no }),
    unknownNo: (sheet, row, no) => format(c.parseUnknownNo, { sheet, row, no }),
    classCountUnknown: (k) => format(c.parseClassCountUnknown, { sheet: f.sheets.settings, k }),
    maxTooSmall: (max, k) => format(c.parseMaxTooSmall, { max, k }),
  }
}

