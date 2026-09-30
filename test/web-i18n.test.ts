import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import XLSX from '../src/solver/xlsx'
import { WEB_COPY } from '../src/copy'
import { EN_COPY, type WebCopyKey } from '../src/copy/en'
import { makeWebI18n, resolveInitialLanguage } from '../src/copy/core'
import { APP_LANGUAGES, LANGUAGE_META, resolveLanguageTags } from '../src/i18n/languages'
import { JA_PARSE_MESSAGES } from '../src/solver/labels'
import { parseWorkbook } from '../src/solver/parse'
import { buildTemplateSheets } from '../src/google/template'

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()

describe('Web 版の言語', () => {
  it('?lang= → 保存済み → ブラウザの言語 → 英語 の順で決める', () => {
    expect(resolveInitialLanguage('?lang=ko', 'de', ['ja'])).toBe('ko')
    expect(resolveInitialLanguage('?lang=pt', null, [])).toBe('pt-BR')
    expect(resolveInitialLanguage('?lang=xx', 'de', ['ja'])).toBe('de')
    expect(resolveInitialLanguage('', 'es', ['ja-JP'])).toBe('es')
    expect(resolveInitialLanguage('', null, ['fr-FR', 'ja-JP'])).toBe('ja')
    expect(resolveInitialLanguage('', 'bogus', ['fr-FR'])).toBe('en')
    expect(resolveLanguageTags(['es-CL'])).toBe('es')
  })

  it.each(APP_LANGUAGES)('%s: 全キーがあり、空でなく、埋め込みが英語と一致する', (lang) => {
    for (const key of Object.keys(EN_COPY) as WebCopyKey[]) {
      const v = WEB_COPY[lang][key]
      expect(v, `${lang}.${key}`).toBeTruthy()
      expect(placeholders(v), `${lang}.${key}`).toEqual(placeholders(EN_COPY[key]))
    }
  })

  it('日本語以外の文言に日本語が混ざらない', () => {
    for (const lang of APP_LANGUAGES.filter((l) => l !== 'ja'))
      for (const [k, v] of Object.entries(WEB_COPY[lang])) expect(v, `${lang}.${k}`).not.toMatch(/[ぁ-んァ-ヶ一-龥]/)
  })

  it('日本語の読み込みメッセージは従来と同じもの', () => {
    expect(makeWebI18n('ja').parseMessages).toBe(JA_PARSE_MESSAGES)
    expect(makeWebI18n('ja').num(1.25, 1)).toBe((1.25).toFixed(1))
    expect(makeWebI18n('de').num(1.5, 1)).toBe('1,5')
  })
})

describe('Google ひな形（言語別）', () => {
  it.each(APP_LANGUAGES)('%s: ひな形のシートから同じ名簿として読み戻せる', (lang) => {
    const b = readFileSync(new URL(`../public/samples/${lang}/sample1.xlsx`, import.meta.url))
    const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
    const sheets = buildTemplateSheets(buf, lang)
    const wb = XLSX.utils.book_new()
    for (const s of sheets) XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(s.rows), s.name)
    const out = XLSX.write(wb, { bookType: 'xlsx', type: 'array' }) as ArrayBuffer
    const a = parseWorkbook(buf)
    const t = parseWorkbook(out)
    expect(t.students).toEqual(a.students)
    expect(t.wantedGroups).toEqual(a.wantedGroups)
    expect(t.numClasses).toBe(a.numClasses)
    // 使い方シートの言語
    if (lang !== 'ja') expect(JSON.stringify(sheets[0].rows)).not.toMatch(/[ぁ-んァ-ヶ一-龥]/)
    expect(LANGUAGE_META[lang].intl).toBeTruthy()
  })
})
