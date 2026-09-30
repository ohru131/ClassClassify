import { describe, expect, it } from 'vitest'
import { COPY } from '../lib/copy'
import { EN_COPY, type CopyKey } from '../lib/copy/en'
import { PRIVACY } from '../lib/copy/privacy'
import { APP_LANGUAGES, format, LANGUAGE_META, resolveDeviceLanguage } from '../lib/i18n'
import { makeI18n } from '../lib/i18n-core'
import { runSliced } from '../lib/runner'
import { loadSample, samplesFor } from '../lib/samples'
import { compile, evaluate, parseWorkbook, resultWorkbook, rosterWorkbook, writeXlsx, FILE_LABELS } from '../lib/solver'

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()

describe('端末の言語の判定', () => {
  it('対応言語はそのまま、pt は pt-BR、未対応は英語', () => {
    expect(resolveDeviceLanguage([{ languageTag: 'ja-JP', languageCode: 'ja' }])).toBe('ja')
    expect(resolveDeviceLanguage([{ languageTag: 'ko-KR', languageCode: 'ko' }])).toBe('ko')
    expect(resolveDeviceLanguage([{ languageTag: 'es-CL', languageCode: 'es' }])).toBe('es')
    expect(resolveDeviceLanguage([{ languageTag: 'es-ES', languageCode: 'es' }])).toBe('es')
    expect(resolveDeviceLanguage([{ languageTag: 'de-AT', languageCode: 'de' }])).toBe('de')
    expect(resolveDeviceLanguage([{ languageTag: 'pt-PT', languageCode: 'pt' }])).toBe('pt-BR')
    expect(resolveDeviceLanguage([{ languageTag: 'en-AU', languageCode: 'en' }])).toBe('en')
    expect(resolveDeviceLanguage([{ languageTag: 'fr-FR', languageCode: 'fr' }])).toBe('en')
    // 優先言語の並びを順に見る
    expect(resolveDeviceLanguage([{ languageTag: 'fr-FR', languageCode: 'fr' }, { languageTag: 'de-DE', languageCode: 'de' }])).toBe('de')
    expect(resolveDeviceLanguage([])).toBe('en')
  })
})

describe('UI 文言', () => {
  it.each(APP_LANGUAGES)('%s: 全キーがあり、空でなく、埋め込み（{n} など）が英語と一致する', (lang) => {
    for (const key of Object.keys(EN_COPY) as CopyKey[]) {
      const v = COPY[lang][key]
      expect(v, `${lang}.${key}`).toBeTruthy()
      expect(placeholders(v), `${lang}.${key}`).toEqual(placeholders(EN_COPY[key]))
    }
  })
  it('日本語以外の文言に日本語が混ざらない', () => {
    for (const lang of APP_LANGUAGES.filter((l) => l !== 'ja'))
      for (const [k, v] of Object.entries(COPY[lang])) expect(v, `${lang}.${k}`).not.toMatch(/[ぁ-んァ-ヶ一-龥]/)
  })
  it('プライバシーポリシーは全言語で同じ構成', () => {
    for (const lang of APP_LANGUAGES) expect(PRIVACY[lang].map((s) => s.body.length)).toEqual(PRIVACY.ja.map((s) => s.body.length))
  })
  it('format は埋め込みを置き換える', () => {
    expect(format('{a} and {b}', { a: 1, b: 'x' })).toBe('1 and x')
    expect(format('{a}', {})).toBe('{a}')
  })
  it('数値・日付はロケールの書式', () => {
    expect(makeI18n('de').num(1.5, 1)).toBe('1,5')
    expect(makeI18n('en').num(1.5, 1)).toBe('1.5')
    expect(makeI18n('ko').date(new Date(2026, 1, 3, 9, 5))).toContain('2026')
    for (const l of APP_LANGUAGES) expect(LANGUAGE_META[l].endonym).toBeTruthy()
  })
})

describe.each(APP_LANGUAGES)('%s のサンプル', (lang) => {
  it.each(samplesFor(lang).map((s) => s.id))('%s: 条件違反0・人数差1以内に到達する', async (id) => {
    const { problem } = loadSample(lang, id)
    const res = await runSliced(compile(problem).compiled, { timeMs: 700, starts: 1, seed: 7, yieldToUi: async () => {} }).promise
    const report = evaluate(problem, res.classOf, problem.numClasses)
    expect(report.violations).toEqual([])
    expect(Math.max(...report.sizes) - Math.min(...report.sizes)).toBeLessThanOrEqual(1)
  })

  it('選択中の言語で書き出したファイルを読み戻せる（シート名・見出しがその言語）', async () => {
    const i18n = makeI18n(lang)
    const { problem } = loadSample(lang, 'sample-group')
    const rwb = rosterWorkbook(problem, problem.numClasses, i18n.fileLang)
    expect(rwb.SheetNames[1]).toBe(FILE_LABELS[i18n.fileLang].sheets.roster)
    expect(parseWorkbook(writeXlsx(rwb, 'array'), i18n.parseMessages).students).toEqual(problem.students)
    const res = await runSliced(compile(problem).compiled, { timeMs: 200, yieldToUi: async () => {} }).promise
    const wb = resultWorkbook(problem, res.classOf, problem.numClasses, evaluate(problem, res.classOf, problem.numClasses), i18n.fileLang)
    expect(wb.SheetNames).toContain(i18n.className(0))
  })

  it('読み込みのエラーはその言語で出る', () => {
    const i18n = makeI18n(lang)
    const empty = writeXlsx(rosterWorkbook({ ...loadSample(lang, 'sample1').problem, students: [], wantedGroups: [], unwantedGroups: [] }, 2, i18n.fileLang), 'array')
    expect(() => parseWorkbook(empty, i18n.parseMessages)).toThrow(i18n.parseMessages.rosterMissing)
  })
})
