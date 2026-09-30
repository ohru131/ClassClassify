import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import * as XLSX from 'xlsx-js-style'
import { parseWorkbook } from '../src/solver/parse'
import { compile } from '../src/solver/compile'
import { anneal } from '../src/solver/anneal'
import { evaluate } from '../src/solver/evaluate'
import { resultWorkbook, rosterWorkbook, writeXlsx } from '../src/solver/export'
import { FILE_LABELS, violationText, type FileLanguage } from '../src/solver/labels'

const load = (f: string) => {
  const b = readFileSync(new URL(`../public/${f}`, import.meta.url))
  return parseWorkbook(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
}
const LANGS = Object.keys(FILE_LABELS) as FileLanguage[]

describe('言語別のシート名・見出し', () => {
  it('既定（引数なし）の書き出しは従来どおり日本語', () => {
    const p = load('sample1.xlsx')
    expect(rosterWorkbook(p, 4).SheetNames).toEqual(['設定', '生徒名簿', '同じ組ペア', '別の組ペア'])
    const { compiled } = compile(p)
    const res = anneal(compiled, { timeMs: 200, seed: 1 })
    const wb = resultWorkbook(p, res.classOf, 4, evaluate(p, res.classOf, 4))
    expect(wb.SheetNames).toEqual(['組分け', 'クラス別名簿', '1組', '2組', '3組', '4組', 'ペア指定', '集計', '組み合わせ失敗', '生徒名簿'])
  })

  it.each(LANGS)('%s の名簿を書き出して読み戻すと同じ名簿になる', (lang) => {
    const p = load('sample1.xlsx')
    const wb = rosterWorkbook(p, 4, lang)
    const L = FILE_LABELS[lang]
    expect(wb.SheetNames).toEqual([L.sheets.settings, L.sheets.roster, L.sheets.wanted, L.sheets.unwanted])
    for (const n of wb.SheetNames) expect(n.length).toBeLessThanOrEqual(31)
    const q = parseWorkbook(writeXlsx(wb, 'array'))
    expect(q.students).toEqual(p.students)
    expect(q.columns).toEqual(p.columns)
    expect(q.numClasses).toBe(4)
    expect(q.maxPerClass).toBe(p.maxPerClass)
    expect(q.wantedGroups).toEqual(p.wantedGroups)
    expect(q.unwantedGroups).toEqual(p.unwantedGroups)
    expect(q.warnings).toEqual([])
  })

  it.each(LANGS)('%s の結果のブックはシート名が一意で、名簿として読み戻せる', (lang) => {
    const p = load('sample-group.xlsx')
    const { compiled } = compile(p)
    const res = anneal(compiled, { timeMs: 200, seed: 1 })
    const report = evaluate(p, res.classOf, p.numClasses)
    const wb = resultWorkbook(p, res.classOf, p.numClasses, report, lang)
    expect(new Set(wb.SheetNames).size).toBe(wb.SheetNames.length)
    for (const n of wb.SheetNames) expect(n).not.toMatch(/[:\\/?*[\]]/)
    expect(wb.SheetNames).toContain(FILE_LABELS[lang].className(0))
    expect(parseWorkbook(writeXlsx(wb, 'array')).students).toEqual(p.students)
  })

  it('手で作った英語のひな形（大文字小文字・見出しの揺れ）も読める', () => {
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['number of classes', 3], ['Max class size', 10]]), 'Settings')
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet([['', 'Weight', 2], ['NO', 'name', 'Girl'], [1, 'Ann', '✓'], [2, 'Bob', ''], [3, 'Cy', ''], [4, 'Di', '✓']]),
      'Roster',
    )
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([[1, 2]]), 'Keep together')
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([[3, 9]]), 'Keep apart')
    const q = parseWorkbook(writeXlsx(wb, 'array'))
    expect(q.numClasses).toBe(3)
    expect(q.maxPerClass).toBe(10)
    expect(q.students.map((s) => s.name)).toEqual(['Ann', 'Bob', 'Cy', 'Di'])
    expect(q.columns[0]).toMatchObject({ name: 'Girl', weight: 2, kind: 'flag' })
    expect(q.wantedGroups).toEqual([[0, 1]])
    // 名簿に無い NO（9）は警告（既定の日本語の文言）。別の組は1人になるので作られない
    expect(q.unwantedGroups).toEqual([])
    expect(q.warnings[0]).toContain('Keep apart')
  })

  it('違反の説明を言語別に作る（日本語は evaluate のメッセージのまま）', () => {
    const p = load('sample1.xlsx')
    const classOf = p.students.map(() => 0)
    classOf[p.wantedGroups[0][0]] = 1
    const report = evaluate(p, classOf, 4)
    const w = report.violations.find((v) => v.type === 'wanted')!
    const u = report.violations.find((v) => v.type === 'unwanted')!
    expect(violationText('ja', w, p, classOf)).toBe(w.message)
    expect(violationText('en', u, p, classOf)).toMatch(/^Keep apart: .* are both in Class 1$/)
    expect(violationText('de', w, p, classOf)).toMatch(/^Zusammen: /)
  })
})
