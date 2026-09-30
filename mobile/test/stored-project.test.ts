import { describe, expect, it } from 'vitest'
import { loadSample, blankProblem, samplesFor } from '../lib/samples'
import { APP_LANGUAGES } from '../lib/i18n'
import { isStoredProject } from '../lib/stored-project'
import { readFileSync } from 'node:fs'
import { SAMPLE_FILES } from '../lib/samples.generated'

describe('サンプル', () => {
  it('埋め込んだ base64 が public/samples/ の Excel と一致する（npm run samples:embed し忘れの検出）', () => {
    for (const lang of APP_LANGUAGES)
      for (const s of SAMPLE_FILES[lang]) {
        const b = readFileSync(new URL(`../../public/samples/${lang}/${s.id}.xlsx`, import.meta.url))
        expect(s.base64, `${lang}/${s.id}`).toBe(b.toString('base64'))
      }
  })
  it.each(APP_LANGUAGES)('%s: 全サンプルが警告なしで読め、3種類の項目とペア指定を含む', (lang) => {
    expect(samplesFor(lang).map((s) => s.id)).toEqual(['sample1', 'sample2', 'sample-group'])
    for (const s of samplesFor(lang)) {
      const { problem } = loadSample(lang, s.id)
      expect(problem.warnings).toEqual([])
      expect(new Set(problem.columns.map((c) => c.kind))).toEqual(new Set(['flag', 'category', 'numeric']))
      expect(problem.unwantedGroups.length).toBeGreaterThan(0)
      expect(new Set(problem.students.map((x) => x.name)).size).toBe(problem.students.length)
      if (lang !== 'ja') expect(JSON.stringify(problem)).not.toMatch(/[ぁ-んァ-ヶ一-龥○]/)
    }
  })
})

describe('保存データの検証', () => {
  const base = { version: 1, fileName: 'x', numClasses: 4, timeSec: 10 }
  it('JSON を往復しても通る', () => {
    const { problem } = loadSample('ja', 'sample1')
    const classOf = problem.students.map((_, i) => i % 4)
    const data = JSON.parse(JSON.stringify({ ...base, problem, solution: { classOf, original: classOf, k: 4, iterations: 1, starts: 1 } }))
    expect(isStoredProject(data)).toBe(true)
    expect(isStoredProject({ ...base, problem: blankProblem('en'), solution: null })).toBe(true)
    expect(isStoredProject({ ...base, problem: null, solution: null })).toBe(true)
  })
  it('壊れたデータは捨てる', () => {
    const { problem } = loadSample('ja', 'sample1')
    expect(isStoredProject(null)).toBe(false)
    expect(isStoredProject({ ...base, version: 2, problem, solution: null })).toBe(false)
    // 生徒数と結果の長さが合わない
    expect(isStoredProject({ ...base, problem, solution: { classOf: [0], original: [0], k: 4, iterations: 1, starts: 1 } })).toBe(false)
    // ペア指定が範囲外の生徒を指す
    expect(isStoredProject({ ...base, problem: { ...problem, wantedGroups: [[0, 9999]] }, solution: null })).toBe(false)
  })
})

describe('Excel の書き出し（スマホ版の経路: base64）', async () => {
  const { base64ToArrayBuffer } = await import('../lib/base64')
  const { compile, evaluate, parseWorkbook, resultWorkbook, rosterWorkbook, writeXlsx } = await import('../lib/solver')
  const { runSliced } = await import('../lib/runner')
  it('名簿を base64 で書き出して読み戻すと同じ名簿になる', () => {
    const { problem } = loadSample('ja', 'sample1')
    const back = parseWorkbook(base64ToArrayBuffer(writeXlsx(rosterWorkbook(problem, 4), 'base64')))
    expect(back.students).toEqual(problem.students)
    expect(back.columns).toEqual(problem.columns)
    expect(back.wantedGroups).toEqual(problem.wantedGroups)
    expect(back.unwantedGroups).toEqual(problem.unwantedGroups)
  })
  it('結果のブックに各組のシートが入る', async () => {
    const { problem } = loadSample('ja', 'sample-group')
    const res = await runSliced(compile(problem).compiled, { timeMs: 300, yieldToUi: async () => {} }).promise
    const report = evaluate(problem, res.classOf, problem.numClasses)
    const wb = resultWorkbook(problem, res.classOf, problem.numClasses, report)
    expect(wb.SheetNames).toEqual(expect.arrayContaining(['組分け', 'クラス別名簿', '1組', `${problem.numClasses}組`, 'ペア指定', '集計', '生徒名簿']))
    expect(writeXlsx(wb, 'base64').length).toBeGreaterThan(1000)
  })
})
