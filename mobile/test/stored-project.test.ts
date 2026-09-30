import { describe, expect, it } from 'vitest'
import { loadSample, blankProblem, SAMPLES } from '../lib/samples'
import { isStoredProject } from '../lib/stored-project'
import { readFileSync } from 'node:fs'
import { SAMPLE_FILES } from '../lib/samples.generated'

describe('サンプル', () => {
  it('埋め込んだ base64 が public/ の Excel と一致する（npm run samples:generate し忘れの検出）', () => {
    for (const s of SAMPLE_FILES) {
      const b = readFileSync(new URL(`../../public/${s.file}`, import.meta.url))
      expect(s.base64).toBe(b.toString('base64'))
    }
  })
  it('全サンプルを読み込める', () => {
    for (const s of SAMPLES) {
      const { problem } = loadSample(s.id)
      expect(problem.students.length).toBeGreaterThan(0)
      expect(problem.numClasses).toBeGreaterThanOrEqual(2)
    }
  })
})

describe('保存データの検証', () => {
  const base = { version: 1, fileName: 'x', numClasses: 4, timeSec: 10 }
  it('JSON を往復しても通る', () => {
    const { problem } = loadSample('sample1')
    const classOf = problem.students.map((_, i) => i % 4)
    const data = JSON.parse(JSON.stringify({ ...base, problem, solution: { classOf, original: classOf, k: 4, iterations: 1, starts: 1 } }))
    expect(isStoredProject(data)).toBe(true)
    expect(isStoredProject({ ...base, problem: blankProblem(), solution: null })).toBe(true)
    expect(isStoredProject({ ...base, problem: null, solution: null })).toBe(true)
  })
  it('壊れたデータは捨てる', () => {
    const { problem } = loadSample('sample1')
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
    const { problem } = loadSample('sample1')
    const back = parseWorkbook(base64ToArrayBuffer(writeXlsx(rosterWorkbook(problem, 4), 'base64')))
    expect(back.students).toEqual(problem.students)
    expect(back.columns).toEqual(problem.columns)
    expect(back.wantedGroups).toEqual(problem.wantedGroups)
    expect(back.unwantedGroups).toEqual(problem.unwantedGroups)
  })
  it('結果のブックに各組のシートが入る', async () => {
    const { problem } = loadSample('sample-group')
    const res = await runSliced(compile(problem).compiled, { timeMs: 300, yieldToUi: async () => {} }).promise
    const report = evaluate(problem, res.classOf, problem.numClasses)
    const wb = resultWorkbook(problem, res.classOf, problem.numClasses, report)
    expect(wb.SheetNames).toEqual(expect.arrayContaining(['組分け', 'クラス別名簿', '1組', `${problem.numClasses}組`, 'ペア指定', '集計', '生徒名簿']))
    expect(writeXlsx(wb, 'base64').length).toBeGreaterThan(1000)
  })
})
