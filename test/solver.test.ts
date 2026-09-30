import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { parseWorkbook } from '../src/solver/parse'
import { compile } from '../src/solver/compile'
import { anneal } from '../src/solver/anneal'
import { evaluate } from '../src/solver/evaluate'
import { exportWorkbook } from '../src/solver/export'

const load = (f: string) => {
  const b = readFileSync(new URL(`../public/${f}`, import.meta.url))
  return parseWorkbook(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
}

describe.each(['sample1.xlsx', 'sample2.xlsx', 'sample-group.xlsx'])('%s', (file) => {
  it('parses and solves without violations', () => {
    const p = load(file)
    expect(p.students.length).toBeGreaterThan(0)
    const { compiled } = compile(p)
    const res = anneal(compiled, { timeMs: 1500, seed: 42 })
    const report = evaluate(p, res.classOf, p.numClasses)
    console.log(file, p.students.length, p.numClasses, 'sizes', report.sizes, 'excess', report.totalExcess, 'cost', res.cost.toFixed(2), 'iters', res.iterations, p.warnings)
    expect(report.violations).toEqual([])
    expect(Math.max(...report.sizes) - Math.min(...report.sizes)).toBeLessThanOrEqual(1)
    expect(exportWorkbook(p, res.classOf, p.numClasses, report).size).toBeGreaterThan(0)
  })
})

describe('Google ひな形', () => {
  it('ひな形の内容が sample1.xlsx と同じ問題として読み込める', async () => {
    const XLSX = await import('xlsx')
    const { buildTemplateSheets } = await import('../src/google/template')
    const b = readFileSync(new URL('../public/sample1.xlsx', import.meta.url))
    const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
    const wb = XLSX.utils.book_new()
    for (const s of buildTemplateSheets(buf)) XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(s.rows), s.name)
    const out = XLSX.write(wb, { bookType: 'xlsx', type: 'array' }) as ArrayBuffer
    const a = parseWorkbook(buf)
    const t = parseWorkbook(out)
    expect(t.students).toEqual(a.students)
    expect(t.columns).toEqual(a.columns)
    expect(t.numClasses).toBe(a.numClasses)
    expect(t.wantedGroups).toEqual(a.wantedGroups)
    expect(t.unwantedGroups).toEqual(a.unwantedGroups)
  })
})
