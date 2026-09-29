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
