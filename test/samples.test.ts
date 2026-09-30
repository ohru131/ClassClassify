import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { parseWorkbook } from '../src/solver/parse'
import { compile } from '../src/solver/compile'
import { anneal } from '../src/solver/anneal'
import { evaluate } from '../src/solver/evaluate'

// public/samples/<lang>/*.xlsx（npm run samples:generate の出力）
const dir = new URL('../public/samples/', import.meta.url)
const LANGS = readdirSync(dir)
const load = (lang: string, f: string) => {
  const b = readFileSync(new URL(`${lang}/${f}`, dir))
  return parseWorkbook(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
}

describe('言語別サンプル', () => {
  it('6言語 × 3サンプルがそろっている', () => {
    expect(LANGS.sort()).toEqual(['de', 'en', 'es', 'ja', 'ko', 'pt-BR'])
    for (const l of LANGS) expect(readdirSync(new URL(`${l}/`, dir)).sort()).toEqual(['sample-group.xlsx', 'sample1.xlsx', 'sample2.xlsx', 'template.zip'])
  })

  describe.each(LANGS)('%s', (lang) => {
    it.each(['sample1.xlsx', 'sample2.xlsx', 'sample-group.xlsx'])('%s: 3種類の項目・ペア指定を含み、違反0・全項目が理想範囲内に届く', (f) => {
      const p = load(lang, f)
      expect(p.warnings).toEqual([])
      const kinds = new Set(p.columns.filter((c) => c.enabled).map((c) => c.kind))
      expect([...kinds].sort()).toEqual(['category', 'flag', 'numeric'])
      expect(p.unwantedGroups.length).toBeGreaterThan(0)
      if (f !== 'sample-group.xlsx') expect(p.wantedGroups.length).toBeGreaterThan(0)
      expect(new Set(p.students.map((s) => s.name)).size).toBe(p.students.length)
      const res = anneal(compile(p).compiled, { timeMs: 800, seed: 3 })
      const report = evaluate(p, res.classOf, p.numClasses)
      expect(report.violations).toEqual([])
      expect(report.totalExcess).toBe(0)
      expect(Math.max(...report.sizes) - Math.min(...report.sizes)).toBeLessThanOrEqual(1)
    })
  })
})
